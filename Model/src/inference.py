# src/inference.py

import os
from src.gradcam import generate_cam
from src.future_generator import generate_future_mri
from src.region_detector import extract_notable_regions
from src.models import AlzheimerResNet, get_device
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


# Utility — Classifier Risk Score
def load_classifier(checkpoint_path="models/resnet18_alzheimer.pth"):
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt["class_names"]

    model = AlzheimerResNet(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, class_names, device


def preprocess_for_classifier(gray_numpy, img_size):
    img = cv2.resize(gray_numpy, (img_size, img_size))
    img_rgb = np.stack([img, img, img], axis=-1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transform(Image.fromarray(img_rgb)).unsqueeze(0)


def get_risk_score(gray_numpy):
    model, class_names, device = load_classifier()
    img_size = 128

    x = preprocess_for_classifier(gray_numpy, img_size).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    no_idx = [i for i, c in enumerate(class_names) if "No" in c][0]
    risk = 1.0 - float(probs[no_idx])
    pred = class_names[int(np.argmax(probs))]

    return risk, pred


def copy_original_as_heatmap(img_path, output_path):
    """Copy original image as heatmap when risk score is 0"""
    orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if orig is None:
        raise ValueError(f"Could not load image from {img_path}")
    
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(output_path, orig_rgb)
    
    dummy_cam = np.zeros((128, 128), dtype=np.float32)
    return dummy_cam


# Main pipeline function
def analyze_brain_scan(
    img_path,
    classifier_path="models/resnet18_alzheimer.pth",
    autoencoder_path="models/autoencoder.pth",
    progression_path="models/progression_vector.pt",
    output_dir="outputs",
    alpha=0.5
):
    """Analyze a brain scan image and return all results"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    
    current_cam_path = os.path.join(output_dir, f"current_cam_{unique_id}.png")
    future_mri_path = os.path.join(output_dir, f"future_mri_{unique_id}.png")
    future_cam_path = os.path.join(output_dir, f"future_cam_{unique_id}.png")
    
    print(f"\n📥 Analyzing MRI: {img_path}")

    # -------------------------------------------------
    # 0. COMPUTE CURRENT RISK SCORE FIRST (for guardrail)
    # -------------------------------------------------
    print("\n📊 Computing current risk score...")
    original_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if original_gray is None:
        raise ValueError(f"Could not load image from {img_path}")
    
    curr_risk, curr_pred = get_risk_score(original_gray)
    print(f"🧠 CURRENT  → class={curr_pred}, risk={curr_risk:.3f}")

    # Current heatmap - guardrail: if risk score is very low (< 0.01), show original image
    # This prevents GradCAM from showing activations when there's no impairment
    if curr_risk < 0.01 or "No" in curr_pred:
        print("🛡️  Guardrail: Risk score is very low or No Impairment detected. Using original scan instead of heatmap...")
        cam_current = copy_original_as_heatmap(img_path, current_cam_path)
        print(f"✅ Saved original scan: {current_cam_path}")
    else:
        print("🔍 Generating current heatmap...")
        cam_current = generate_cam(
            img_path,
            checkpoint_path=classifier_path,
            output_path=current_cam_path
        )
        print(f"✅ Saved: {current_cam_path}")

    print("🧠 Extracting current brain regions...")
    regions_current = extract_notable_regions(cam_current)
    print(f"📍 Current regions: {regions_current}")

    print("⏳ Generating future MRI...")
    future_mri_path, future_cam_path, cam_future_temp = generate_future_mri(
        img_path,
        autoencoder_path=autoencoder_path,
        progression_path=progression_path,
        cam_model_path=classifier_path,
        output_path=future_mri_path,
        output_cam_path=future_cam_path,
        alpha=alpha
    )

    print("\n📊 Computing future risk score...")
    future_gray = cv2.imread(future_mri_path, cv2.IMREAD_GRAYSCALE)
    if future_gray is None:
        raise ValueError(f"Could not load future MRI from {future_mri_path}")
    
    futu_risk, futu_pred = get_risk_score(future_gray)
    print(f"🧠 FUTURE   → class={futu_pred}, risk={futu_risk:.3f}")

    # Guardrail: if future risk score is very low (< 0.01), show future MRI instead of heatmap
    if futu_risk < 0.01 or "No" in futu_pred:
        print("🛡️  Guardrail: Future risk score is very low or No Impairment detected. Using future MRI instead of heatmap...")
        cam_future = copy_original_as_heatmap(future_mri_path, future_cam_path)
        print(f"✅ Saved future MRI: {future_cam_path}")
    else:
        # Use the GradCAM that was already generated
        cam_future = cam_future_temp

    # Extract brain regions from future CAM array
    print("🧠 Extracting future brain regions...")
    regions_future = extract_notable_regions(cam_future)
    print(f"📍 Future regions: {regions_future}")

    # Return results as dictionary
    return {
        "current_risk_score": float(curr_risk),
        "current_prediction": curr_pred,
        "future_risk_score": float(futu_risk),
        "future_prediction": futu_pred,
        "current_heatmap_path": current_cam_path,
        "future_mri_path": future_mri_path,
        "future_heatmap_path": future_cam_path,
        "original_image_path": img_path,
        "current_regions": regions_current,
        "future_regions": regions_future
    }


def run_full_inference():
    img_path = "data/raw/test/Moderate Impairment/14.jpg"

    classifier_path = "models/resnet18_alzheimer.pth"
    autoencoder_path = "models/autoencoder.pth"
    progression_path = "models/progression_vector.pt"

    results = analyze_brain_scan(
        img_path,
        classifier_path=classifier_path,
        autoencoder_path=autoencoder_path,
        progression_path=progression_path
    )
    
    print("\n🎉 COMPLETED! RESULTS ARE READY:")
    print(f"• Current Heatmap    → {results['current_heatmap_path']}")
    print(f"• Future MRI         → {results['future_mri_path']}")
    print(f"• Future Heatmap     → {results['future_heatmap_path']}")
    print(f"• Current Risk Score → {results['current_risk_score']:.3f} ({results['current_prediction']})")
    print(f"• Future Risk Score  → {results['future_risk_score']:.3f} ({results['future_prediction']})")


if __name__ == "__main__":
    run_full_inference()
