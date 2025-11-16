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


# -------------------------------------------------
# Utility — Classifier Risk Score
# -------------------------------------------------
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

    # find "No Impairment"
    no_idx = [i for i, c in enumerate(class_names) if "No" in c][0]
    risk = 1.0 - float(probs[no_idx])
    pred = class_names[int(np.argmax(probs))]

    return risk, pred


def copy_original_as_heatmap(img_path, output_path):
    """
    Copy the original image (processed to match GradCAM format) to output path.
    This is used as a guardrail when risk score is 0 (no impairment detected).
    
    Args:
        img_path: Path to the original image
        output_path: Path where the processed original image should be saved
    
    Returns:
        numpy array: A dummy CAM array (zeros) for compatibility with region detection
    """
    # Load original MRI
    orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if orig is None:
        raise ValueError(f"Could not load image from {img_path}")
    
    # Convert to RGB (same format as GradCAM overlay output)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
    
    # Save the original image (no heatmap overlay)
    cv2.imwrite(output_path, orig_rgb)
    
    # Return a dummy CAM array (zeros) for compatibility with region detection
    # Since there's no impairment, there are no notable regions
    dummy_cam = np.zeros((128, 128), dtype=np.float32)
    
    return dummy_cam


# -------------------------------------------------
# MAIN PIPELINE - Reusable function for Flask
# -------------------------------------------------
def analyze_brain_scan(
    img_path,
    classifier_path="models/resnet18_alzheimer.pth",
    autoencoder_path="models/autoencoder.pth",
    progression_path="models/progression_vector.pt",
    output_dir="outputs",
    alpha=0.5
):
    """
    Analyze a brain scan image and return all results.
    
    Args:
        img_path: Path to the input brain scan image
        classifier_path: Path to classifier checkpoint
        autoencoder_path: Path to autoencoder checkpoint
        progression_path: Path to progression vector
        output_dir: Directory to save output images
        alpha: Progression factor (default 0.5)
    
    Returns:
        dict: Results containing risk scores, predictions, and image paths
    """
    import os
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filenames to avoid conflicts
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

    # -------------------------------------------------
    # 1. CURRENT HEATMAP (classifier → GradCAM)
    # Guardrail: If risk score is 0, show original image instead
    # -------------------------------------------------
    if curr_risk == 0.0:
        print("🛡️  Guardrail: Risk score is 0. Using original scan instead of heatmap...")
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

    # Extract brain regions from current CAM array
    print("🧠 Extracting current brain regions...")
    regions_current = extract_notable_regions(cam_current)
    print(f"📍 Current regions: {regions_current}")

    # -------------------------------------------------
    # 2. FUTURE MRI + FUTURE HEATMAP
    # -------------------------------------------------
    print("⏳ Generating future MRI...")
    future_mri_path, future_cam_path, cam_future = generate_future_mri(
        img_path,
        autoencoder_path=autoencoder_path,
        progression_path=progression_path,
        cam_model_path=classifier_path,
        output_path=future_mri_path,
        output_cam_path=future_cam_path,
        alpha=alpha
    )

    # -------------------------------------------------
    # 3. COMPUTE FUTURE RISK SCORE (for guardrail)
    # -------------------------------------------------
    print("\n📊 Computing future risk score...")
    future_gray = cv2.imread(future_mri_path, cv2.IMREAD_GRAYSCALE)
    if future_gray is None:
        raise ValueError(f"Could not load future MRI from {future_mri_path}")
    
    futu_risk, futu_pred = get_risk_score(future_gray)
    print(f"🧠 FUTURE   → class={futu_pred}, risk={futu_risk:.3f}")

    # Guardrail: If future risk score is 0, show future MRI instead of heatmap
    if futu_risk == 0.0:
        print("🛡️  Guardrail: Future risk score is 0. Using future MRI instead of heatmap...")
        cam_future = copy_original_as_heatmap(future_mri_path, future_cam_path)
        print(f"✅ Saved future MRI: {future_cam_path}")
    # Note: If future risk > 0, the heatmap was already generated by generate_future_mri()

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
    # 🔥 Set your MRI path here
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
    
    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    print("\n🎉 COMPLETED! RESULTS ARE READY:")
    print(f"• Current Heatmap    → {results['current_heatmap_path']}")
    print(f"• Future MRI         → {results['future_mri_path']}")
    print(f"• Future Heatmap     → {results['future_heatmap_path']}")
    print(f"• Current Risk Score → {results['current_risk_score']:.3f} ({results['current_prediction']})")
    print(f"• Future Risk Score  → {results['future_risk_score']:.3f} ({results['future_prediction']})")


# -------------------------------------------------
# Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    run_full_inference()
