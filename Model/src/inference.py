# src/inference.py

import os
from src.gradcam import generate_cam
from src.future_generator import generate_future_mri
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


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
def run_full_inference():
    # 🔥 Set your MRI path here
    img_path = "data/raw/test/Moderate Impairment/14.jpg"

    classifier_path = "models/resnet18_alzheimer.pth"
    autoencoder_path = "models/autoencoder.pth"
    progression_path = "models/progression_vector.pt"

    print(f"\n📥 Using input MRI: {img_path}")

    # -------------------------------------------------
    # 1. CURRENT HEATMAP (classifier → GradCAM)
    # -------------------------------------------------
    print("🔍 Generating current heatmap...")
    generate_cam(
        img_path,
        checkpoint_path=classifier_path,
        output_path="outputs/current_cam.png"
    )
    print("✅ Saved: outputs/current_cam.png")

    # -------------------------------------------------
    # 2. FUTURE MRI + FUTURE HEATMAP
    # -------------------------------------------------
    print("⏳ Generating future MRI + heatmap...")
    future_mri_path, future_cam_path = generate_future_mri(
        img_path,
        autoencoder_path=autoencoder_path,
        progression_path=progression_path,
        cam_model_path=classifier_path,
        output_path="outputs/future_mri.png",
        output_cam_path="outputs/future_cam.png"
    )

    # -------------------------------------------------
    # 3. RISK SCORE (Optional but nice for UI)
    # -------------------------------------------------
    print("\n📊 Computing risk scores...")

    original_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    future_gray = cv2.imread(future_mri_path, cv2.IMREAD_GRAYSCALE)

    curr_risk, curr_pred = get_risk_score(original_gray)
    futu_risk, futu_pred = get_risk_score(future_gray)

    print(f"🧠 CURRENT  → class={curr_pred}, risk={curr_risk:.3f}")
    print(f"🧠 FUTURE   → class={futu_pred}, risk={futu_risk:.3f}")

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    print("\n🎉 COMPLETED! RESULTS ARE READY:")
    print("• Current Heatmap    → outputs/current_cam.png")
    print("• Future MRI         → outputs/future_mri.png")
    print("• Future Heatmap     → outputs/future_cam.png")
    print(f"• Current Risk Score → {curr_risk:.3f} ({curr_pred})")
    print(f"• Future Risk Score  → {futu_risk:.3f} ({futu_pred})")


# -------------------------------------------------
# Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    run_full_inference()
