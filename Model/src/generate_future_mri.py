# src/generate_future_mri.py

import os
from typing import Tuple

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from src.vae import ConvVAE
from src.models import AlzheimerResNet, get_device
from src.gradcam import generate_cam


def load_vae_and_prog(
    vae_checkpoint: str = "models/vae_conv.pth",
    prog_path: str = "models/progression_vector.pt",
):
    device = get_device()
    ckpt = torch.load(vae_checkpoint, map_location=device)
    latent_dim = ckpt["latent_dim"]
    img_size = ckpt.get("img_size", 128)

    vae = ConvVAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()

    prog = torch.load(prog_path, map_location=device)
    progression_vector = prog["progression_vector"].to(device)

    return vae, progression_vector, img_size, device


def load_mri_for_vae(img_path: str, img_size: int = 128) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert("L")
    x = transform(img).unsqueeze(0)  # [1,1,H,W]
    return x


def tensor_to_np_image(x: torch.Tensor) -> np.ndarray:
    """
    x: [1, 1, H, W], values in [0,1]
    returns uint8 HxW
    """
    x = x.detach().cpu().squeeze(0).squeeze(0).numpy()
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255).astype(np.uint8)
    return x


def load_classifier(
    checkpoint_path: str = "models/resnet18_alzheimer.pth",
):
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", 128)

    model = AlzheimerResNet(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names, img_size, device


def prep_for_classifier(img_gray: np.ndarray, img_size: int) -> torch.Tensor:
    """
    Convert grayscale MRI to the RGB+normalized format the classifier expects.
    """
    img_resized = cv2.resize(img_gray, (img_size, img_size))
    img_rgb = np.stack([img_resized, img_resized, img_resized], axis=-1)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0)
    return tensor


def get_risk_score(img_gray: np.ndarray) -> Tuple[float, str]:
    """
    Compute risk score using trained classifier.
    Risk score = 1 - P(No Impairment).
    """
    model, class_names, img_size, device = load_classifier()
    x = prep_for_classifier(img_gray, img_size).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # find index of "No Impairment" class
    no_idx = [i for i, c in enumerate(class_names) if "No" in c][0]
    risk = 1.0 - float(probs[no_idx])
    pred_class = class_names[int(probs.argmax())]
    return risk, pred_class


def generate_future_mri(
    img_path: str,
    alpha: float = 0.5,
    vae_checkpoint: str = "models/vae_conv.pth",
    prog_path: str = "models/progression_vector.pt",
    out_current: str = "outputs/current_mri.png",
    out_future: str = "outputs/future_mri.png",
):
    vae, progression_vector, img_size, device = load_vae_and_prog(
        vae_checkpoint, prog_path
    )

    os.makedirs(os.path.dirname(out_current), exist_ok=True)

    # Load for VAE
    x = load_mri_for_vae(img_path, img_size).to(device)

    with torch.no_grad():
        mu, logvar = vae.encode(x)
        z_current = mu  # use mean
        z_future = z_current + alpha * progression_vector.unsqueeze(0)

        x_rec = vae.decode(z_current)
        x_future = vae.decode(z_future)

    img_current = tensor_to_np_image(x_rec)
    img_future = tensor_to_np_image(x_future)

    cv2.imwrite(out_current, img_current)
    cv2.imwrite(out_future, img_future)

    # Risk scores using classifier
    risk_current, class_curr = get_risk_score(img_current)
    risk_future, class_fut = get_risk_score(img_future)

    print(f"Saved current recon to {out_current}")
    print(f"Saved future MRI to {out_future}")
    print(f"Current: risk={risk_current:.3f}, class={class_curr}")
    print(f"Future:  risk={risk_future:.3f}, class={class_fut}")

    # --- Grad-CAM on current & future synthetic MRIs ---
    current_cam_path = out_current.replace(".png", "_cam.png")
    future_cam_path = out_future.replace(".png", "_cam.png")
    
    generate_cam(
        img_path=out_current,
        output_path=current_cam_path,
    )
    generate_cam(
        img_path=out_future,
        output_path=future_cam_path,
    )
    
    print(f"Saved current CAM to {current_cam_path}")
    print(f"Saved future CAM to {future_cam_path}")


if __name__ == "__main__":
    # Example usage
    test_img = "data/raw/test/Moderate Impairment/14.jpg"
    generate_future_mri(
        img_path=test_img,
        alpha=0.5,
        out_current="outputs/current_example_2.png",
        out_future="outputs/future_example_2.png",
    )
