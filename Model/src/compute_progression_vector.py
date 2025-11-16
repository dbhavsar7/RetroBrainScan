# src/compute_progression_vector.py

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.autoencoder import Autoencoder
from src.models import get_device

# ---------------------------
# Load single MRI as grayscale tensor
# ---------------------------
def load_mri(img_path, img_size=128):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert("L")
    return transform(img).unsqueeze(0)  # (1,1,H,W)


# ---------------------------
# Compute progression vector
# ---------------------------
def compute_progression_vector(
    healthy_dir="data/raw/train/No Impairment",
    demented_dir="data/raw/train/Moderate Impairment",
    autoencoder_path="models/autoencoder.pth",
    latent_dim=64,
    img_size=128,
    output_path="models/progression_vector.pt",
):
    device = get_device()

    # Load AE with backward compatibility
    ckpt = torch.load(autoencoder_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        latent_dim = ckpt.get("latent_dim", latent_dim)
        img_size = ckpt.get("img_size", img_size)
        state_dict = ckpt["model_state_dict"]
    else:
        # Backward compatibility: old format was just state_dict
        state_dict = ckpt
    
    ae = Autoencoder(latent_dim=latent_dim).to(device)
    ae.load_state_dict(state_dict)
    ae.eval()

    def collect_latents(folder):
        Z = []
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".png")):
                img = load_mri(os.path.join(folder, fname), img_size).to(device)
                with torch.no_grad():
                    _, z = ae(img)         # encoder returns (recon, latent)
                Z.append(z.squeeze(0).cpu().numpy())
        return np.array(Z)

    print("Extracting healthy/... latents...")
    Z_healthy = collect_latents(healthy_dir)

    print("Extracting demented/... latents...")
    Z_demented = collect_latents(demented_dir)

    print("Computing progression vector...")
    mu_healthy = Z_healthy.mean(axis=0)
    mu_demented = Z_demented.mean(axis=0)

    progression_vector = mu_demented - mu_healthy
    progression_vector = torch.tensor(progression_vector, dtype=torch.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({"progression_vector": progression_vector}, output_path)

    print(f"Saved progression vector to {output_path}")


if __name__ == "__main__":
    compute_progression_vector()
