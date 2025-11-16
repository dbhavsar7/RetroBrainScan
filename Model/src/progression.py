# src/progression.py

import os
from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from src.autoencoder import Autoencoder
from src.models import get_device


# --------------------------------------------------------
# Load Autoencoder for inference (not VAE)
# --------------------------------------------------------
def get_ae_for_inference(checkpoint_path: str):
    device = get_device()

    ckpt = torch.load(checkpoint_path, map_location=device)
    latent_dim = ckpt["latent_dim"]
    img_size = ckpt.get("img_size", 128)

    ae = Autoencoder(latent_dim=latent_dim).to(device)
    ae.load_state_dict(ckpt["model_state_dict"])
    ae.eval()

    return ae, latent_dim, img_size, device


# --------------------------------------------------------
# Compute latent progression vector
# CN → No Impairment
# AD → Moderate Impairment
# --------------------------------------------------------
def compute_progression_vector(
    train_dir: str = "data/raw/train",
    ae_checkpoint: str = "models/autoencoder.pth",
    save_path: str = "models/progression_vector.pt",
):
    ae, latent_dim, img_size, device = get_ae_for_inference(ae_checkpoint)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    class_names = dataset.classes

    print(f"📁 Classes found: {class_names}")

    cn_latents = []  # No Impairment
    ad_latents = []  # Moderate Impairment

    for imgs, labels in tqdm(loader, desc="Extracting latents"):
        imgs = imgs.to(device)

        # z = encoder(x)
        with torch.no_grad():
            _, z = ae(imgs)

        for i, lbl in enumerate(labels):
            cls = class_names[lbl]

            if "No" in cls:           # No Impairment
                cn_latents.append(z[i].cpu())

            elif "Moderate" in cls:   # Moderate Impairment
                ad_latents.append(z[i].cpu())

    cn_latents = torch.stack(cn_latents, dim=0)
    ad_latents = torch.stack(ad_latents, dim=0)

    z_cn = cn_latents.mean(dim=0)
    z_ad = ad_latents.mean(dim=0)

    progression_vector = z_ad - z_cn

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "z_cn": z_cn,
            "z_ad": z_ad,
            "progression_vector": progression_vector,
            "latent_dim": latent_dim,
            "img_size": img_size,
        },
        save_path,
    )

    print(f"✅ Saved Autoencoder progression vector → {save_path}")


if __name__ == "__main__":
    compute_progression_vector()
