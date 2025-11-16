# src/progression.py

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from src.autoencoder import Autoencoder
from src.models import get_device


def load_autoencoder(checkpoint_path):
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device)

    # ---- New format ----
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        latent_dim = ckpt.get("latent_dim", 64)
        img_size = ckpt.get("img_size", 128)
        state_dict = ckpt["model_state_dict"]

    # ---- Old format ----
    else:
        latent_dim = 64
        img_size = 128
        state_dict = ckpt

    ae = Autoencoder(latent_dim=latent_dim).to(device)
    ae.load_state_dict(state_dict)
    ae.eval()

    return ae, latent_dim, img_size, device


def compute_progression_vector(
    train_dir="data/raw/train",
    ae_checkpoint="models/autoencoder.pth",
    save_path="models/progression_vector.pt",
):
    ae, latent_dim, img_size, device = load_autoencoder(ae_checkpoint)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(train_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32)

    class_names = dataset.classes
    print("📁 Classes:", class_names)

    cn_latents = []
    ad_latents = []

    for imgs, lbls in tqdm(loader, desc="Extracting latents"):
        imgs = imgs.to(device)
        with torch.no_grad():
            _, z = ae(imgs)

        for i, label in enumerate(lbls):
            name = class_names[label]

            if "No" in name:
                cn_latents.append(z[i].cpu())

            if "Moderate" in name:
                ad_latents.append(z[i].cpu())

    cn_latents = torch.stack(cn_latents)
    ad_latents = torch.stack(ad_latents)

    z_cn = cn_latents.mean(0)
    z_ad = ad_latents.mean(0)

    progression_vector = z_ad - z_cn

    torch.save(
        {
            "progression_vector": progression_vector,
            "latent_dim": latent_dim,
            "img_size": img_size,
        },
        save_path,
    )

    print(f"✅ Saved progression vector → {save_path}")


if __name__ == "__main__":
    compute_progression_vector()
