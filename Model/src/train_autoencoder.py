# src/train_autoencoder.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from src.autoencoder import Autoencoder
from src.models import get_device


def train_autoencoder(
    train_dir="data/raw/train",
    latent_dim=64,
    img_size=128,
    epochs=20,
    lr=1e-3,
    out_path="models/autoencoder.pth",
):
    device = get_device()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    ae = Autoencoder(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("🚀 Training Autoencoder...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for imgs, _ in loader:
            imgs = imgs.to(device)

            recon, z = ae(imgs)
            loss = criterion(recon, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

    # ---- SAVE CORRECT CHECKPOINT FORMAT ----
    ckpt = {
        "model_state_dict": ae.state_dict(),
        "latent_dim": latent_dim,
        "img_size": img_size,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"✅ Autoencoder saved to {out_path}")
    print("🔑 Saved keys:", ckpt.keys())


if __name__ == "__main__":
    train_autoencoder()
