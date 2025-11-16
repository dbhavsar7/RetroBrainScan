# src/vae.py

import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from src.models import get_device


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 1 x 128 x 128  -> 256 x 8 x 8
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.enc_out_dim = 256 * 8 * 8

        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        # Decoder: latent_dim -> 256 x 8 x 8 -> 1 x 128 x 128
        self.fc_dec = nn.Linear(latent_dim, self.enc_out_dim)

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # 128x128
            nn.Sigmoid(),  # output in [0,1]
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_conv(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(z.size(0), 256, 8, 8)
        x_rec = self.dec_conv(h)
        return x_rec

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar


def vae_loss_fn(x, x_rec, mu, logvar, beta: float = 1.0):
    # Reconstruction loss (MSE) + KL divergence
    recon_loss = F.mse_loss(x_rec, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


def get_vae_dataloader(
    train_dir: str,
    img_size: int = 128,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    """
    Unsupervised: we only use images, labels are ignored.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # [0,1]
    ])

    dataset = datasets.ImageFolder(root=train_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def train_vae(
    train_dir: str = "data/raw/train",
    latent_dim: int = 128,
    img_size: int = 128,
    batch_size: int = 64,
    num_epochs: int = 30,
    lr: float = 1e-3,
    beta: float = 1.0,
    checkpoint_path: str = "models/vae_conv.pth",
):
    device = get_device()
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    loader = get_vae_dataloader(
        train_dir=train_dir,
        img_size=img_size,
        batch_size=batch_size,
    )

    model = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting VAE training on", device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for x, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            x = x.to(device)

            x_rec, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss_fn(x, x_rec, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_recon += recon_loss.item() * x.size(0)
            total_kl += kl_loss.item() * x.size(0)

        n = len(loader.dataset)
        print(
            f"Epoch {epoch+1}: "
            f"loss={total_loss/n:.4f}, recon={total_recon/n:.4f}, kl={total_kl/n:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "latent_dim": latent_dim,
            "img_size": img_size,
        },
        checkpoint_path,
    )
    print(f"✅ VAE saved to {checkpoint_path}")
