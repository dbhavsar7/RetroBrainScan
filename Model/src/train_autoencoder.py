import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.autoencoder import Autoencoder
from src.models import get_device
import os

def train_autoencoder(
    data_dir="data/raw/train",
    epochs=25,
    batch_size=32,
    latent_dim=64,
    lr=1e-3,
    model_path="models/autoencoder.pth"
):

    device = get_device()
    print(f"Training Autoencoder on device: {device}")

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for img, _ in loader:
            img = img.to(device)

            recon, _ = model(img)
            loss = criterion(recon, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} — Loss: {avg:.5f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Autoencoder saved to {model_path}")


if __name__ == "__main__":
    train_autoencoder()
