# src/train_vae.py

from src.vae import train_vae

if __name__ == "__main__":
    train_vae(
        train_dir="data/raw/train",
        latent_dim=128,
        img_size=128,
        batch_size=64,   # reduce to 32 if memory issues
        num_epochs=30,   # you can shorten to 15 if time is tight
        lr=1e-3,
        beta=0.1,
        checkpoint_path="models/vae_conv.pth",
    )
