# src/future_generator.py

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from src.autoencoder import Autoencoder
from src.gradcam import generate_cam
from src.models import get_device


def load_autoencoder(path):
    device = get_device()
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        latent_dim = ckpt.get("latent_dim", 64)
        state_dict = ckpt["model_state_dict"]
    else:
        latent_dim = 64
        state_dict = ckpt

    ae = Autoencoder(latent_dim).to(device)
    ae.load_state_dict(state_dict)
    ae.eval()

    return ae, latent_dim, device


def load_image_gray(img_path, img_size=128):
    img = Image.open(img_path).convert("L")
    img = img.resize((img_size, img_size))
    return transforms.ToTensor()(img).unsqueeze(0)


def generate_future_mri(
    img_path,
    autoencoder_path="models/autoencoder.pth",
    progression_path="models/progression_vector.pt",
    cam_model_path="models/resnet18_alzheimer.pth",
    output_path="outputs/future_mri.png",
    output_cam_path="outputs/future_cam.png",
    alpha=0.5,
):

    # Load autoencoder
    ae, latent_dim, device = load_autoencoder(autoencoder_path)

    # Load progression vector
    prog = torch.load(progression_path, map_location=device)
    progression_vector = prog["progression_vector"].to(device)

    # Load grayscale image
    img = load_image_gray(img_path, 128).to(device)

    # Encode
    _, z = ae(img)

    # Latent shift
    z_future = z + alpha * progression_vector.unsqueeze(0)

    # Decode
    future = ae.decoder(z_future).detach().cpu().numpy()[0][0]
    future_uint8 = (future * 255).astype(np.uint8)

    cv2.imwrite(output_path, future_uint8)

    # CAM for future image
    cv2.imwrite("temp_future.png", future_uint8)
    cam_future = generate_cam("temp_future.png", checkpoint_path=cam_model_path,
                              output_path=output_cam_path)

    return output_path, output_cam_path, cam_future
