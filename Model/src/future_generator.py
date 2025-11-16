import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from src.autoencoder import Autoencoder
from src.gradcam import generate_cam
from src.models import get_device


# --------------------------
# Load grayscale MRI for Autoencoder
# --------------------------
def load_image_gray(img_path, img_size=128):
    img = Image.open(img_path).convert("L")
    img = img.resize((img_size, img_size))
    tensor = transforms.ToTensor()(img)  # [1, H, W]
    return tensor.unsqueeze(0)           # [1, 1, H, W]


# --------------------------
# Future MRI Generator (Latent-Space AD Progression)
# --------------------------
def generate_future_mri(
    img_path,
    autoencoder_path="models/autoencoder.pth",
    progression_path="models/progression_vector.pt",
    cam_model_path="models/resnet18_alzheimer.pth",
    alpha=0.30,
    img_size=128,
    output_path="outputs/future_mri.png",
    output_cam_path="outputs/future_cam.png"
):
    device = get_device()

    # --------------------------
    # 1. LOAD AUTOENCODER
    # --------------------------
    ae = Autoencoder(latent_dim=64).to(device)
    ae.load_state_dict(torch.load(autoencoder_path, map_location=device))
    ae.eval()

    # --------------------------
    # 2. LOAD PROGRESSION VECTOR
    # --------------------------
    prog = torch.load(progression_path, map_location=device)
    progression_vector = prog["progression_vector"].to(device)  # [64]

    # --------------------------
    # 3. LOAD GRAYSCALE INPUT MRI
    # --------------------------
    img_gray = load_image_gray(img_path, img_size).to(device)
    _, z = ae(img_gray)            # z: (1, latent_dim)

    # --------------------------
    # 4. COMPUTE FUTURE LATENT STATE
    # --------------------------
    z_future = z + alpha * progression_vector.unsqueeze(0)

    # --------------------------
    # 5. DECODE FUTURE MRI
    # --------------------------
    future = ae.decoder(z_future).detach().cpu().numpy()[0][0]

    future_uint8 = (future * 255).astype(np.uint8)
    cv2.imwrite(output_path, future_uint8)

    # --------------------------
    # 6. GENERATE FUTURE HEATMAP (GradCAM)
    # --------------------------
    temp_path = "temp_future.png"
    cv2.imwrite(temp_path, future_uint8)

    generate_cam(
        img_path=temp_path,
        checkpoint_path=cam_model_path,
        output_path=output_cam_path
    )

    return output_path, output_cam_path
