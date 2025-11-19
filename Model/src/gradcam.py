# src/gradcam.py

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from src.models import AlzheimerResNet, get_device


# GradCAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        loss = logits[:, class_idx]

        self.model.zero_grad()
        loss.backward()

        grads = self.gradients            # [1, C, H, W]
        activations = self.activations    # [1, C, H, W]

        # Global average pooling of gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * activations).sum(dim=1)         # [1, H, W]

        cam = F.relu(cam)
        cam = cam[0].cpu().numpy()

        # Normalize to [0, 1]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam


# Image loading (grayscale + crop + RGB convert)
def load_image(img_path, img_size=128):
    """
    1. Force grayscale
    2. Crop black borders
    3. Resize to img_size
    4. Convert grayscale → 3-channel (RGB)
    5. Normalize for ResNet
    """
    img = Image.open(img_path).convert("L")

    # Crop black borders
    np_img = np.array(img)
    coords = cv2.findNonZero((np_img > 10).astype(np.uint8))

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = np_img[y:y+h, x:x+w]
    else:
        cropped = np_img

    # Resize
    cropped = cv2.resize(cropped, (img_size, img_size)).astype(np.uint8)

    # Expand grayscale → RGB
    img_rgb = np.stack([cropped, cropped, cropped], axis=-1)
    img_rgb = Image.fromarray(img_rgb)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(img_rgb).unsqueeze(0)


# Overlay heatmap
def overlay_heatmap(img_path, heatmap, output_path, alpha=0.45):
    """
    1. Resize CAM to MRI shape
    2. Mask to brain only
    3. Keep top 20% activations
    4. Gaussian smooth
    5. Blend with MRI grayscale
    """
    # Load original MRI
    orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)

    # Resize heatmap to MRI resolution
    heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

    # Mask to brain region only
    brain_mask = (orig > 10).astype(np.uint8)
    heatmap_resized = heatmap_resized * brain_mask

    # Keep only top 20% activations
    if np.any(heatmap_resized > 0):
        thresh = np.percentile(heatmap_resized[heatmap_resized > 0], 80)
        heatmap_resized = np.where(heatmap_resized >= thresh, heatmap_resized, 0)

    # Smooth CAM for better visualization
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (7, 7), sigmaX=2)

    # Convert CAM to RGB heatmap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Create 3-channel mask for blending
    mask = np.repeat(brain_mask[:, :, None], 3, axis=2)  # HxWx3

    # Keep heatmap only where CAM > 0 (inside brain region)
    heatmap_color = heatmap_color * mask

    # Blend ONLY inside brain region, preserve original black background
    blended = orig_rgb.copy()
    blended[mask == 1] = (
        alpha * heatmap_color[mask == 1]
        + (1 - alpha) * orig_rgb[mask == 1]
    )

    cv2.imwrite(output_path, blended)


# Get CAM array only (no saving)
def get_cam_array(
    img_path,
    checkpoint_path="models/resnet18_alzheimer.pth",
    img_size=128,
):
    """Returns CAM array without saving overlay."""
    device = get_device()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = len(checkpoint["class_names"])
    
    model = AlzheimerResNet(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    target_layer = model.backbone.layer3[-1]
    cam_generator = GradCAM(model, target_layer)
    
    input_tensor = load_image(img_path, img_size).to(device)
    cam = cam_generator(input_tensor)
    
    return cam


# Main entrypoint
def generate_cam(
    img_path,
    checkpoint_path="models/resnet18_alzheimer.pth",
    img_size=128,
    output_path="outputs/heatmap.png",
):
    device = get_device()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load trained model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = len(checkpoint["class_names"])

    model = AlzheimerResNet(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Choose layer with good spatial resolution
    target_layer = model.backbone.layer3[-1]  # preferred over layer4

    cam_generator = GradCAM(model, target_layer)

    # Prepare input
    input_tensor = load_image(img_path, img_size).to(device)

    # Compute CAM
    cam = cam_generator(input_tensor)

    # Save overlay image
    overlay_heatmap(img_path, cam, output_path)

    print(f"🔥 Saved heatmap to: {output_path}")
    
    # Return CAM array for reuse (e.g., region detection)
    return cam