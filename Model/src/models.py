# src/models.py
from typing import List

import torch
import torch.nn as nn
from torchvision import models


class AlzheimerResNet(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = False):
        super().__init__()
        # Pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Optionally freeze feature extractor
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final fully-connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def get_device():
    """
    Use MPS on Apple Silicon if available, otherwise CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
