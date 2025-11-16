# src/data.py
import os
from typing import Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets


class TransformDataset(Dataset):
    """Wrapper to apply different transforms to a subset of a dataset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


# You can tweak this if your paths differ
def get_default_data_transforms(img_size: int = 128):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    img_size: int = 128,
    batch_size: int = 32,
    val_split: float = 0.15,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Creates train/val/test dataloaders using torchvision.datasets.ImageFolder.
    Assumes:
      train_dir/
        No Impairment/
        Very Mild Impairment/
        Mild Impairment/
        Moderate Impairment/
      test_dir/
        same 4 subfolders
    """
    train_transform, val_transform = get_default_data_transforms(img_size)

    # ImageFolder automatically assigns class indices based on folder names (sorted)
    # Create without transform first, we'll apply transforms in the wrapper
    full_train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=None,  # No transform here, apply in wrapper
    )

    # Save class names (in sorted order)
    class_names = full_train_dataset.classes  # e.g. ['Mild Impairment', 'Moderate Impairment', ...]

    # Split train into train + val
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Wrap subsets with appropriate transforms
    # Train keeps augmentation, val uses deterministic transforms
    train_dataset = TransformDataset(train_subset, transform=train_transform)
    val_dataset = TransformDataset(val_subset, transform=val_transform)

    # Test dataset
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names
