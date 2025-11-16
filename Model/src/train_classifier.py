# src/train_classifier.py
import os
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data import create_dataloaders
from src.models import AlzheimerResNet, get_device


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    return {"loss": epoch_loss, "acc": acc}


@torch.no_grad()
def evaluate_and_report(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names,
):
    model.eval()
    all_labels = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    print("\nTest Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


def main():
    # === CONFIG ===
    train_dir = "data/raw/train"
    test_dir = "data/raw/test"
    img_size = 128
    batch_size = 32  # you can reduce to 16 if it's slow
    num_epochs = 15  # start with 10–15 for M1, can increase later
    lr = 1e-4
    val_split = 0.15
    num_workers = 4
    checkpoint_path = "models/resnet18_alzheimer.pth"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # === DATA ===
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        img_size=img_size,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
    )
    num_classes = len(class_names)
    print("Classes:", class_names)

    # === MODEL ===
    device = get_device()
    print("Using device:", device)

    model = AlzheimerResNet(num_classes=num_classes, freeze_backbone=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics_val = evaluate(model, val_loader, criterion, device)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {metrics_val['loss']:.4f} | "
            f"Val Acc: {metrics_val['acc']:.4f}"
        )

        # Save best model
        if metrics_val["acc"] > best_val_acc:
            best_val_acc = metrics_val["acc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "img_size": img_size,
                },
                checkpoint_path,
            )
            print(f"✅ Saved new best model with Val Acc {best_val_acc:.4f}")

    # === Final test evaluation with best checkpoint ===
    print("\nLoading best checkpoint for test evaluation...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluate_and_report(model, test_loader, device, class_names)
    print("Done.")


if __name__ == "__main__":
    main()
