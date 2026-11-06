"""Deep Learning Training Script (Member 2).

Fine-tunes a convolutional neural network (MobileNetV4 / ResNet / Lightweight CNN)
for 6-class waste classification with checkpointing and early stopping.
"""

import os
import sys
import time
import yaml
import argparse
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.base_predictor import CLASS_NAMES
from src.member2_deeplearning.dataset_aug import WasteDataset, get_transforms


class LightWasteCNN(nn.Module):
    """Clean, self-contained lightweight ConvNet for waste classification.
    Runs efficiently on both CPU and GPU without external model weights dependency.
    """
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 5 (Final Conv feature layer for Grad-CAM)
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1), # 7x7
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(384, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_model(model_name: str = "light_cnn", num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """Builds model using timm or fallback LightWasteCNN."""
    try:
        import timm
        print(f"[*] Building model '{model_name}' via timm...")
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        return model
    except Exception as e:
        print(f"[!] timm load failed or unavailable ({e}). Using native LightWasteCNN architecture.")
        return LightWasteCNN(num_classes=num_classes)


def train_cnn_model(
    config_path: str = "configs/config.yaml",
    model_name: str = "light_cnn",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None
):
    """Executes the PyTorch training loop."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    dl_cfg = cfg.get("deep_learning", {})
    epochs = epochs if epochs is not None else dl_cfg.get("epochs", 30)
    batch_size = batch_size if batch_size is not None else dl_cfg.get("batch_size", 32)
    lr = lr if lr is not None else dl_cfg.get("learning_rate", 0.0003)
    weight_path = dl_cfg.get("weight_path", "weights/mobilenetv4_best.pth")

    train_dir = cfg.get("dataset", {}).get("paths", {}).get("train_dir", "data/standard_split/train")
    val_dir = cfg.get("dataset", {}).get("paths", {}).get("val_dir", "data/standard_split/val")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Using computational device: {device}")

    # Datasets and Loaders
    train_dataset = WasteDataset(train_dir, class_names=CLASS_NAMES, transform=get_transforms(is_train=True))
    val_dataset = WasteDataset(val_dir, class_names=CLASS_NAMES, transform=get_transforms(is_train=False))

    if len(train_dataset) == 0:
        print(f"[!] Error: No training data found at {train_dir}. Please run data/download_and_split.py first.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[+] Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # Model, Loss, Optimizer
    model = build_model(model_name=model_name, num_classes=len(CLASS_NAMES), pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("\n" + "=" * 60)
    print(f"MEMBER 2: TRAINING DEEP LEARNING MODEL ({model.__class__.__name__})")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        epoch_start = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        scheduler.step()
        epoch_train_loss = train_loss / train_total if train_total > 0 else 0
        epoch_train_acc = train_correct / train_total if train_total > 0 else 0

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        epoch_val_loss = val_loss / val_total if val_total > 0 else 0
        epoch_val_acc = val_correct / val_total if val_total > 0 else 0
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) - "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%"
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "model_class": model.__class__.__name__,
                    "class_names": CLASS_NAMES,
                },
                weight_path
            )
            print(f"  >>> Checkpoint saved! New Best Val Accuracy: {best_val_acc * 100:.2f}%")

    print(f"\n[+] Training complete! Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"[+] Model weights saved at: {weight_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default="light_cnn", help="Model name: mobilenetv4_conv_small or light_cnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    args = parser.parse_args()

    train_cnn_model(
        config_path=args.config,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

