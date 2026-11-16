"""Deep Learning Training Script (Member 2).

Fine-tunes a CNN (timm MobileNetV4, torchvision MobileNetV3/ResNet18/EfficientNet,
or native LightWasteCNN) for 6-class waste classification with early stopping,
learning curves, and test evaluation.
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.base_predictor import CLASS_NAMES
from src.member2_deeplearning.dataset_aug import WasteDataset, get_transforms
from src.utils import load_config, resolve_path, set_seed

OUT_DIR = os.path.join(ROOT, "results", "member2")


class LightWasteCNN(nn.Module):
    """Self-contained lightweight ConvNet when external backbones are unavailable."""

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(384, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _build_torchvision_model(model_name: str, num_classes: int, pretrained: bool) -> Optional[nn.Module]:
    """Builds supported torchvision backbones with replaced classifier head."""
    try:
        from torchvision import models
    except ImportError:
        return None

    weights_flag = "DEFAULT" if pretrained else None
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights_flag)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights_flag)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "resnet18":
        model = models.resnet18(weights=weights_flag)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights_flag)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    return None


def build_model(model_name: str = "mobilenet_v3_small", num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """Builds CNN backbone via torchvision, timm, or native LightWasteCNN fallback."""
    if model_name == "light_cnn":
        return LightWasteCNN(num_classes=num_classes)

    tv_model = _build_torchvision_model(model_name, num_classes, pretrained)
    if tv_model is not None:
        return tv_model

    try:
        import timm
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception as e:
        print(f"[!] Cannot load '{model_name}' ({e}). Falling back to LightWasteCNN.")
        return LightWasteCNN(num_classes=num_classes)


def evaluate_loader(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, List[int], List[int]]:
    """Evaluates model on a DataLoader and returns (avg_loss, accuracy, y_true, y_pred)."""
    model.eval()
    total_loss, total = 0.0, 0
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            total += targets.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    return avg_loss, float(acc), y_true, y_pred


def save_training_curves(history: List[Dict[str, float]], out_dir: str) -> None:
    """Saves history.csv and Loss/Accuracy curve plot."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [r["train_loss"] for r in history], label="Train Loss")
    ax1.plot(epochs, [r["val_loss"] for r in history], label="Val Loss")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [r["train_acc"] * 100 for r in history], label="Train Acc")
    ax2.plot(epochs, [r["val_acc"] * 100 for r in history], label="Val Acc")
    ax2.set_title("Accuracy (%) per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=150)
    plt.close(fig)


def train_cnn_model(
    config_path: str = "configs/config.yaml",
    model_name: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
):
    """Executes the PyTorch training, early stopping, and test evaluation workflow."""
    cfg = load_config(config_path)
    seed = cfg.get("project", {}).get("random_seed", 42)
    set_seed(seed)

    dl_cfg = cfg.get("deep_learning", {})
    model_name = model_name or dl_cfg.get("model_name", "mobilenet_v3_small")
    epochs = epochs if epochs is not None else dl_cfg.get("epochs", 30)
    batch_size = batch_size if batch_size is not None else dl_cfg.get("batch_size", 32)
    lr = lr if lr is not None else dl_cfg.get("learning_rate", 0.0003)
    weight_decay = dl_cfg.get("weight_decay", 0.0001)
    patience = dl_cfg.get("early_stopping_patience", 7)
    weight_path = resolve_path(dl_cfg.get("weight_path", "weights/mobilenetv4_best.pth"))

    paths_cfg = cfg.get("dataset", {}).get("paths", {})
    train_dir = resolve_path(paths_cfg.get("train_dir", "data/standard_split/train"))
    val_dir = resolve_path(paths_cfg.get("val_dir", "data/standard_split/val"))
    test_dir = resolve_path(paths_cfg.get("test_dir", "data/standard_split/test"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device} | Seed: {seed}")

    train_dataset = WasteDataset(train_dir, CLASS_NAMES, transform=get_transforms(is_train=True))
    val_dataset = WasteDataset(val_dir, CLASS_NAMES, transform=get_transforms(is_train=False))
    test_dataset = WasteDataset(test_dir, CLASS_NAMES, transform=get_transforms(is_train=False))

    if len(train_dataset) == 0:
        print(f"[!] Error: No training data found at {train_dir}.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(model_name=model_name, num_classes=len(CLASS_NAMES), pretrained=True).to(device)
    effective_name = "light_cnn" if isinstance(model, LightWasteCNN) else model_name
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    best_val_acc = 0.0
    no_improve_epochs = 0
    history: List[Dict[str, float]] = []

    print(f"\n=== MEMBER 2: TRAINING {effective_name} ({model.__class__.__name__}) ===")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (torch.argmax(outputs, dim=1) == targets).sum().item()
            train_total += targets.size(0)

        scheduler.step()
        tr_loss = train_loss / train_total if train_total > 0 else 0.0
        tr_acc = train_correct / train_total if train_total > 0 else 0.0
        val_loss, val_acc, _, _ = evaluate_loader(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({time.time() - t0:.1f}s) | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc * 100:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.1f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": effective_name,
                    "model_class": model.__class__.__name__,
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "class_names": CLASS_NAMES,
                },
                weight_path,
            )
            print(f"  >>> Saved best checkpoint (Val Acc: {best_val_acc * 100:.1f}%)")
        else:
            no_improve_epochs += 1
            if patience and no_improve_epochs >= patience:
                print(f"[*] Early stopping triggered after {patience} epochs without improvement.")
                break

    save_training_curves(history, OUT_DIR)

    # Final single-pass evaluation on Test Set using best checkpoint
    if len(test_dataset) > 0 and os.path.exists(weight_path):
        ckpt = torch.load(weight_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        _, test_acc, y_true, y_pred = evaluate_loader(model, test_loader, criterion, device)
        test_f1 = f1_score(y_true, y_pred, average="macro")
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
        print(f"\n[Test] Accuracy: {test_acc * 100:.1f}% | Macro F1: {test_f1 * 100:.1f}%")
        print(report)

        with open(os.path.join(OUT_DIR, "report_dl.txt"), "w", encoding="utf-8") as f:
            f.write(f"Model: {effective_name} ({model.__class__.__name__})\n")
            f.write(f"Val accuracy : {best_val_acc * 100:.2f}%\n")
            f.write(f"Test accuracy: {test_acc * 100:.2f}%\n")
            f.write(f"Test macro F1: {test_f1 * 100:.2f}%\n\n")
            f.write(report)

        cm = confusion_matrix(y_true, y_pred)
        fig = plt.figure(figsize=(7, 6))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"Confusion Matrix - {effective_name} (test)")
        plt.colorbar()
        plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
        plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_dl.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default=None, help="mobilenetv4_conv_small, mobilenet_v3_small, resnet18, light_cnn")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    train_cnn_model(
        config_path=args.config,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
