"""Train 6-class CNN, ResNet18 or MobileNetV4 without synthetic/architecture fallback.

Run from project root:
python -m src.member2_deeplearning.train_cnn --model resnet18 --epochs 20 --freeze_epochs 3
"""
import argparse
import csv
import json
import os
import sys
import random
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.base_predictor import CLASS_NAMES
from src.member2_deeplearning.audit_data import audit_dataset
from src.member2_deeplearning.dataset_aug import WasteDataset, get_transforms
from src.member2_deeplearning.model_factory import (
    LightWasteCNN, SUPPORTED_MODELS, build_model, set_backbone_trainable,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        pass


def worker_seed(worker_id):
    worker_seed_value = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed_value)
    random.seed(worker_seed_value)


def one_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    # During classifier-only warm-up, frozen BatchNorm statistics must not drift.
    if is_training and not any(p.requires_grad for n, p in model.named_parameters()
                               if not ("fc." in n or "classifier." in n or "head." in n)):
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    total_loss, n_total, n_correct = 0., 0, 0
    true_labels, predictions = [], []
    with torch.set_grad_enabled(is_training):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if is_training:
                loss.backward()
                optimizer.step()
            pred = logits.argmax(1)
            bs = targets.shape[0]
            total_loss += float(loss.item()) * bs
            n_total += bs
            n_correct += int((pred == targets).sum().item())
            true_labels.extend(targets.cpu().tolist())
            predictions.extend(pred.detach().cpu().tolist())
    if not n_total:
        raise ValueError("Empty training/validation dataloader")
    return {"loss": total_loss / n_total,
            "accuracy": n_correct / n_total,
            "macro_f1": float(f1_score(true_labels, predictions,
                                       labels=list(range(len(CLASS_NAMES))),
                                       average="macro", zero_division=0))}


def _weight_name(model_name):
    return "mobilenetv4_best.pth" if model_name == "mobilenetv4_conv_small" else f"{model_name}_best.pth"


def train_cnn_model(config_path="configs/config.yaml", model_name=None, epochs=None,
                    batch_size=None, lr=None, freeze_epochs=0, patience=None,
                    workers=None, data_root=None, output_root=".", pretrained=None,
                    allow_synthetic=False, seed=None, label_smoothing=None):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) if Path(config_path).exists() else {}
    cfg = cfg or {}
    dl_cfg = cfg.get("deep_learning", {})
    model_name = model_name or dl_cfg.get("model_name", "resnet18")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Supported: {SUPPORTED_MODELS}")
    pretrained = (model_name != "light_cnn" and dl_cfg.get("pretrained", True)) if pretrained is None else pretrained
    epochs = int(epochs if epochs is not None else dl_cfg.get("epochs", 30))
    batch_size = int(batch_size if batch_size is not None else dl_cfg.get("batch_size", 32))
    lr = float(lr if lr is not None else dl_cfg.get("learning_rate", .0003))
    label_smoothing = float(label_smoothing if label_smoothing is not None else dl_cfg.get("label_smoothing", 0.0))
    patience = int(patience if patience is not None else dl_cfg.get("early_stopping_patience", 7))
    workers = int(workers if workers is not None else dl_cfg.get("num_workers", 0))
    seed = int(seed if seed is not None else cfg.get("project", {}).get("random_seed", 42))
    if epochs < 1 or batch_size < 1 or freeze_epochs < 0 or freeze_epochs >= epochs:
        raise ValueError("Require epochs>=1, batch_size>=1, 0<=freeze_epochs<epochs")
    if model_name == "light_cnn" and (pretrained or freeze_epochs):
        raise ValueError("light_cnn must train from scratch without freeze_epochs")
    root = Path(data_root or "data/standard_split")
    output_root = Path(output_root)
    audit = audit_dataset(root, allow_synthetic=allow_synthetic)
    seed_everything(seed)
    train_ds = WasteDataset(str(root / "train"), CLASS_NAMES, get_transforms(True))
    val_ds = WasteDataset(str(root / "val"), CLASS_NAMES, get_transforms(False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DL] model={model_name} device={device} train={len(train_ds)} val={len(val_ds)} "
          f"seed={seed} pretrained={pretrained}")
    g = torch.Generator().manual_seed(seed)
    common = {"batch_size": batch_size, "num_workers": workers,
              "pin_memory": device.type == "cuda", "worker_init_fn": worker_seed,
              "generator": g}
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    model = build_model(model_name, len(CLASS_NAMES), pretrained=pretrained).to(device)
    if freeze_epochs:
        set_backbone_trainable(model, model_name, False)
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=float(dl_cfg.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    ckpt_path = output_root / "weights" / _weight_name(model_name)
    out_dir = output_root / "results" / "dl" / model_name
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    rows = []
    best_f1, best_loss, best_epoch, stale = -1., float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        if freeze_epochs and epoch == freeze_epochs + 1:
            set_backbone_trainable(model, model_name, True)
            print("[DL] Unfrozen all backbone parameters")
        t0 = time.perf_counter()
        tr = one_epoch(model, train_loader, criterion, device, optimizer)
        va = one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = {"epoch": epoch, "stage": "frozen" if epoch <= freeze_epochs else "finetune",
               "train_loss": tr["loss"], "train_accuracy": tr["accuracy"],
               "train_macro_f1": tr["macro_f1"],
               "val_loss": va["loss"], "val_accuracy": va["accuracy"],
               "val_macro_f1": va["macro_f1"], "learning_rate": optimizer.param_groups[0]["lr"],
               "seconds": round(time.perf_counter() - t0, 2)}
        rows.append(row)
        print(f"Epoch {epoch:02d}/{epochs}: train={tr['accuracy']:.3f} "
              f"val_acc={va['accuracy']:.3f} val_F1={va['macro_f1']:.3f} "
              f"val_loss={va['loss']:.4f} time={row['seconds']}s")
        improved = (va["macro_f1"] > best_f1 + 1e-7 or
                    (abs(va["macro_f1"] - best_f1) <= 1e-7 and va["loss"] < best_loss - 1e-7))
        if improved:
            best_f1, best_loss, best_epoch, stale = va["macro_f1"], va["loss"], epoch, 0
            torch.save({"checkpoint_kind": "trained", "model_name": model_name,
                        "model_state_dict": model.state_dict(), "epoch": epoch,
                        "class_names": list(CLASS_NAMES), "img_size": 224,
                        "val_accuracy": va["accuracy"], "val_macro_f1": va["macro_f1"],
                        "seed": seed, "dataset_manifest_sha256": audit["manifest_sha256"],
                        "pretrained_initialization": bool(pretrained)}, ckpt_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"[DL] Early stopping at epoch {epoch}")
                break
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"model_name": model_name, "checkpoint": str(ckpt_path),
              "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
              "best_val_loss": best_loss, "epochs_completed": len(rows),
              "seed": seed, "pretrained_initialization": bool(pretrained),
              "data_audit": audit, "device": str(device)}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[DL] Checkpoint: {ckpt_path} | best val macro F1 = {best_f1:.4f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--freeze_epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--output_root", default=".")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow_synthetic", action="store_true", help="DEBUG ONLY")
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()
    train_cnn_model(config_path=args.config, model_name=args.model, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr, freeze_epochs=args.freeze_epochs,
                    patience=args.patience, workers=args.workers, data_root=args.data,
                    output_root=args.output_root,
                    pretrained=False if args.no_pretrained else None,
                    allow_synthetic=args.allow_synthetic, seed=args.seed,
                    label_smoothing=args.label_smoothing)
