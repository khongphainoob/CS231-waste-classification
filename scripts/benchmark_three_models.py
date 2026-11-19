"""Benchmark the three verified Member 2 checkpoints on one held-out test split."""
import csv
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR if (SCRIPT_DIR / "data").is_dir() else SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.base_predictor import CLASS_NAMES
from src.member2_deeplearning.dataset_aug import EXTENSIONS
from src.member2_deeplearning.predictor_dl import DeepLearningPredictor


DATA = ROOT / "data" / "standard_split" / "test"
OUT = ROOT / "results" / "three_model_comparison"
CHECKPOINTS = {
    "light_cnn": ROOT / "weights" / "light_cnn_best.pth",
    "resnet18": ROOT / "weights" / "resnet18_best.pth",
    "mobilenetv4_conv_small": ROOT / "weights" / "mobilenetv4_best.pth",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples = [
        (p, label)
        for label, cls in enumerate(CLASS_NAMES)
        for p in sorted((DATA / cls).iterdir())
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]
    if len(samples) != 716:
        raise ValueError(f"Expected 716 test images, found {len(samples)}")
    rows = []
    device = "cpu"
    for name, checkpoint_path in CHECKPOINTS.items():
        model = DeepLearningPredictor(str(checkpoint_path), device=device)
        try:
            for _ in range(3):
                model.predict(np.zeros((224, 224, 3), dtype=np.uint8), with_explanation=False)
            truth, predictions, latencies = [], [], []
            for path, label in samples:
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"Cannot decode {path}")
                rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                start = time.perf_counter()
                result = model.predict(rgb, with_explanation=False)
                latencies.append((time.perf_counter() - start) * 1000.0)
                truth.append(label)
                predictions.append(result["class_id"])
            rows.append({
                "model": name,
                "n_test": len(truth),
                "accuracy_pct": round(100 * accuracy_score(truth, predictions), 3),
                "macro_f1_pct": round(100 * f1_score(truth, predictions, labels=list(range(6)), average="macro", zero_division=0), 3),
                "mean_latency_ms": round(float(np.mean(latencies)), 3),
                "median_latency_ms": round(float(np.median(latencies)), 3),
                "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
                "fps": round(1000 / float(np.mean(latencies)), 3),
                "checkpoint_mib": round(checkpoint_path.stat().st_size / 2**20, 3),
                "device": device,
                "epochs": int(torch.load(checkpoint_path, map_location="cpu", weights_only=True)["epoch"]),
            })
        finally:
            model.close()
    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "metadata.json").write_text(json.dumps({
        "dataset": "data/standard_split/test",
        "n_test": len(samples),
        "device": device,
        "warmup_calls": 3,
        "timing": "preprocessing + forward; excludes disk I/O, resize and Grad-CAM",
        "note": "Smoke comparison: LightWasteCNN and MobileNetV4 trained for one epoch locally; ResNet18 uses the existing epoch-13 checkpoint.",
    }, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
