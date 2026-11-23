"""Evaluate horizontal-flip TTA and a ResNet50+MobileNetV4 probability ensemble."""
import csv
import time
from pathlib import Path
import sys

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR if (SCRIPT_DIR / "data").is_dir() else SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

from src.base_predictor import CLASS_NAMES
from src.member2_deeplearning.dataset_aug import EXTENSIONS
from src.member2_deeplearning.predictor_dl import DeepLearningPredictor


def main() -> None:
    data = ROOT / "data" / "standard_split" / "test"
    output = ROOT / "results" / "tta_ensemble"
    output.mkdir(parents=True, exist_ok=True)
    samples = [(p, label) for label, name in enumerate(CLASS_NAMES)
               for p in sorted((data / name).iterdir())
               if p.is_file() and p.suffix.lower() in EXTENSIONS]
    if len(samples) != 716:
        raise ValueError(f"Expected 716 test images, found {len(samples)}")

    models = {
        "resnet50": DeepLearningPredictor(str(ROOT / "weights" / "resnet50_best.pth"), device="cpu"),
        "mobilenetv4_conv_small": DeepLearningPredictor(str(ROOT / "weights" / "mobilenetv4_best.pth"), device="cpu"),
    }
    all_probs = {name: [] for name in models}
    labels = []
    tta_ms = {name: [] for name in models}
    try:
        for path, label in samples:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot decode {path}")
            rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            flipped = np.ascontiguousarray(rgb[:, ::-1])
            labels.append(label)
            for name, model in models.items():
                start = time.perf_counter()
                p1 = np.asarray(model.predict(rgb, with_explanation=False)["probabilities"], dtype=np.float64)
                p2 = np.asarray(model.predict(flipped, with_explanation=False)["probabilities"], dtype=np.float64)
                tta_ms[name].append((time.perf_counter() - start) * 1000.0)
                all_probs[name].append((p1 + p2) / 2.0)
    finally:
        for model in models.values():
            model.close()

    probs = {name: np.vstack(values) for name, values in all_probs.items()}
    rows = []
    for name, values in probs.items():
        pred = values.argmax(axis=1)
        rows.append({"model": f"{name}_hflip_tta", "n_test": len(labels),
                     "accuracy_pct": round(100 * accuracy_score(labels, pred), 3),
                     "macro_f1_pct": round(100 * f1_score(labels, pred, labels=list(range(6)), average="macro", zero_division=0), 3),
                     "mean_latency_ms": round(float(np.mean(tta_ms[name])), 3),
                     "fps": round(1000 / float(np.mean(tta_ms[name])), 3)})
    ensemble = ((probs["resnet50"] + probs["mobilenetv4_conv_small"]) / 2.0).argmax(axis=1)
    combined_latency = np.asarray(tta_ms["resnet50"]) + np.asarray(tta_ms["mobilenetv4_conv_small"])
    rows.append({"model": "resnet50_mobilenetv4_hflip_ensemble", "n_test": len(labels),
                 "accuracy_pct": round(100 * accuracy_score(labels, ensemble), 3),
                 "macro_f1_pct": round(100 * f1_score(labels, ensemble, labels=list(range(6)), average="macro", zero_division=0), 3),
                 "mean_latency_ms": round(float(np.mean(combined_latency)), 3),
                 "fps": round(1000 / float(np.mean(combined_latency)), 3)})
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(rows)


if __name__ == "__main__":
    main()
