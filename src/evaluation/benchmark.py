"""Fair Benchmark Comparison between Classical CV (TV1) and Deep Learning (TV2).

Evaluates both models on identical Test sets:
  - Standard Test Set (data/standard_split/test)
  - Vietnam Real-World Test Set (data/vn_real_test)

Outputs saved to results/:
  - benchmark_comparison.csv (Accuracy, Macro F1, Latency, FPS, Size MB)
  - benchmark_per_class.csv  (Per-class Precision, Recall, F1 comparison)
  - cm_classical_*.png & cm_dl_*.png
  - case_tv1_wrong_tv2_right.png (Case study visual proof for slides/report)
  - case_tv1_right_tv2_wrong.png
"""

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.base_predictor import CLASS_NAMES
from src.member1_classical.predictor_ml import ClassicalPredictor
from src.member2_deeplearning.predictor_dl import DeepLearningPredictor
from src.utils import list_class_images, read_image_rgb, resolve_path


def compute_metrics(y_true: List[int], y_pred: List[int], num_classes: int = 6):
    """Computes accuracy, macro F1, per-class metrics, and confusion matrix."""
    if not y_true:
        return 0.0, 0.0, [], np.zeros((num_classes, num_classes), dtype=int)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        if 0 <= yt < num_classes and 0 <= yp < num_classes:
            cm[yt, yp] += 1

    total = np.sum(cm)
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0

    per_class = []
    f1_list = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(np.sum(cm[:, c]) - tp)
        fn = int(np.sum(cm[c, :]) - tp)
        support = tp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / support if support > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        per_class.append({
            "class_name": CLASS_NAMES[c],
            "precision": round(prec * 100, 1),
            "recall": round(rec * 100, 1),
            "f1": round(f1 * 100, 1),
            "support": support,
        })

    return accuracy, float(np.mean(f1_list)), per_class, cm


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str) -> None:
    """Saves confusion matrix heatmap."""
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.colorbar()
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=45)
    plt.yticks(ticks, CLASS_NAMES)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Saved confusion matrix: {save_path}")


def evaluate_model(
    predictor, dataset: List[Tuple[str, int]], model_label: str
) -> Dict[str, Any]:
    """Runs inference and computes metrics on a list of (image_path, label)."""
    y_true: List[int] = []
    y_pred: List[int] = []
    confs: List[float] = []
    latencies: List[float] = []
    valid_paths: List[str] = []

    print(f"\n[*] Evaluating '{model_label}' on {len(dataset)} samples...")
    for img_path, gt in dataset:
        rgb = read_image_rgb(img_path)
        if rgb is None:
            continue
        rgb_224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        result = predictor.predict(rgb_224)

        valid_paths.append(img_path)
        y_true.append(gt)
        y_pred.append(result["class_id"])
        confs.append(result["confidence"])
        latencies.append(result["latency_ms"])

    acc, macro_f1, per_class, cm = compute_metrics(y_true, y_pred, len(CLASS_NAMES))
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    fps = (1000.0 / avg_latency) if avg_latency > 0 else 0.0
    print(
        f"    - Accuracy: {acc * 100:.1f}% | Macro F1: {macro_f1 * 100:.1f}% | "
        f"Latency: {avg_latency:.1f} ms ({fps:.1f} FPS)"
    )

    return {
        "model": model_label,
        "accuracy": round(acc * 100, 1),
        "macro_f1": round(macro_f1 * 100, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "fps": round(fps, 1),
        "per_class": per_class,
        "cm": cm,
        "paths": valid_paths,
        "y_true": y_true,
        "y_pred": y_pred,
        "confs": confs,
    }


def save_disagreement_cases(
    p_ml: ClassicalPredictor,
    p_dl: DeepLearningPredictor,
    res_ml: Dict[str, Any],
    res_dl: Dict[str, Any],
    save_path: str,
    title: str,
    pick_ml_wrong_dl_right: bool = True,
    max_cases: int = 4,
) -> None:
    """Renders [Original | TV1 HOG | TV2 Grad-CAM] grid for disagreement cases."""
    indices = []
    for i, (yt, p1, p2) in enumerate(zip(res_ml["y_true"], res_ml["y_pred"], res_dl["y_pred"])):
        if pick_ml_wrong_dl_right and (p1 != yt and p2 == yt):
            indices.append(i)
        elif not pick_ml_wrong_dl_right and (p1 == yt and p2 != yt):
            indices.append(i)

    if not indices:
        return

    rng = np.random.RandomState(42)
    picked = rng.choice(indices, size=min(max_cases, len(indices)), replace=False)

    fig, axes = plt.subplots(len(picked), 3, figsize=(11, 3.3 * len(picked)))
    if len(picked) == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle(f"{title} (Tổng: {len(indices)} ca)", fontsize=13, fontweight="bold")

    for row_idx, idx in enumerate(picked):
        rgb = read_image_rgb(res_ml["paths"][idx])
        rgb_224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        out1 = p_ml.predict(rgb_224)
        out2 = p_dl.predict(rgb_224)
        gt_name = CLASS_NAMES[res_ml["y_true"][idx]]

        axes[row_idx, 0].imshow(rgb_224)
        axes[row_idx, 0].set_title(f"Ảnh gốc | Nhãn thật: {gt_name}", fontsize=10)
        axes[row_idx, 1].imshow(out1["visual_explanation"])
        axes[row_idx, 1].set_title(
            f"TV1 (HOG+SVM): {out1['class_name']} ({out1['confidence']*100:.1f}%)",
            fontsize=10,
            color="red" if out1["class_name"] != gt_name else "green",
        )
        axes[row_idx, 2].imshow(out2["visual_explanation"])
        axes[row_idx, 2].set_title(
            f"TV2 (CNN+GradCAM): {out2['class_name']} ({out2['confidence']*100:.1f}%)",
            fontsize=10,
            color="red" if out2["class_name"] != gt_name else "green",
        )
        for ax in axes[row_idx]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[+] Saved disagreement case grid: {save_path} ({len(indices)} cases found)")


def _file_size_mb(path: str) -> float:
    return round(os.path.getsize(path) / (1024 * 1024), 1) if os.path.exists(path) else 0.0


def run_full_benchmark(
    test_dir: str = "data/standard_split/test",
    vn_test_dir: str = "data/vn_real_test",
    output_dir: str = "results",
    svm_weights: str = "weights/classical_svm_pipeline.pkl",
    dl_weights: str = "weights/mobilenetv4_best.pth",
):
    """Executes full comparative benchmark between Member 1 and Member 2."""
    out_abs = resolve_path(output_dir)
    os.makedirs(out_abs, exist_ok=True)

    svm_abs = resolve_path(svm_weights)
    dl_abs = resolve_path(dl_weights)

    print("\n" + "=" * 70)
    print("CS231 BENCHMARK: CLASSICAL ML vs DEEP LEARNING COMPARISON")
    print("=" * 70)

    p_ml: Optional[ClassicalPredictor] = ClassicalPredictor(svm_abs) if os.path.exists(svm_abs) else None
    if p_ml is None:
        print(f"[!] Classical weights not found at {svm_abs}. Skipping TV1.")
    p_dl = DeepLearningPredictor(weight_path=dl_abs, fallback_mode=True)

    ml_size_mb = _file_size_mb(svm_abs)
    dl_size_mb = _file_size_mb(dl_abs)

    datasets = [
        ("Standard Test Set", test_dir),
        ("Vietnam Real-World Test Set", vn_test_dir),
    ]

    summary_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []

    for set_name, set_path in datasets:
        data = list_class_images(set_path, CLASS_NAMES)
        if not data:
            print(f"[!] Warning: No images found in '{set_path}' ({set_name}). Skipping.")
            continue

        slug = set_name.lower().replace(" ", "_")
        print(f"\n>>> BENCHMARKING ON: {set_name.upper()} ({len(data)} items)")

        res_ml = None
        if p_ml is not None:
            res_ml = evaluate_model(p_ml, data, "Member 1 (HSV+LBP+HOG+SVM)")
            plot_confusion_matrix(
                res_ml["cm"],
                f"Confusion Matrix: Classical ML ({set_name})",
                os.path.join(out_abs, f"cm_classical_{slug}.png"),
            )
            summary_rows.append({
                "Evaluation Dataset": set_name,
                "Samples": len(res_ml["y_true"]),
                "Model": "Member 1 (HSV+LBP+HOG+SVM)",
                "Accuracy (%)": res_ml["accuracy"],
                "Macro F1 (%)": res_ml["macro_f1"],
                "Latency (ms)": res_ml["avg_latency_ms"],
                "Throughput (FPS)": res_ml["fps"],
                "Size (MB)": ml_size_mb,
            })

        res_dl = evaluate_model(p_dl, data, f"Member 2 ({p_dl.model_name} + Grad-CAM)")
        plot_confusion_matrix(
            res_dl["cm"],
            f"Confusion Matrix: Deep Learning ({set_name})",
            os.path.join(out_abs, f"cm_dl_{slug}.png"),
        )
        summary_rows.append({
            "Evaluation Dataset": set_name,
            "Samples": len(res_dl["y_true"]),
            "Model": f"Member 2 ({p_dl.model_name} + Grad-CAM)",
            "Accuracy (%)": res_dl["accuracy"],
            "Macro F1 (%)": res_dl["macro_f1"],
            "Latency (ms)": res_dl["avg_latency_ms"],
            "Throughput (FPS)": res_dl["fps"],
            "Size (MB)": dl_size_mb,
        })

        if res_ml is not None:
            for c_ml, c_dl in zip(res_ml["per_class"], res_dl["per_class"]):
                per_class_rows.append({
                    "Dataset": set_name,
                    "Class": c_ml["class_name"],
                    "Support": c_ml["support"],
                    "TV1 Precision (%)": c_ml["precision"],
                    "TV1 Recall (%)": c_ml["recall"],
                    "TV1 F1 (%)": c_ml["f1"],
                    "TV2 Precision (%)": c_dl["precision"],
                    "TV2 Recall (%)": c_dl["recall"],
                    "TV2 F1 (%)": c_dl["f1"],
                })

            save_disagreement_cases(
                p_ml, p_dl, res_ml, res_dl,
                os.path.join(out_abs, f"case_tv1_wrong_tv2_right_{slug}.png"),
                f"{set_name}: TV1 sai nhưng TV2 đúng",
                pick_ml_wrong_dl_right=True,
            )
            save_disagreement_cases(
                p_ml, p_dl, res_ml, res_dl,
                os.path.join(out_abs, f"case_tv1_right_tv2_wrong_{slug}.png"),
                f"{set_name}: TV1 đúng nhưng TV2 sai",
                pick_ml_wrong_dl_right=False,
            )

    if summary_rows:
        csv_path = os.path.join(out_abs, "benchmark_comparison.csv")
        fieldnames = [
            "Evaluation Dataset", "Samples", "Model",
            "Accuracy (%)", "Macro F1 (%)", "Latency (ms)", "Throughput (FPS)", "Size (MB)",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        if per_class_rows:
            pc_path = os.path.join(out_abs, "benchmark_per_class.csv")
            with open(pc_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()))
                writer.writeheader()
                writer.writerows(per_class_rows)
            print(f"[+] Saved per-class report: {pc_path}")

        print("\n" + "=" * 86)
        print("SUMMARY BENCHMARK REPORT:")
        print("=" * 86)
        header = f"{'Dataset':<26} | {'Model':<28} | {'Acc(%)':<6} | {'F1(%)':<6} | {'Lat(ms)':<7} | {'FPS':<5} | {'MB':<5}"
        print(header)
        print("-" * len(header))
        for r in summary_rows:
            print(
                f"{r['Evaluation Dataset']:<26} | {r['Model']:<28} | "
                f"{r['Accuracy (%)']:<6.1f} | {r['Macro F1 (%)']:<6.1f} | "
                f"{r['Latency (ms)']:<7.1f} | {r['Throughput (FPS)']:<5.1f} | {r['Size (MB)']:<5.1f}"
            )
        print(f"\n[+] Saved summary CSV to: {csv_path}")
    else:
        print("[!] No evaluation images found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="data/standard_split/test")
    parser.add_argument("--vn_dir", default="data/vn_real_test")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    run_full_benchmark(test_dir=args.test_dir, vn_test_dir=args.vn_dir, output_dir=args.out)
