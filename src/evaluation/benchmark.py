"""Fair Benchmark Comparison between Classical CV and Deep Learning.

Evaluates both models on identical Test sets:
  - Standard Test Set (720 images)
  - Vietnam Real-World Test Set (120 images)
Computes Accuracy, Macro F1, Latency (ms), and saves Confusion Matrices and CSV.
"""

import os
import sys
import glob
import time
import csv
import argparse
from typing import Dict, List, Any
import numpy as np
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.base_predictor import CLASS_NAMES
from src.member1_classical.predictor_ml import ClassicalPredictor
from src.member2_deeplearning.predictor_dl import DeepLearningPredictor


def load_test_images(test_dir: str) -> List[Dict[str, Any]]:
    """Loads all test image paths and their ground truth labels."""
    dataset = []
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")

    for class_id, class_name in enumerate(CLASS_NAMES):
        folder = os.path.join(test_dir, class_name)
        if not os.path.isdir(folder):
            continue
        files = []
        for ext in valid_exts:
            files.extend(glob.glob(os.path.join(folder, ext)))

        for f in files:
            dataset.append({"path": f, "ground_truth": class_id, "class_name": class_name})

    return dataset


def compute_metrics(y_true: List[int], y_pred: List[int], num_classes: int = 6):
    """Pure-NumPy implementation of classification metrics."""
    if len(y_true) == 0:
        return 0.0, 0.0, np.zeros((num_classes, num_classes), dtype=int)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        if 0 <= yt < num_classes and 0 <= yp < num_classes:
            cm[yt, yp] += 1

    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0

    f1_list = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list))
    return float(accuracy), float(macro_f1), cm


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str):
    """Draws and saves a clean confusion matrix using matplotlib or OpenCV fallback."""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(title, fontsize=12, fontweight="bold")
        plt.colorbar()
        tick_marks = np.arange(len(CLASS_NAMES))
        plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
        plt.yticks(tick_marks, CLASS_NAMES)

        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black"
                )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[+] Saved confusion matrix to: {save_path}")
    except Exception as e:
        # OpenCV fallback visualizer
        img_cm = np.full((360, 480, 3), 255, dtype=np.uint8)
        cv2.putText(img_cm, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imwrite(save_path, img_cm)


def evaluate_model(predictor, dataset: List[Dict[str, Any]], model_label: str) -> Dict[str, Any]:
    """Runs inference and computes evaluation metrics for a single model."""
    y_true = []
    y_pred = []
    latencies = []

    print(f"\n[*] Evaluating '{model_label}' on {len(dataset)} samples...")

    for item in dataset:
        bgr = cv2.imread(item["path"])
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_224 = cv2.resize(rgb, (224, 224))

        result = predictor.predict(rgb_224)

        y_true.append(item["ground_truth"])
        y_pred.append(result["class_id"])
        latencies.append(result["latency_ms"])

    acc, macro_f1, cm = compute_metrics(y_true, y_pred, num_classes=len(CLASS_NAMES))
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    fps = (1000.0 / avg_latency) if avg_latency > 0 else 0.0

    print(f"    - Accuracy: {acc * 100:.2f}% | Macro F1: {macro_f1 * 100:.2f}% | Latency: {avg_latency:.2f} ms ({fps:.1f} FPS)")

    return {
        "model": model_label,
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "fps": round(fps, 1),
        "cm": cm,
        "y_true": y_true,
        "y_pred": y_pred
    }


def run_full_benchmark(
    test_dir: str = "data/standard_split/test",
    vn_test_dir: str = "data/vn_real_test",
    output_dir: str = "results"
):
    """Executes full comparative benchmark between Member 1 and Member 2."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("CS231 BENCHMARK: CLASSICAL ML vs DEEP LEARNING COMPARISON")
    print("=" * 70)

    # Instantiate predictors
    p_ml = ClassicalPredictor(weight_path="weights/classical_svm_pipeline.pkl", fallback_mode=True)
    p_dl = DeepLearningPredictor(weight_path="weights/mobilenetv4_best.pth", fallback_mode=True)

    datasets = [
        ("Standard Test Set", test_dir),
        ("Vietnam Real-World Test Set", vn_test_dir)
    ]

    all_rows = []

    for set_name, set_path in datasets:
        data = load_test_images(set_path)
        if len(data) == 0:
            print(f"[!] Warning: No images found in '{set_path}' ({set_name}). Skipping.")
            continue

        print(f"\n>>> BENCHMARKING ON: {set_name.upper()} ({len(data)} items)")
        
        # 1. Classical
        res_ml = evaluate_model(p_ml, data, "Classical (HOG+LBP+SVM)")
        plot_confusion_matrix(
            res_ml["cm"],
            f"Confusion Matrix: Classical ML ({set_name})",
            os.path.join(output_dir, f"cm_classical_{set_name.lower().replace(' ', '_')}.png")
        )

        # 2. Deep Learning
        res_dl = evaluate_model(p_dl, data, "Deep Learning (CNN + Grad-CAM)")
        plot_confusion_matrix(
            res_dl["cm"],
            f"Confusion Matrix: Deep Learning ({set_name})",
            os.path.join(output_dir, f"cm_dl_{set_name.lower().replace(' ', '_')}.png")
        )

        all_rows.append({
            "Evaluation Dataset": set_name,
            "Model": "Member 1 (HOG+LBP+SVM)",
            "Accuracy (%)": res_ml["accuracy"],
            "Macro F1 (%)": res_ml["macro_f1"],
            "Latency (ms)": res_ml["avg_latency_ms"],
            "Throughput (FPS)": res_ml["fps"]
        })
        all_rows.append({
            "Evaluation Dataset": set_name,
            "Model": "Member 2 (CNN + Grad-CAM)",
            "Accuracy (%)": res_dl["accuracy"],
            "Macro F1 (%)": res_dl["macro_f1"],
            "Latency (ms)": res_dl["avg_latency_ms"],
            "Throughput (FPS)": res_dl["fps"]
        })

    if all_rows:
        csv_path = os.path.join(output_dir, "benchmark_comparison.csv")
        fieldnames = ["Evaluation Dataset", "Model", "Accuracy (%)", "Macro F1 (%)", "Latency (ms)", "Throughput (FPS)"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        print("\n" + "=" * 70)
        print("SUMMARY BENCHMARK REPORT:")
        print("=" * 70)
        header = f"{'Evaluation Dataset':<28} | {'Model':<25} | {'Acc (%)':<8} | {'F1 (%)':<8} | {'Latency':<9} | {'FPS':<6}"
        print(header)
        print("-" * len(header))
        for row in all_rows:
            print(f"{row['Evaluation Dataset']:<28} | {row['Model']:<25} | {row['Accuracy (%)']:<8.2f} | {row['Macro F1 (%)']:<8.2f} | {str(row['Latency (ms)']) + 'ms':<9} | {row['Throughput (FPS)']:<6.1f}")
        print(f"\n[+] Saved summary CSV to: {csv_path}")
    else:
        print("[!] No evaluation images found. Place test images in data/ folders to benchmark.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="data/standard_split/test")
    parser.add_argument("--vn_dir", default="data/vn_real_test")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    run_full_benchmark(test_dir=args.test_dir, vn_test_dir=args.vn_dir, output_dir=args.out)
