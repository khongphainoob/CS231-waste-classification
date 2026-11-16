"""Thí nghiệm Ablation: mỗi nhóm đặc trưng đóng góp bao nhiêu?

Train cùng 1 pipeline (Scaler -> PCA 95% -> SVM, cùng C và gamma đã chọn) nhưng chỉ dùng
1 phần của vector đặc trưng: chỉ HSV, chỉ LBP, chỉ HOG, từng cặp, và cả 3.
Ngoài accuracy chung còn in recall từng lớp để thấy đặc trưng nào giúp lớp nào.

Cần chạy train_ml.py trước (để có cache đặc trưng và bộ C, gamma tốt nhất).
    python src/member1_classical/ablation.py
"""

import os
import sys
import json

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
from src.base_predictor import CLASS_NAMES
from src.member1_classical.feature_extractor import WasteFeatureExtractor
from src.member1_classical.train_ml import load_config, build_pipeline, CACHE_DIR, OUT_DIR

COMBOS = [
    ["HSV"], ["LBP"], ["HOG"],
    ["HSV", "LBP"], ["HSV", "HOG"], ["LBP", "HOG"],
    ["HSV", "LBP", "HOG"],
]


def load_cache(split):
    data = np.load(os.path.join(CACHE_DIR, f"features_{split}.npz"), allow_pickle=False)
    return data["X"], data["y"], json.loads(str(data["params"]))


def main():
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    X_train, y_train, params = load_cache("train")
    X_val, y_val, _ = load_cache("val")
    X_test, y_test, _ = load_cache("test")
    groups = WasteFeatureExtractor(**params).feature_groups()

    # Dùng lại C, gamma đã chọn khi train model chính
    bundle = joblib.load(os.path.join(ROOT, cfg["classical"]["weight_path"]))
    C, gamma = bundle["best_params"]["C"], bundle["best_params"]["gamma"]
    print(f"SVM dùng C={C}, gamma={gamma}\n")

    rows = []
    header = f"{'Đặc trưng':<14}{'Số chiều':>9}{'PCA':>6}{'Val acc':>9}{'Test acc':>10}{'Test F1':>9}  " + \
             "  ".join(f"{c[:5]:>5}" for c in CLASS_NAMES)
    print(header)
    print("-" * len(header))

    for combo in COMBOS:
        idx = np.concatenate([np.arange(X_train.shape[1])[groups[g]] for g in combo])
        pipe = build_pipeline("svm", cfg, C=C, gamma=gamma)
        pipe.fit(X_train[:, idx], y_train)

        val_acc = accuracy_score(y_val, pipe.predict(X_val[:, idx]))
        y_pred = pipe.predict(X_test[:, idx])
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="macro")
        recalls = recall_score(y_test, y_pred, average=None, labels=range(len(CLASS_NAMES)))

        name = " + ".join(combo)
        n_pca = pipe.named_steps["pca"].n_components_
        rows.append([name, len(idx), n_pca, val_acc, test_acc, test_f1] + list(recalls))
        print(f"{name:<14}{len(idx):>9}{n_pca:>6}{val_acc * 100:>8.1f}%{test_acc * 100:>9.1f}%{test_f1 * 100:>8.1f}%  " +
              "  ".join(f"{r * 100:>5.0f}" for r in recalls))

    # Lưu CSV
    csv_path = os.path.join(OUT_DIR, "ablation.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("features,dims,pca_dims,val_acc,test_acc,test_macro_f1," + ",".join(f"recall_{c}" for c in CLASS_NAMES) + "\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]}," + ",".join(f"{v:.4f}" for v in r[3:]) + "\n")

    # Vẽ biểu đồ cột accuracy
    names = [r[0] for r in rows]
    accs = [r[4] * 100 for r in rows]
    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(names, accs, color=["#9bb7d4"] * 6 + ["#1f4e79"])
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, a + 0.8, f"{a:.1f}", ha="center")
    plt.ylabel("Test accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Ablation: đóng góp của từng nhóm đặc trưng (SVM RBF)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "ablation.png")
    plt.savefig(png_path, dpi=150)
    print(f"\nĐã lưu: {csv_path}\n        {png_path}")


if __name__ == "__main__":
    main()
