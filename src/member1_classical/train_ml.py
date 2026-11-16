"""Huấn luyện nhánh Thị giác cổ điển (Thành viên 1).

Quy trình:
    1. Đọc ảnh train / val / test, trích đặc trưng HSV + LBP + HOG (lưu cache .npz cho lần sau)
    2. Pipeline: StandardScaler -> PCA (giữ 95% phương sai) -> SVM kernel RBF
    3. Chọn siêu tham số (C, gamma) bằng tập VALIDATION
    4. Train lại với bộ tham số tốt nhất, đánh giá 1 lần duy nhất trên tập TEST
    5. Lưu model + báo cáo + ma trận nhầm lẫn + ảnh các ca dự đoán sai

Chạy (từ thư mục gốc đồ án):
    python src/member1_classical/train_ml.py
    python src/member1_classical/train_ml.py --model xgboost
"""

import os
import sys
import json
import time
import argparse

import cv2
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.base_predictor import CLASS_NAMES
from src.member1_classical.feature_extractor import WasteFeatureExtractor
from src.utils import load_config, list_class_images, read_image_rgb, resolve_path

OUT_DIR = os.path.join(ROOT, "results", "member1")
CACHE_DIR = os.path.join(OUT_DIR, "cache")


def make_extractor(cfg):
    """Tạo bộ trích đặc trưng theo các tham số trong configs/config.yaml."""
    c = cfg["classical"]
    return WasteFeatureExtractor(
        h_bins=c["color"]["h_bins"],
        s_bins=c["color"]["s_bins"],
        v_bins=c["color"]["v_bins"],
        lbp_points=c["lbp"]["points"],
        lbp_radius=c["lbp"]["radius"],
        hog_size=c["hog"]["resize"],
        hog_orientations=c["hog"]["orientations"],
        hog_pixels_per_cell=c["hog"]["pixels_per_cell"],
        hog_cells_per_block=c["hog"]["cells_per_block"],
    )


def extract_split(split_name, data_dir, extractor, use_cache=True):
    """Trích đặc trưng cho cả 1 tập ảnh. Kết quả được cache để lần sau chạy nhanh."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"features_{split_name}.npz")
    params_str = json.dumps(extractor.get_params(), sort_keys=True)

    if use_cache and os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=False)
        # Chỉ dùng cache nếu tham số trích đặc trưng không đổi
        if str(data["params"]) == params_str:
            print(f"[{split_name}] Dùng cache: {len(data['y'])} ảnh")
            return data["X"], data["y"], list(data["paths"])

    items = list_class_images(data_dir, CLASS_NAMES)
    X, y, paths = [], [], []
    for path, label in tqdm(items, desc=f"Trích đặc trưng {split_name}"):
        rgb = read_image_rgb(path)
        if rgb is None:
            print("  Bỏ qua ảnh lỗi:", path)
            continue
        X.append(extractor.extract(rgb))
        y.append(label)
        paths.append(path)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    np.savez(cache_file, X=X, y=y, paths=np.array(paths), params=params_str)
    return X, y, paths


def build_pipeline(model_type, cfg, C=10.0, gamma="scale", probability=False):
    variance = cfg["classical"]["pca"]["variance_ratio"]
    if model_type == "svm":
        clf = SVC(kernel="rbf", C=C, gamma=gamma, probability=probability, random_state=42)
    else:
        from xgboost import XGBClassifier
        x = cfg["classical"]["xgboost"]
        clf = XGBClassifier(
            n_estimators=x["n_estimators"], max_depth=x["max_depth"],
            learning_rate=x["learning_rate"], random_state=42,
        )
    return Pipeline([
        ("scaler", StandardScaler()),      # đưa mọi chiều về mean 0, độ lệch chuẩn 1
        ("pca", PCA(n_components=variance, random_state=42)),  # giảm chiều, giữ 95% phương sai
        ("clf", clf),
    ])


def tune_svm(X_train, y_train, X_val, y_val, cfg):
    """Thử từng cặp (C, gamma) -> train trên train, chấm điểm trên val, chọn cặp tốt nhất."""
    grid = cfg["classical"]["svm"]
    results = []
    print("\nChọn siêu tham số SVM trên tập validation:")
    for C in grid["C_list"]:
        for gamma in grid["gamma_list"]:
            pipe = build_pipeline("svm", cfg, C=C, gamma=gamma)
            pipe.fit(X_train, y_train)
            acc = accuracy_score(y_val, pipe.predict(X_val))
            results.append({"C": C, "gamma": gamma, "val_acc": acc})
            print(f"  C={C:<6} gamma={str(gamma):<8} -> val acc = {acc * 100:.2f}%")

    best = max(results, key=lambda r: r["val_acc"])
    print(f"=> Tốt nhất: C={best['C']}, gamma={best['gamma']} (val acc {best['val_acc'] * 100:.2f}%)")

    with open(os.path.join(OUT_DIR, "svm_tuning.csv"), "w", encoding="utf-8") as f:
        f.write("C,gamma,val_acc\n")
        for r in results:
            f.write(f"{r['C']},{r['gamma']},{r['val_acc']:.4f}\n")
    return best


def save_confusion_matrix(cm, title, path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            plt.text(j, i, cm[i, j], ha="center", va="center", color=color)
    plt.xlabel("Dự đoán")
    plt.ylabel("Nhãn thật")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_wrong_examples(paths, y_true, y_pred, path, n=12):
    """Lưu lưới ảnh các ca dự đoán sai để phân tích lỗi khi báo cáo."""
    wrong = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    if not wrong:
        return
    rng = np.random.RandomState(42)
    pick = rng.choice(wrong, size=min(n, len(wrong)), replace=False)

    cols = 4
    rows = int(np.ceil(len(pick) / cols))
    plt.figure(figsize=(cols * 3, rows * 3.6))
    for k, i in enumerate(pick):
        img = read_image_rgb(paths[i])
        plt.subplot(rows, cols, k + 1)
        plt.imshow(cv2.resize(img, (224, 224)))
        plt.title(f"Thật: {CLASS_NAMES[y_true[i]]}\nĐoán: {CLASS_NAMES[y_pred[i]]}", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    parser.add_argument("--model", default=None, choices=["svm", "xgboost"],
                        help="Mặc định lấy theo config (classical.model_type)")
    parser.add_argument("--no_cache", action="store_true", help="Bắt buộc trích lại đặc trưng")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_type = args.model or cfg["classical"]["model_type"]
    paths_cfg = cfg["dataset"]["paths"]
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------------- Bước 1: trích đặc trưng ----------------
    extractor = make_extractor(cfg)
    t0 = time.time()
    X_train, y_train, _ = extract_split("train", os.path.join(ROOT, paths_cfg["train_dir"]), extractor, not args.no_cache)
    X_val, y_val, _ = extract_split("val", os.path.join(ROOT, paths_cfg["val_dir"]), extractor, not args.no_cache)
    X_test, y_test, test_paths = extract_split("test", os.path.join(ROOT, paths_cfg["test_dir"]), extractor, not args.no_cache)
    if len(X_train) == 0:
        print("Không có ảnh train. Hãy chạy data/download_and_split.py trước.")
        return
    print(f"Số ảnh: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"Số chiều vector đặc trưng: {X_train.shape[1]}  (mất {time.time() - t0:.1f}s)")

    # ---------------- Bước 2 + 3: chọn siêu tham số ----------------
    if model_type == "svm":
        best = tune_svm(X_train, y_train, X_val, y_val, cfg)
        best_params = {"C": best["C"], "gamma": best["gamma"]}
        # probability=True để SVM xuất được xác suất (Platt scaling) cho demo
        pipe = build_pipeline("svm", cfg, C=best["C"], gamma=best["gamma"], probability=True)
    else:
        best_params = dict(cfg["classical"]["xgboost"])
        pipe = build_pipeline("xgboost", cfg)

    # ---------------- Bước 4: train model cuối cùng ----------------
    t0 = time.time()
    pipe.fit(X_train, y_train)
    print(f"\nTrain xong sau {time.time() - t0:.1f}s")
    n_pca = pipe.named_steps["pca"].n_components_
    print(f"PCA: {X_train.shape[1]} chiều -> {n_pca} chiều (giữ 95% phương sai)")

    val_acc = accuracy_score(y_val, pipe.predict(X_val))
    y_pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3)

    print(f"\nVal accuracy : {val_acc * 100:.2f}%")
    print(f"Test accuracy: {test_acc * 100:.2f}%   Macro F1: {test_f1 * 100:.2f}%")
    print(report)

    # ---------------- Bước 5: lưu kết quả ----------------
    with open(os.path.join(OUT_DIR, f"report_{model_type}.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: {model_type}  params: {best_params}\n")
        f.write(f"So chieu dac trung: {X_train.shape[1]} -> PCA {n_pca}\n")
        f.write(f"Val accuracy : {val_acc * 100:.2f}%\n")
        f.write(f"Test accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Test macro F1: {test_f1 * 100:.2f}%\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, f"Ma trận nhầm lẫn - {model_type.upper()} (test)",
                          os.path.join(OUT_DIR, f"confusion_matrix_{model_type}.png"))
    save_wrong_examples(test_paths, y_test, y_pred, os.path.join(OUT_DIR, f"wrong_examples_{model_type}.png"))

    # Lưu model kèm tham số trích đặc trưng -> lúc predict dùng đúng y hệt lúc train
    bundle = {
        "pipeline": pipe,
        "extractor_params": extractor.get_params(),
        "class_names": CLASS_NAMES,
        "model_type": model_type,
        "best_params": best_params,
        "val_acc": val_acc,
        "test_acc": test_acc,
    }
    weight_path = os.path.join(ROOT, cfg["classical"]["weight_path"])
    if model_type != "svm":
        weight_path = weight_path.replace("svm", model_type)
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    joblib.dump(bundle, weight_path)
    print(f"Đã lưu model: {weight_path} ({os.path.getsize(weight_path) / 1e6:.1f} MB)")
    print(f"Báo cáo và hình ảnh nằm trong: {OUT_DIR}")


if __name__ == "__main__":
    main()
