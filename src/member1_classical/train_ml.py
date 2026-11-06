"""Training script for Member 1 (Classical CV + ML Pipeline).

Pipeline components:
  1. Hand-crafted Feature Extraction (HSV + LBP + HOG)
  2. StandardScaler normalization
  3. PCA dimensionality reduction (keeps >= 95% variance)
  4. Classifier: Support Vector Classifier (RBF Kernel) or XGBoost
"""

import os
import sys
import glob
import time
import yaml
import argparse
from typing import List, Tuple
import numpy as np
import cv2
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.member1_classical.feature_extractor import WasteFeatureExtractor
from src.base_predictor import CLASS_NAMES


def load_dataset_features(
    data_dir: str,
    extractor: WasteFeatureExtractor,
    max_per_class: int = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads images from subfolders corresponding to class names and extracts features."""
    features_list = []
    labels_list = []
    filepaths_list = []

    print(f"[*] Scanning data directory: {data_dir}")
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.exists(class_folder):
            print(f"[-] Warning: Folder not found: {class_folder}")
            continue

        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
        image_files = []
        for ext in valid_exts:
            image_files.extend(glob.glob(os.path.join(class_folder, ext)))

        if max_per_class and len(image_files) > max_per_class:
            image_files = image_files[:max_per_class]

        print(f"    - Class [{class_id}] '{class_name}': {len(image_files)} images found")

        for img_path in image_files:
            bgr = cv2.imread(img_path)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            feat, _ = extractor.extract_features(rgb, return_viz=False)
            features_list.append(feat)
            labels_list.append(class_id)
            filepaths_list.append(img_path)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    print(f"[+] Loaded total {len(X)} samples. Feature vector dimension: {X.shape[1] if len(X) > 0 else 0}")
    return X, y, filepaths_list


def train_classical_pipeline(
    config_path: str = "configs/config.yaml",
    model_choice: str = "svm",
    max_train_samples: int = None
):
    """Executes the training and evaluation workflow."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    train_dir = cfg.get("dataset", {}).get("paths", {}).get("train_dir", "data/standard_split/train")
    val_dir = cfg.get("dataset", {}).get("paths", {}).get("val_dir", "data/standard_split/val")
    weight_path = cfg.get("classical", {}).get("weight_path", "weights/classical_svm_pipeline.pkl")
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    os.makedirs("results", exist_ok=True)

    extractor = WasteFeatureExtractor()

    print("\n" + "=" * 60)
    print("MEMBER 1: TRAINING CLASSICAL ML PIPELINE")
    print("=" * 60)

    # 1. Feature Extraction
    start_t = time.time()
    X_train, y_train, _ = load_dataset_features(train_dir, extractor, max_per_class=max_train_samples)
    if len(X_train) == 0:
        print("[!] Error: No training data found. Please run data/download_and_split.py first.")
        return

    X_val, y_val, _ = load_dataset_features(val_dir, extractor)
    print(f"[+] Feature extraction completed in {time.time() - start_t:.2f} seconds.")

    # 2. Pipeline Definition
    variance_ratio = cfg.get("classical", {}).get("pca", {}).get("variance_ratio", 0.95)
    
    if model_choice == "svm":
        clf = SVC(
            C=cfg.get("classical", {}).get("svm", {}).get("C", 10.0),
            kernel=cfg.get("classical", {}).get("svm", {}).get("kernel", "rbf"),
            probability=True,
            random_state=42
        )
    else:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            random_state=42,
            eval_metric="mlogloss"
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=variance_ratio, random_state=42)),
        ("classifier", clf)
    ])

    # 3. Fit Pipeline
    print(f"\n[*] Training {model_choice.upper()} pipeline with PCA (variance ratio: {variance_ratio})...")
    fit_start = time.time()
    pipeline.fit(X_train, y_train)
    fit_duration = time.time() - fit_start
    print(f"[+] Training completed in {fit_duration:.2f} seconds.")

    pca_step = pipeline.named_steps["pca"]
    print(f"[i] PCA retained {pca_step.n_components_} components (from original {X_train.shape[1]} dimensions).")

    # 4. Evaluation on Validation Set
    if len(X_val) > 0:
        y_pred = pipeline.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print("\n" + "-" * 50)
        print(f"VALIDATION ACCURACY: {acc * 100:.2f}%")
        print("-" * 50)
        report = classification_report(y_val, y_pred, target_names=CLASS_NAMES)
        print(report)
        
        # Save metrics to results
        with open("results/member1_validation_report.txt", "w", encoding="utf-8") as f:
            f.write(f"Validation Accuracy: {acc * 100:.2f}%\n\n")
            f.write(report)

    # 5. Save Model Pipeline
    joblib.dump(pipeline, weight_path)
    file_size_mb = os.path.getsize(weight_path) / (1024 * 1024)
    print(f"\n[+] Successfully saved trained pipeline to: {weight_path} ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", default="svm", choices=["svm", "xgboost"], help="Classifier type")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples per class for quick testing")
    args = parser.parse_args()

    train_classical_pipeline(config_path=args.config, model_choice=args.model, max_train_samples=args.max_samples)

