"""Dự đoán bằng nhánh Thị giác cổ điển (Thành viên 1).

Kế thừa BasePredictor nên dùng chung được với demo webcam, Streamlit và benchmark.
Các bước khi dự đoán 1 ảnh:
    ảnh 224x224 -> trích HSV + LBP + HOG (2302 chiều) -> Scaler -> PCA -> SVM -> xác suất 6 lớp
"""

import os
import sys
import time

import joblib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.base_predictor import BasePredictor
from src.member1_classical.feature_extractor import WasteFeatureExtractor, hog_overlay
from src.utils import resolve_path


class ClassicalPredictor(BasePredictor):
    def __init__(self, weight_path="weights/classical_svm_pipeline.pkl"):
        weight_path = resolve_path(weight_path)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Chưa có model: {weight_path}\n"
                "Hãy train trước: python src/member1_classical/train_ml.py"
            )

        bundle = joblib.load(weight_path)
        super().__init__(class_names=bundle["class_names"])
        self.pipeline = bundle["pipeline"]
        self.model_type = bundle["model_type"]
        # Dùng đúng tham số trích đặc trưng như lúc train
        self.extractor = WasteFeatureExtractor(**bundle["extractor_params"])
        print(f"[ClassicalPredictor] Đã load {self.model_type.upper()} từ {weight_path} "
              f"(test acc lúc train: {bundle['test_acc'] * 100:.1f}%)")

    def predict(self, image_rgb_224):
        img = self.validate_input(image_rgb_224)
        t0 = time.perf_counter()

        features, steps = self.extractor.extract_with_steps(img)
        probs = self.pipeline.predict_proba(features.reshape(1, -1))[0]

        latency_ms = (time.perf_counter() - t0) * 1000
        class_id = int(np.argmax(probs))
        class_name = self.class_names[class_id]

        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(float(probs[class_id]), 4),
            "probabilities": [round(float(p), 4) for p in probs],
            "latency_ms": round(latency_ms, 2),
            "visual_explanation": hog_overlay(steps["rgb"], steps["hog_image"]),
            "action_bin": self.get_action_bin(class_name),
        }
