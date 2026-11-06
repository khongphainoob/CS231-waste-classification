"""Classical Machine Learning Predictor.

Inherits from BasePredictor. Integrates FeatureExtractor + Trained Scikit-Learn Pipeline.
"""

import os
import sys
import time
from typing import Dict, Any
import numpy as np
import joblib

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.base_predictor import BasePredictor, CLASS_NAMES
from src.member1_classical.feature_extractor import WasteFeatureExtractor


class MockClassicalPipeline:
    """Lightweight dummy pipeline matching scikit-learn interface for offline testing."""
    def __init__(self, n_classes=6):
        self.n_classes = n_classes

    def predict_proba(self, X):
        batch_size = X.shape[0]
        probs = np.zeros((batch_size, self.n_classes), dtype=np.float32)
        for i in range(batch_size):
            h = int(np.abs(np.sum(X[i, :10])) * 1000) % self.n_classes
            p = np.full(self.n_classes, 0.05, dtype=np.float32)
            p[h] = 0.75
            probs[i] = p / p.sum()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class ClassicalPredictor(BasePredictor):
    """Predictor using Hand-crafted Features + SVM/XGBoost Pipeline."""

    def __init__(
        self,
        weight_path: str = "weights/classical_svm_pipeline.pkl",
        fallback_mode: bool = True
    ):
        super().__init__(class_names=CLASS_NAMES)
        self.weight_path = weight_path
        self.extractor = WasteFeatureExtractor()
        self.pipeline = None
        self.fallback_mode = fallback_mode

        self._load_pipeline()

    def _load_pipeline(self):
        """Loads trained pipeline or warns user if not found."""
        if os.path.exists(self.weight_path):
            try:
                self.pipeline = joblib.load(self.weight_path)
                print(f"[ClassicalPredictor] Loaded weights from: {self.weight_path}")
            except Exception as e:
                print(f"[ClassicalPredictor] Failed to load {self.weight_path}: {e}")
                self.pipeline = None
        else:
            if not self.fallback_mode:
                raise FileNotFoundError(f"Weight file not found: {self.weight_path}")
            print(f"[ClassicalPredictor] Weight file not found: {self.weight_path}. Running in FALLBACK mode.")

    def predict(self, image_rgb_224: np.ndarray) -> Dict[str, Any]:
        """Runs feature extraction, PCA, and classification on input image.

        Args:
            image_rgb_224: np.ndarray (224, 224, 3)

        Returns:
            Standard dictionary output contract.
        """
        img = self.validate_input(image_rgb_224)
        t0 = time.perf_counter()

        # 1. Extract hand-crafted features + HOG visual explanation
        features, viz_blended = self.extractor.extract_features(img, return_viz=True)
        if viz_blended is None:
            viz_blended = img.copy()

        # 2. Pipeline inference
        if self.pipeline is not None:
            feat_input = features.reshape(1, -1)
            # Predict probabilities
            if hasattr(self.pipeline, "predict_proba"):
                probs = self.pipeline.predict_proba(feat_input)[0]
            else:
                decision = self.pipeline.decision_function(feat_input)[0]
                exp_d = np.exp(decision - np.max(decision))
                probs = exp_d / exp_d.sum()

            pred_class_id = int(np.argmax(probs))
            confidence = float(probs[pred_class_id])
            probabilities = [float(p) for p in probs]
        else:
            # Fallback mock prediction when weights are not yet generated
            np.random.seed(int(features[:5].sum() * 1000) % 10000)
            mock_logits = np.random.uniform(0.1, 1.0, size=len(self.class_names))
            probs = mock_logits / mock_logits.sum()
            pred_class_id = int(np.argmax(probs))
            confidence = float(probs[pred_class_id])
            probabilities = [float(p) for p in probs]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        class_name = self.class_names[pred_class_id]

        return {
            "class_id": pred_class_id,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "probabilities": [round(p, 4) for p in probabilities],
            "latency_ms": round(latency_ms, 2),
            "visual_explanation": viz_blended,
            "action_bin": self.get_action_bin(class_name)
        }
