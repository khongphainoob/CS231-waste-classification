"""Deep Learning Predictor (Member 2).

Inherits from BasePredictor. Integrates CNN inference + Grad-CAM visual explanation.
"""

import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.base_predictor import BasePredictor, CLASS_NAMES
from src.member2_deeplearning.dataset_aug import IMAGENET_MEAN, IMAGENET_STD
from src.member2_deeplearning.gradcam_viz import GradCAMVisualizer
from src.member2_deeplearning.train_cnn import build_model
from src.utils import resolve_path


class DeepLearningPredictor(BasePredictor):
    """Deep Learning Predictor with Grad-CAM visual explanations."""

    def __init__(
        self,
        weight_path: str = "weights/mobilenetv4_best.pth",
        model_name: str = "light_cnn",
        device: Optional[str] = None,
        fallback_mode: bool = True,
    ):
        super().__init__(class_names=CLASS_NAMES)
        self.weight_path = resolve_path(weight_path)
        self.model_name = model_name
        self.fallback_mode = fallback_mode
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model: Optional[nn.Module] = None
        self.gradcam: Optional[GradCAMVisualizer] = None
        self._load_model()

    def _load_model(self):
        """Loads trained weights and infers backbone name from checkpoint metadata."""
        try:
            state_dict = None
            resolved_model_name = self.model_name

            if os.path.exists(self.weight_path):
                checkpoint = torch.load(self.weight_path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    if "model_name" in checkpoint:
                        resolved_model_name = checkpoint["model_name"]
                    elif checkpoint.get("model_class") == "LightWasteCNN":
                        resolved_model_name = "light_cnn"
                    if "class_names" in checkpoint:
                        self.class_names = checkpoint["class_names"]
                    state_dict = checkpoint["model_state_dict"]
                elif isinstance(checkpoint, dict):
                    state_dict = checkpoint
                else:
                    self.model = checkpoint
            elif not self.fallback_mode:
                raise FileNotFoundError(f"Weight file not found: {self.weight_path}")
            else:
                print(f"[DeepLearningPredictor] Weight file not found: {self.weight_path}. Running initialized weights.")

            if self.model is None:
                self.model = build_model(
                    model_name=resolved_model_name,
                    num_classes=len(self.class_names),
                    pretrained=False,
                )
                if state_dict is not None:
                    self.model.load_state_dict(state_dict)
                    print(f"[DeepLearningPredictor] Loaded {resolved_model_name} from: {self.weight_path}")

            self.model_name = resolved_model_name
            self.model.to(self.device)
            self.model.eval()
            self.gradcam = GradCAMVisualizer(self.model)
            # Warm up PyTorch kernels so the first timed prediction excludes initialization overhead
            with torch.no_grad():
                _ = self.model(torch.zeros((1, 3, 224, 224), device=self.device))

        except Exception as e:
            print(f"[DeepLearningPredictor] Failed to load DL model: {e}")
            if not self.fallback_mode:
                raise
            self.model = None

    def preprocess(self, image_rgb_224: np.ndarray) -> torch.Tensor:
        """Standardizes input RGB image to ImageNet tensor (1, 3, 224, 224)."""
        img_norm = (image_rgb_224.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image_rgb_224: np.ndarray) -> Dict[str, Any]:
        """Runs single-pass PyTorch CNN inference + Grad-CAM heatmap."""
        img = self.validate_input(image_rgb_224)
        t0 = time.perf_counter()

        if self.model is not None and self.gradcam is not None:
            input_tensor = self.preprocess(img)
            heatmap, logits = self.gradcam.generate_heatmap_and_logits(input_tensor)
            viz_blended = self.gradcam.create_overlay(img, heatmap, alpha=0.55)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class_id = int(np.argmax(probs))
            confidence = float(probs[pred_class_id])
            probabilities = [float(p) for p in probs]
        else:
            np.random.seed(int(img.mean() * 1000) % 10000)
            mock_logits = np.random.uniform(0.1, 1.0, size=len(self.class_names))
            probs = mock_logits / mock_logits.sum()
            pred_class_id = int(np.argmax(probs))
            confidence = float(probs[pred_class_id])
            probabilities = [float(p) for p in probs]
            viz_blended = img.copy()

        latency_ms = (time.perf_counter() - t0) * 1000.0
        class_name = self.class_names[pred_class_id]

        return {
            "class_id": pred_class_id,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "probabilities": [round(p, 4) for p in probabilities],
            "latency_ms": round(latency_ms, 2),
            "visual_explanation": viz_blended,
            "action_bin": self.get_action_bin(class_name),
        }
