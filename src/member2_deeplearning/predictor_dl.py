"""Strict real-checkpoint inference compatible with src.base_predictor.BasePredictor.
Timings report preprocessing + model forward, excluding Grad-CAM.
"""
from pathlib import Path
from time import perf_counter
from typing import Optional
import cv2
import numpy as np
import torch
from src.base_predictor import BasePredictor, CLASS_NAMES
from src.member2_deeplearning.dataset_aug import IMAGENET_MEAN, IMAGENET_STD
from src.member2_deeplearning.model_factory import build_model, gradcam_target_layer
from src.member2_deeplearning.gradcam_viz import GradCAMVisualizer
from src.utils import resolve_path


class DeepLearningPredictor(BasePredictor):
    def __init__(self, weight_path="weights/mobilenetv4_best.pth", model_name: Optional[str] = None,
                 device: Optional[str] = None, fallback_mode=False):
        super().__init__(class_names=CLASS_NAMES)
        self.weight_path = resolve_path(str(weight_path))
        if not Path(self.weight_path).is_file():
            raise FileNotFoundError(f"Real trained checkpoint required: {self.weight_path}. "
                                    "Mock predictions are not permitted.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(self.weight_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_kind") != "trained":
            raise ValueError("Unverified/legacy/mock checkpoint. Retrain using the upgraded train_cnn.py")
        if checkpoint.get("class_names") != list(CLASS_NAMES):
            raise ValueError("Checkpoint class order differs from src.base_predictor.CLASS_NAMES")
        ckpt_arch = checkpoint.get("model_name")
        if model_name is not None and model_name != ckpt_arch:
            raise ValueError(f"Requested architecture {model_name} != checkpoint {ckpt_arch}")
        self.model_name = ckpt_arch
        self.img_size = int(checkpoint.get("img_size", 224))
        if self.img_size != 224:
            raise ValueError("Repo BasePredictor interface requires 224x224 input")
        self.model = build_model(self.model_name, len(CLASS_NAMES), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.gradcam = None  # Lazily register hooks only if needed for demo.

    def preprocess(self, image_rgb_224: np.ndarray):
        normalized = (image_rgb_224.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0).to(self.device)

    def predict(self, image_rgb_224: np.ndarray, with_explanation=True):
        rgb = self.validate_input(image_rgb_224)
        t0 = perf_counter()
        x = self.preprocess(rgb)
        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, 1).cpu().numpy()[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (perf_counter() - t0) * 1000
        class_id = int(probs.argmax())
        explanation = rgb.copy()
        explanation_ms = 0.0
        if with_explanation:
            start = perf_counter()
            if self.gradcam is None:
                self.gradcam = GradCAMVisualizer(self.model, gradcam_target_layer(self.model, self.model_name))
            heatmap = self.gradcam.generate_heatmap(x, target_class=class_id)
            explanation = self.gradcam.create_overlay(rgb, heatmap)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            explanation_ms = (perf_counter() - start) * 1000
        cls = self.class_names[class_id]
        return {
            "class_id": class_id,
            "class_name": cls,
            "confidence": float(probs[class_id]),
            "probabilities": probs.astype(float).tolist(),
            "latency_ms": round(latency_ms, 3),
            "explanation_latency_ms": round(explanation_ms, 3),
            "visual_explanation": explanation,
            "action_bin": self.get_action_bin(cls),
        }

    def close(self):
        if self.gradcam is not None:
            self.gradcam.remove_hooks()
            self.gradcam = None
