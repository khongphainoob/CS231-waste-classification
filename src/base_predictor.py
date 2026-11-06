"""Base Predictor Interface for Waste Classification.

Both Member 1 (Classical ML) and Member 2 (Deep Learning) must inherit from
BasePredictor to guarantee seamless integration in Demo and Benchmark modules.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


CLASS_NAMES: List[str] = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "biological"
]

BIN_MAPPING: Dict[str, str] = {
    "cardboard": "Thùng Tái Chế (Trắng / Xanh Dương)",
    "glass": "Thùng Tái Chế (Trắng / Xanh Dương)",
    "metal": "Thùng Tái Chế (Trắng / Xanh Dương)",
    "paper": "Thùng Tái Chế (Trắng / Xanh Dương)",
    "plastic": "Thùng Tái Chế (Trắng / Xanh Dương)",
    "biological": "Thùng Hữu Cơ (Xanh Lá)"
}


class BasePredictor(ABC):
    """Abstract Base Class specifying standard prediction contract."""

    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names if class_names is not None else CLASS_NAMES

    @abstractmethod
    def predict(self, image_rgb_224: np.ndarray) -> Dict[str, Any]:
        """Predicts waste category for an input RGB image of size (224, 224, 3).

        Args:
            image_rgb_224: np.ndarray of shape (224, 224, 3), dtype uint8 or float32.

        Returns:
            dict containing:
                - class_id: int in [0, 5]
                - class_name: str
                - confidence: float in [0.0, 1.0]
                - probabilities: list of 6 floats summing to 1.0
                - latency_ms: float (inference latency in milliseconds)
                - visual_explanation: np.ndarray shape (224, 224, 3), uint8 (HOG/LBP or Grad-CAM)
                - action_bin: str (recyclable or organic bin recommendation)
        """
        pass

    def get_action_bin(self, class_name: str) -> str:
        """Helper to get Vietnamese municipal bin guideline for a class name."""
        return BIN_MAPPING.get(class_name, "Không xác định")

    def validate_input(self, image: np.ndarray) -> np.ndarray:
        """Validates and ensures input is (224, 224, 3) uint8."""
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy ndarray.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Input image must have shape (H, W, 3), got {image.shape}")
        
        # Ensure image is resized to 224x224
        if image.shape[0] != 224 or image.shape[1] != 224:
            import cv2
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        return image

