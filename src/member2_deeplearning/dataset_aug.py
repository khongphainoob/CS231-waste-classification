"""Dataset and Data Augmentation Pipeline for Waste Classification.

Applies robust augmentations to simulate real-world classroom lighting,
orientation changes, and scale differences.
"""

import os
import sys
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.utils import list_class_images, read_image_bgr

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SimpleTransform:
    """OpenCV/NumPy image transformation pipeline."""

    def __init__(self, is_train: bool = True, img_size: int = 224):
        self.is_train = is_train
        self.img_size = img_size

    def __call__(self, img_bgr: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.is_train:
            if np.random.rand() > 0.5:
                img_rgb = np.fliplr(img_rgb)

            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15.0, 15.0)
                h, w = img_rgb.shape[:2]
                mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                img_rgb = cv2.warpAffine(img_rgb, mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)

            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.85, 1.15)
                beta = np.random.uniform(-0.1, 0.1)
                img_rgb = np.clip(alpha * img_rgb + beta, 0.0, 1.0)

        img_norm = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(img_norm.transpose(2, 0, 1)).float()


def get_transforms(is_train: bool = True, img_size: int = 224) -> Callable:
    """Returns transform pipeline."""
    return SimpleTransform(is_train=is_train, img_size=img_size)


class WasteDataset(Dataset):
    """PyTorch Dataset for single-item waste classification."""

    def __init__(
        self,
        data_dir: str,
        class_names: List[str],
        transform: Optional[Callable] = None,
        max_samples_per_class: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.class_names = class_names
        self.transform = transform if transform is not None else get_transforms(is_train=False)
        self.samples: List[Tuple[str, int]] = list_class_images(
            data_dir, class_names, max_per_class=max_samples_per_class
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        bgr = read_image_bgr(img_path)
        if bgr is None:
            bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        return self.transform(bgr), label
