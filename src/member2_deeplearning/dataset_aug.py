"""Dataset and Data Augmentation Pipeline for Waste Classification.

Applies robust augmentations to simulate real-world classroom lighting,
orientation changes, and scale differences.
"""

import os
import glob
from typing import List, Tuple, Callable, Optional
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SimpleTransform:
    """Robust OpenCV/NumPy based image transformation pipeline."""

    def __init__(self, is_train: bool = True, img_size: int = 224):
        self.is_train = is_train
        self.img_size = img_size

    def __call__(self, img_bgr: np.ndarray) -> torch.Tensor:
        # Resize to standard input dimension
        img = cv2.resize(img_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.is_train:
            # 1. Random horizontal flip
            if np.random.rand() > 0.5:
                img_rgb = np.fliplr(img_rgb)

            # 2. Random rotation (-15 to +15 deg)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15.0, 15.0)
                h, w = img_rgb.shape[:2]
                mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                img_rgb = cv2.warpAffine(img_rgb, mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)

            # 3. Brightness and contrast jitter (simulating classroom lighting)
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.85, 1.15)  # Contrast
                beta = np.random.uniform(-0.1, 0.1)     # Brightness
                img_rgb = np.clip(alpha * img_rgb + beta, 0.0, 1.0)

        # 4. Standardize according to ImageNet statistics
        img_norm = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD

        # 5. Convert (H, W, C) -> PyTorch Tensor (C, H, W)
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        return tensor


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
        self.samples: List[Tuple[str, int]] = []

        self._load_samples(max_samples_per_class)

    def _load_samples(self, max_samples_per_class: Optional[int]):
        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
        for class_id, class_name in enumerate(self.class_names):
            class_folder = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_folder):
                continue

            images = []
            for ext in valid_exts:
                images.extend(glob.glob(os.path.join(class_folder, ext)))

            if max_samples_per_class and len(images) > max_samples_per_class:
                images = images[:max_samples_per_class]

            for img_path in images:
                self.samples.append((img_path, class_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        bgr = cv2.imread(img_path)
        if bgr is None:
            # Fallback black image if corrupt file
            bgr = np.zeros((224, 224, 3), dtype=np.uint8)

        tensor = self.transform(bgr)
        return tensor, label

