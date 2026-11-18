"""Shared-ImageNet preprocessing and conservative train-only augmentation.
The interface accepts BGR arrays (as read by cv2) to retain repo compatibility.
"""
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([.485, .456, .406], dtype=np.float32)
IMAGENET_STD = np.array([.229, .224, .225], dtype=np.float32)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class SimpleTransform:
    def __init__(self, is_train=True, img_size=224):
        self.is_train = bool(is_train)
        self.img_size = int(img_size)

    def __call__(self, img_bgr: np.ndarray) -> torch.Tensor:
        if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError("Expected valid 3-channel BGR image")
        img = cv2.resize(img_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.is_train:
            if np.random.rand() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1])
            if np.random.rand() < 0.5:
                angle = float(np.random.uniform(-15, 15))
                m = cv2.getRotationMatrix2D((self.img_size / 2, self.img_size / 2), angle, 1.0)
                rgb = cv2.warpAffine(rgb, m, (self.img_size, self.img_size),
                                     borderMode=cv2.BORDER_REFLECT_101)
            if np.random.rand() < 0.7:
                alpha = float(np.random.uniform(.85, 1.15))
                beta = float(np.random.uniform(-.08, .08))
                rgb = np.clip(rgb * alpha + beta, 0., 1.)
        normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).float()


def get_transforms(is_train=True, img_size=224) -> Callable:
    return SimpleTransform(is_train, img_size)


class WasteDataset(Dataset):
    def __init__(self, data_dir: str, class_names: List[str],
                 transform: Optional[Callable] = None,
                 max_samples_per_class: Optional[int] = None):
        self.data_dir = str(data_dir)
        self.class_names = list(class_names)
        self.transform = transform or get_transforms(False)
        self.samples: List[Tuple[str, int]] = []
        for class_id, name in enumerate(self.class_names):
            folder = Path(self.data_dir) / name
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing class folder: {folder}")
            paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if max_samples_per_class is not None:
                paths = paths[:max_samples_per_class]
            self.samples.extend((str(p), class_id) for p in paths)
        if not self.samples:
            raise ValueError(f"No images found in {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable/corrupted image: {path}")
        return self.transform(image), label
