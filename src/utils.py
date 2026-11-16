"""Shared utilities for path resolution, config loading, and Unicode-safe image I/O."""

import glob
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")


def resolve_path(path: str) -> str:
    """Resolves a path relative to PROJECT_ROOT if it is not already absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file."""
    abs_path = resolve_path(config_path)
    if not os.path.exists(abs_path):
        return {}
    with open(abs_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_image_bgr(path: str) -> Optional[np.ndarray]:
    """Reads an image as BGR. Uses np.fromfile to support Unicode/Vietnamese Windows paths."""
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def read_image_rgb(path: str) -> Optional[np.ndarray]:
    """Reads an image as RGB. Safe for Unicode/Vietnamese Windows paths."""
    bgr = read_image_bgr(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_image(path: str, image_bgr: np.ndarray) -> bool:
    """Writes a BGR image to disk safely on Unicode/Vietnamese Windows paths."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, image_bgr)
    if not ok:
        return False
    buf.tofile(path)
    return True


def list_class_images(
    data_dir: str,
    class_names: Sequence[str],
    max_per_class: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """Lists (image_path, class_id) pairs without duplicates on case-insensitive OS."""
    abs_dir = resolve_path(data_dir)
    items: List[Tuple[str, int]] = []
    for class_id, class_name in enumerate(class_names):
        class_folder = os.path.join(abs_dir, class_name)
        if not os.path.isdir(class_folder):
            continue
        files: List[str] = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(glob.glob(os.path.join(class_folder, f"*.{ext}")))
        unique_files = sorted(set(os.path.normpath(f) for f in files))
        if max_per_class is not None:
            unique_files = unique_files[:max_per_class]
        for f in unique_files:
            items.append((f, class_id))
    return items


def set_seed(seed: int = 42) -> None:
    """Sets random seed across random, NumPy, and PyTorch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
