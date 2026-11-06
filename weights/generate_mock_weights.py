"""Generates lightweight valid initial model weights for smoke testing.

Ensures that the entire repo, demo webcam, and streamlit dashboard can run immediately
out-of-the-box before the full 4,800 images finish training.
"""

import os
import sys
import pickle
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.base_predictor import CLASS_NAMES
from src.member1_classical.predictor_ml import MockClassicalPipeline
from src.member2_deeplearning.train_cnn import LightWasteCNN


def generate_classical_mock_weights(save_path: str = "weights/classical_svm_pipeline.pkl"):
    """Creates a mock fitted pipeline with 6 classes."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mock_pipe = MockClassicalPipeline(n_classes=len(CLASS_NAMES))
    with open(save_path, "wb") as f:
        pickle.dump(mock_pipe, f)
    size_kb = os.path.getsize(save_path) / 1024
    print(f"[+] Created classical weights: {save_path} ({size_kb:.2f} KB)")


def generate_dl_mock_weights(save_path: str = "weights/mobilenetv4_best.pth"):
    """Creates a valid PyTorch checkpoint for LightWasteCNN."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.manual_seed(42)

    model = LightWasteCNN(num_classes=6)
    checkpoint = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "val_acc": 0.85,
        "model_class": "LightWasteCNN",
        "class_names": CLASS_NAMES,
    }
    torch.save(checkpoint, save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[+] Created deep learning weights: {save_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    generate_classical_mock_weights()
    generate_dl_mock_weights()
