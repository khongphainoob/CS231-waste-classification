"""Dataset Acquisition, Mapping, and Deterministic Splitting Script.

Functions:
  1. Downloads/processes the Kaggle Garbage Classification dataset (12 classes).
  2. Maps and aggregates into 6 target categories:
       - cardboard  -> cardboard
       - brown-glass, green-glass, white-glass -> glass
       - metal      -> metal
       - paper      -> paper
       - plastic    -> plastic
       - biological -> biological
  3. Samples 800 balanced images per class (Total: 4,800 images).
  4. Splits deterministically with seed=42:
       - Train: 560/class (3,360 total)
       - Val:   120/class (720 total)
       - Test:  120/class (720 total)
  5. Initializes data/vn_real_test structure for Vietnam real-world photos.
"""

import os
import sys
import shutil
import random
import glob
import argparse
from typing import Dict, List
import numpy as np
import cv2

# Target Classes
TARGET_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "biological"
]

# Source folder mapping for Kaggle dataset
CLASS_MAPPING = {
    "cardboard": ["cardboard"],
    "glass": ["brown-glass", "green-glass", "white-glass"],
    "metal": ["metal"],
    "paper": ["paper"],
    "plastic": ["plastic"],
    "biological": ["biological"]
}


def create_directories(base_dir: str = "data/standard_split"):
    """Creates directory layout for train, val, and test splits."""
    for split in ["train", "val", "test"]:
        for cls in TARGET_CLASSES:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

    for cls in TARGET_CLASSES:
        os.makedirs(os.path.join("data/vn_real_test", cls), exist_ok=True)

    print("[+] Verified target dataset directory tree.")


def process_and_split(
    raw_dir: str,
    target_base: str = "data/standard_split",
    seed: int = 42,
    samples_per_class: int = 800
):
    """Processes source images, aggregates classes, and splits deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    create_directories(target_base)

    train_count = 560
    val_count = 120
    test_count = 120

    print(f"\n[*] Processing and partitioning data from: {raw_dir}")
    print(f"[*] Target split per class: Train={train_count}, Val={val_count}, Test={test_count} (Total={samples_per_class})")

    for target_cls, src_subdirs in CLASS_MAPPING.items():
        all_candidate_files = []
        for src_sub in src_subdirs:
            folder = os.path.join(raw_dir, src_sub)
            if os.path.isdir(folder):
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
                    all_candidate_files.extend(glob.glob(os.path.join(folder, ext)))

        if len(all_candidate_files) == 0:
            print(f"[!] Warning: No images found for class '{target_cls}' in subfolders {src_subdirs}.")
            continue

        random.shuffle(all_candidate_files)
        selected_files = all_candidate_files[:samples_per_class]

        train_files = selected_files[:train_count]
        val_files = selected_files[train_count:train_count + val_count]
        test_files = selected_files[train_count + val_count:train_count + val_count + test_count]

        # Copy to destinations
        for split_name, file_list in [("train", train_files), ("val", val_files), ("test", test_files)]:
            dest_dir = os.path.join(target_base, split_name, target_cls)
            for idx, src_file in enumerate(file_list):
                ext = os.path.splitext(src_file)[1]
                dest_filename = f"{target_cls}_{idx:04d}{ext}"
                dest_path = os.path.join(dest_dir, dest_filename)
                shutil.copyfile(src_file, dest_path)

        print(f"[+] Class '{target_cls}': Copied {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")

    print("\n[+] Dataset partitioning complete!")


def generate_synthetic_data(target_base: str = "data/standard_split", per_class: int = 30):
    """Generates synthetic colored pattern images for offline local pipeline smoke tests."""
    print(f"\n[*] Generating synthetic test data ({per_class} per class)...")
    create_directories(target_base)

    color_schemes = {
        "cardboard": (160, 130, 90),   # Brownish
        "glass": (220, 230, 240),       # Translucent cyan-white
        "metal": (180, 180, 190),       # Silver gray
        "paper": (245, 245, 245),       # White
        "plastic": (70, 150, 220),      # Blue bottle
        "biological": (50, 180, 50),    # Green leaf
    }

    n_train = int(per_class * 0.7)
    n_val = int(per_class * 0.15)
    n_test = per_class - n_train - n_val

    for cls, base_color in color_schemes.items():
        for split, count in [("train", n_train), ("val", n_val), ("test", n_test)]:
            dest_dir = os.path.join(target_base, split, cls)
            for i in range(count):
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                noise = np.random.randint(-20, 20, (224, 224, 3))
                c_val = np.clip(np.array(base_color) + noise, 0, 255).astype(np.uint8)
                img[:] = c_val
                # Add shape signature
                cv2.circle(img, (112, 112), 40 + (i % 10), (0, 0, 0), 2)
                cv2.imwrite(os.path.join(dest_dir, f"synth_{cls}_{i:03d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Also generate samples in vn_real_test
    for cls, base_color in color_schemes.items():
        dest_dir = os.path.join("data/vn_real_test", cls)
        for i in range(10):
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            noise = np.random.randint(-15, 15, (224, 224, 3))
            img[:] = np.clip(np.array(base_color) + noise, 0, 255).astype(np.uint8)
            cv2.putText(img, f"VN-{cls[:3]}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.imwrite(os.path.join(dest_dir, f"vn_{cls}_{i:03d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print("[+] Synthetic dataset generated successfully for instant pipeline testing!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw/garbage-classification", help="Path to unzipped raw Kaggle dataset")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test data for offline debugging")
    parser.add_argument("--samples_per_class", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_data()
    else:
        if os.path.isdir(args.raw_dir):
            process_and_split(args.raw_dir, samples_per_class=args.samples_per_class, seed=args.seed)
        else:
            print(f"[!] Directory not found: {args.raw_dir}")
            print("    Run with '--synthetic' to create local dummy images for instant verification, or")
            print("    download Kaggle dataset first:")
            print("    kaggle datasets download -d mostafaabla/garbage-classification -p data/raw/ --unzip")
            create_directories()

