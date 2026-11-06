"""Generator script to create valid Jupyter Notebooks for the project."""
import json
import os

def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.11"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

# 1. EDA Notebook
nb1 = make_nb([
    md("# CS231 - Đồ án Thị Giác Máy Tính: 01. Exploratory Data Analysis (EDA)\n## Phân loại rác thải rắn sinh hoạt đơn lẻ tại nguồn"),
    code("""import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

with open('../configs/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

classes = list(config['dataset']['classes'].values())
print('Target Classes:', classes)"""),
    md("### 1. Thống kê số lượng ảnh theo từng lớp\nKiểm tra tính cân bằng dữ liệu giữa các lớp trên tập Train."),
    code("""train_dir = '../data/standard_split/train'
counts = {}
for c in classes:
    files = glob.glob(os.path.join(train_dir, c, '*.*'))
    counts[c] = len(files)

plt.figure(figsize=(9, 4))
colors = ['#D2691E', '#4682B4', '#708090', '#F5DEB3', '#00CED1', '#2E8B57']
plt.bar(counts.keys(), counts.values(), color=colors)
plt.title('Phân bố số lượng ảnh tập Train (560 ảnh / lớp chuẩn)')
plt.ylabel('Số lượng')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()"""),
    md("### 2. Trực quan hóa mẫu ảnh tiêu biểu của 6 lớp rác"),
    code("""fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for i, c in enumerate(classes):
    img_files = glob.glob(os.path.join(train_dir, c, '*.*'))
    if img_files:
        bgr = cv2.imread(img_files[0])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        axes[i].imshow(rgb)
        bin_guide = config['dataset']['bin_mapping'].get(c, '')
        axes[i].set_title(f"Class: {c}\\n[{bin_guide}]", fontsize=10)
    axes[i].axis('off')

plt.tight_layout()
plt.show()"""),
    md("### 3. Phân tích không gian màu: RGB vs HSV\nChứng minh tại sao HSV Histogram hiệu quả trong việc phân biệt rác hữu cơ (biological) và bìa carton (cardboard)."),
    code("""bio_files = glob.glob(os.path.join(train_dir, 'biological', '*.*'))
if bio_files:
    bgr = cv2.imread(bio_files[0])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(rgb); ax[0].set_title('RGB Image'); ax[0].axis('off')
    ax[1].imshow(hsv[:, :, 0], cmap='hsv'); ax[1].set_title('Hue Channel (Sắc độ màu)'); ax[1].axis('off')
    ax[2].imshow(hsv[:, :, 1], cmap='viridis'); ax[2].set_title('Saturation Channel (Độ bão hòa)'); ax[2].axis('off')
    ax[3].imshow(hsv[:, :, 2], cmap='gray'); ax[3].set_title('Value Channel (Cường độ sáng)'); ax[3].axis('off')
    plt.tight_layout()
    plt.show()""")
])

# 2. Member 1 Classical Notebook
nb2 = make_nb([
    md("# CS231 - Thành viên 1: Trích xuất đặc trưng Thủ công (Hand-crafted) & Máy học (SVM / XGBoost)"),
    code("""import sys, os
sys.path.append('..')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.member1_classical.feature_extractor import WasteFeatureExtractor

extractor = WasteFeatureExtractor()
print("Initialized WasteFeatureExtractor (HSV 3D + LBP + HOG)")"""),
    md("### 1. Trích xuất và trực quan hóa 3 nhóm đặc trưng trên cùng 1 ảnh"),
    code("""import glob
sample_files = glob.glob('../data/standard_split/train/metal/*.*')
if sample_files:
    bgr = cv2.imread(sample_files[0])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    features, viz_blended = extractor.extract_features(rgb, return_viz=True)
    
    print(f"Tổng số chiều vector đặc trưng kết hợp: {len(features)}")
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rgb); ax[0].set_title("Ảnh gốc (Metal Can)"); ax[0].axis('off')
    ax[1].imshow(viz_blended); ax[1].set_title("Biểu diễn đặc trưng HOG Gradients"); ax[1].axis('off')
    plt.tight_layout()
    plt.show()"""),
    md("### 2. Huấn luyện Pipeline (StandardScaler + PCA + SVM/XGBoost)"),
    code("""from src.member1_classical.train_ml import train_classical_pipeline

# Huấn luyện pipeline và lưu trọng số vào weights/classical_svm_pipeline.pkl
train_classical_pipeline(config_path='../configs/config.yaml', model_choice='svm')"""),
    md("### 3. Kiểm thử suy luận với ClassicalPredictor"),
    code("""from src.member1_classical.predictor_ml import ClassicalPredictor

predictor = ClassicalPredictor(weight_path='../weights/classical_svm_pipeline.pkl')
if sample_files:
    result = predictor.predict(cv2.resize(rgb, (224, 224)))
    print("Dự đoán:", result['class_name'])
    print("Độ tự tin:", result['confidence'])
    print("Độ trễ:", result['latency_ms'], "ms")
    print("Thùng rác kiến nghị:", result['action_bin'])""")
])

# 3. Member 2 Deep Learning Notebook
nb3 = make_nb([
    md("# CS231 - Thành viên 2: Mạng nơ-ron Tích chập Sâu (CNN) & Giải thích AI (Grad-CAM)"),
    code("""import sys, os
sys.path.append('..')

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.member2_deeplearning.dataset_aug import WasteDataset, get_transforms
from src.member2_deeplearning.train_cnn import train_cnn_model
from src.base_predictor import CLASS_NAMES

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())"""),
    md("### 1. Kiểm tra Pipeline Tăng cường Dữ liệu (Data Augmentation)"),
    code("""train_dir = '../data/standard_split/train'
dataset = WasteDataset(train_dir, class_names=CLASS_NAMES, transform=get_transforms(is_train=True))

if len(dataset) > 0:
    tensor, label = dataset[0]
    print(f"Tensor shape: {tensor.shape}, Label: {CLASS_NAMES[label]}")"""),
    md("### 2. Huấn luyện Mô hình CNN"),
    code("""# Huấn luyện mạng CNN và lưu trọng số tốt nhất vào weights/mobilenetv4_best.pth
train_cnn_model(config_path='../configs/config.yaml', model_name='light_cnn', epochs=15, batch_size=32)"""),
    md("### 3. Trực quan hóa Giải thích bằng Grad-CAM"),
    code("""from src.member2_deeplearning.predictor_dl import DeepLearningPredictor

predictor = DeepLearningPredictor(weight_path='../weights/mobilenetv4_best.pth')

import glob
test_samples = glob.glob('../data/standard_split/test/plastic/*.*')
if test_samples:
    bgr = cv2.imread(test_samples[0])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_224 = cv2.resize(rgb, (224, 224))
    
    result = predictor.predict(rgb_224)
    print("CNN Predicts:", result['class_name'], "| Conf:", result['confidence'])
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rgb_224); ax[0].set_title("Ảnh đầu vào (Plastic Bottle)"); ax[0].axis('off')
    ax[1].imshow(result['visual_explanation']); ax[1].set_title("Vùng chú ý Grad-CAM"); ax[1].axis('off')
    plt.tight_layout()
    plt.show()""")
])

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/01_EDA_Data_Visualization.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb1, f, indent=2)

with open('notebooks/02_Member1_Classical_HOG_LBP_ML.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb2, f, indent=2)

with open('notebooks/03_Member2_DeepLearning_CNN_GradCAM.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb3, f, indent=2)

print("[+] Successfully generated all 3 Jupyter Notebooks!")

