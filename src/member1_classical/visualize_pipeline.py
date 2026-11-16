"""Vẽ từng bước xử lý của nhánh cổ điển trên 1 ảnh (dùng cho slide / demo / vấn đáp).

Hàng trên : Ảnh gốc | Ảnh xám (đã làm mờ) | Ảnh mã LBP | Các nét HOG đè lên ảnh
Hàng dưới : Histogram HSV (512) | Histogram LBP (26) | Vector HOG (1764) | Xác suất 6 lớp của SVM

Chạy:
    python src/member1_classical/visualize_pipeline.py --image duong_dan_anh.jpg
    python src/member1_classical/visualize_pipeline.py --cls metal      (lấy ngẫu nhiên 1 ảnh test lớp metal)
"""

import os
import sys
import glob
import random
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
from src.base_predictor import CLASS_NAMES
from src.member1_classical.feature_extractor import WasteFeatureExtractor, read_image_rgb, hog_overlay


def hsv_bin_colors(extractor):
    """Màu đại diện của từng ô histogram HSV (để tô màu cột cho dễ nhìn)."""
    colors = []
    for h in range(extractor.h_bins):
        for s in range(extractor.s_bins):
            for v in range(extractor.v_bins):
                colors.append(hsv_to_rgb([
                    (h + 0.5) / extractor.h_bins,
                    (s + 0.5) / extractor.s_bins,
                    (v + 0.5) / extractor.v_bins,
                ]))
    return colors


def draw_pipeline(image_rgb, extractor, predictor=None, title="", save_path=None, show=False):
    features, st = extractor.extract_with_steps(image_rgb)

    fig, ax = plt.subplots(2, 4, figsize=(18, 8.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ----- Hàng 1: ảnh trung gian -----
    ax[0, 0].imshow(st["rgb"])
    ax[0, 0].set_title("B1. Ảnh đầu vào 224x224")
    ax[0, 1].imshow(st["gray"], cmap="gray")
    ax[0, 1].set_title("B1. Ảnh xám + Gaussian 3x3")
    ax[0, 2].imshow(st["lbp_image"], cmap="nipy_spectral")
    ax[0, 2].set_title(f"B3. Ảnh mã LBP (P={extractor.lbp_points}, R={extractor.lbp_radius})")
    ax[0, 3].imshow(hog_overlay(st["rgb"], st["hog_image"]))
    ax[0, 3].set_title(f"B4. HOG (ô {extractor.hog_pixels_per_cell}x{extractor.hog_pixels_per_cell} trên ảnh {extractor.hog_size})")
    for a in ax[0]:
        a.axis("off")

    # ----- Hàng 2: vector đặc trưng -----
    f_color = st["f_color"]
    ax[1, 0].bar(range(len(f_color)), f_color, color=hsv_bin_colors(extractor), width=3.0)
    ax[1, 0].plot(f_color, color="black", linewidth=0.4)  # viền mảnh để thấy cả các ô màu nhạt
    ax[1, 0].set_title(f"B2. Histogram HSV ({len(f_color)} chiều)")
    ax[1, 0].set_xlabel("ô (H, S, V)  - tô theo màu của ô")

    f_lbp = st["f_lbp"]
    ax[1, 1].bar(range(len(f_lbp)), f_lbp, color="tab:purple")
    ax[1, 1].set_title(f"B3. Histogram LBP ({len(f_lbp)} chiều)")
    ax[1, 1].set_xlabel(f"mã LBP uniform 0..{len(f_lbp) - 2}, cột cuối = non-uniform")

    f_hog = st["f_hog"]
    ax[1, 2].plot(f_hog, color="tab:orange", linewidth=0.5)
    ax[1, 2].set_title(f"B4. Vector HOG ({len(f_hog)} chiều)")
    ax[1, 2].set_xlabel(f"Tổng vector sau khi nối: {len(features)} chiều")

    if predictor is not None:
        res = predictor.predict(st["rgb"])
        probs = res["probabilities"]
        colors = ["tab:green" if i == res["class_id"] else "tab:gray" for i in range(len(probs))]
        ax[1, 3].barh(CLASS_NAMES, probs, color=colors)
        ax[1, 3].set_xlim(0, 1)
        ax[1, 3].invert_yaxis()
        ax[1, 3].set_title(f"B5. SVM: {res['class_name']} ({res['confidence'] * 100:.1f}%)")
        ax[1, 3].set_xlabel(f"{res['action_bin']}  |  {res['latency_ms']:.0f} ms")
    else:
        ax[1, 3].text(0.5, 0.5, "Chưa có model\n(chạy train_ml.py trước)", ha="center", va="center")
        ax[1, 3].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=130)
        print("Đã lưu:", save_path)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Đường dẫn 1 ảnh bất kỳ")
    parser.add_argument("--cls", default=None, choices=CLASS_NAMES, help="Lấy ngẫu nhiên 1 ảnh test của lớp này")
    parser.add_argument("--weights", default="weights/classical_svm_pipeline.pkl")
    parser.add_argument("--out", default=None, help="Nơi lưu hình (mặc định results/member1/pipeline_<tên>.png)")
    parser.add_argument("--show", action="store_true", help="Mở cửa sổ xem hình")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    if args.image:
        path = args.image
    else:
        cls = args.cls or "metal"
        files = sorted(glob.glob(os.path.join(ROOT, "data", "standard_split", "test", cls, "*")))
        if not files:
            print("Không tìm thấy ảnh test, hãy chạy data/download_and_split.py trước.")
            return
        random.seed(args.seed)
        path = random.choice(files)

    image = read_image_rgb(path)
    if image is None:
        print("Không đọc được ảnh:", path)
        return

    predictor = None
    weight_path = os.path.join(ROOT, args.weights)
    if os.path.exists(weight_path):
        from src.member1_classical.predictor_ml import ClassicalPredictor
        predictor = ClassicalPredictor(weight_path)
        extractor = predictor.extractor
    else:
        extractor = WasteFeatureExtractor()

    name = os.path.splitext(os.path.basename(path))[0]
    out = args.out or os.path.join(ROOT, "results", "member1", f"pipeline_{name}.png")
    draw_pipeline(image, extractor, predictor, title=f"Quy trình nhánh cổ điển - {name}", save_path=out, show=args.show)


if __name__ == "__main__":
    main()
