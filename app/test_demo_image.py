"""Standalone Single-Image Visual Comparison Demo.

Takes any image (e.g. from data/standard_split/test or data/vn_real_test),
runs both models (TV1 Classical & TV2 Deep Learning), and saves a side-by-side
visual diagnostic image showing:
  - Original cropped ROI
  - Member 1: HOG feature edge overlay + Prediction + Confidence + Bin
  - Member 2: Grad-CAM heatmap overlay + Prediction + Confidence + Bin
"""

import os
import sys
import glob
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.member1_classical.predictor_ml import ClassicalPredictor
from src.member2_deeplearning.predictor_dl import DeepLearningPredictor


def run_image_demo(image_path: str = None, save_out: str = "results/demo_visual_result.png"):
    """Runs comparative inference on a single image and generates visualization."""
    os.makedirs(os.path.dirname(save_out), exist_ok=True)

    # If no path provided, pick a sample from data
    if image_path is None or not os.path.exists(image_path):
        candidates = glob.glob("data/standard_split/test/*/*.*") + glob.glob("data/vn_real_test/*/*.*")
        if not candidates:
            print("[!] No images found in data/. Generating a sample test image...")
            sample = np.full((300, 300, 3), 60, dtype=np.uint8)
            cv2.circle(sample, (150, 150), 80, (0, 140, 255), -1)
            cv2.putText(sample, "CAN", (110, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            test_img_path = "results/sample_input.jpg"
            cv2.imwrite(test_img_path, sample)
            image_path = test_img_path
        else:
            image_path = candidates[0]

    print(f"\n[*] Running Visual Demo on: {image_path}")

    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"[!] Error: Cannot read image at {image_path}")
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

    # Initialize predictors
    p_ml = ClassicalPredictor(weight_path="weights/classical_svm_pipeline.pkl", fallback_mode=True)
    p_dl = DeepLearningPredictor(weight_path="weights/mobilenetv4_best.pth", fallback_mode=True)

    # Inference
    res_ml = p_ml.predict(rgb_224)
    res_dl = p_dl.predict(rgb_224)

    # Ensure UTF-8 output on Windows terminal
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # Print terminal report
    print("\n" + "=" * 65)
    print(f"INPUT IMAGE: {os.path.basename(image_path)}")
    print("=" * 65)
    print(f"1. TV1 (Classical HOG+LBP+SVM):")
    print(f"   - Predicted: {res_ml['class_name'].upper()} ({res_ml['confidence']*100:.1f}%)")
    print(f"   - Latency:   {res_ml['latency_ms']:.2f} ms")
    print(f"   - Action:    {res_ml['action_bin']}")
    print("-" * 65)
    print(f"2. TV2 (Deep Learning CNN + Grad-CAM):")
    print(f"   - Predicted: {res_dl['class_name'].upper()} ({res_dl['confidence']*100:.1f}%)")
    print(f"   - Latency:   {res_dl['latency_ms']:.2f} ms")
    print(f"   - Action:    {res_dl['action_bin']}")
    print("=" * 65)

    # Create visual diagnostic board (3 panels: Original, HOG overlay, Grad-CAM overlay)
    panel_w, panel_h = 320, 320
    orig_panel = cv2.resize(bgr, (panel_w, panel_h))
    hog_bgr = cv2.cvtColor(res_ml["visual_explanation"], cv2.COLOR_RGB2BGR)
    hog_panel = cv2.resize(hog_bgr, (panel_w, panel_h))
    gradcam_bgr = cv2.cvtColor(res_dl["visual_explanation"], cv2.COLOR_RGB2BGR)
    cam_panel = cv2.resize(gradcam_bgr, (panel_w, panel_h))

    # Add labels
    def add_banner(img, title, subtitle, color):
        res = img.copy()
        cv2.rectangle(res, (0, 0), (panel_w, 65), (20, 20, 20), -1)
        cv2.putText(res, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(res, subtitle, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return res

    p1 = add_banner(orig_panel, "1. Anh Goc Dau Vao", "Original ROI", (200, 200, 200))
    p2 = add_banner(hog_panel, "2. TV1: HOG + SVM", f"{res_ml['class_name'].upper()} ({res_ml['confidence']*100:.1f}%)", (0, 200, 255))
    p3 = add_banner(cam_panel, "3. TV2: CNN + Grad-CAM", f"{res_dl['class_name'].upper()} ({res_dl['confidence']*100:.1f}%)", (0, 255, 120))

    board = np.hstack([p1, p2, p3])
    cv2.imwrite(save_out, board)
    print(f"\n[+] Visual comparison board saved to: {save_out}")
    return save_out


if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_image_demo(image_path=img_arg)
