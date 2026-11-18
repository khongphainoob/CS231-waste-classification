"""Standalone Single-Image Visual Comparison Demo."""

import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.member1_classical.predictor_ml import ClassicalPredictor
from src.member2_deeplearning.predictor_dl import DeepLearningPredictor
from src.utils import read_image_rgb, resolve_path, write_image


def run_image_demo(image_path: str = None, save_out: str = "results/demo_visual_result.png"):
    """Runs comparative inference on a single image and generates visualization."""
    save_abs = resolve_path(save_out)
    os.makedirs(os.path.dirname(save_abs), exist_ok=True)

    if image_path is None or not os.path.exists(resolve_path(image_path)):
        test_glob = glob.glob(os.path.join(resolve_path("data/standard_split/test"), "*", "*.*"))
        vn_glob = glob.glob(os.path.join(resolve_path("data/vn_real_test"), "*", "*.*"))
        candidates = sorted(set(test_glob + vn_glob))
        if not candidates:
            print("[!] No images found in data/. Generating a sample test image...")
            sample = np.full((300, 300, 3), 60, dtype=np.uint8)
            cv2.circle(sample, (150, 150), 80, (0, 140, 255), -1)
            cv2.putText(sample, "CAN", (110, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            image_path = resolve_path("results/sample_input.jpg")
            write_image(image_path, sample)
        else:
            image_path = candidates[0]
    else:
        image_path = resolve_path(image_path)

    print(f"\n[*] Running Visual Demo on: {image_path}")
    rgb = read_image_rgb(image_path)
    if rgb is None:
        print(f"[!] Error: Cannot read image at {image_path}")
        return

    rgb_224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    bgr_224 = cv2.cvtColor(rgb_224, cv2.COLOR_RGB2BGR)

    p_ml = ClassicalPredictor(weight_path="weights/classical_svm_pipeline.pkl")
    p_dl = DeepLearningPredictor(weight_path="weights/mobilenetv4_best.pth", fallback_mode=False)

    res_ml = p_ml.predict(rgb_224)
    res_dl = p_dl.predict(rgb_224)

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 65)
    print(f"INPUT IMAGE: {os.path.basename(image_path)}")
    print("=" * 65)
    print("1. TV1 (Classical HSV+LBP+HOG + SVM):")
    print(f"   - Predicted: {res_ml['class_name'].upper()} ({res_ml['confidence']*100:.1f}%)")
    print(f"   - Latency:   {res_ml['latency_ms']:.2f} ms")
    print(f"   - Action:    {res_ml['action_bin']}")
    print("-" * 65)
    print(f"2. TV2 (Deep Learning {p_dl.model_name} + Grad-CAM):")
    print(f"   - Predicted: {res_dl['class_name'].upper()} ({res_dl['confidence']*100:.1f}%)")
    print(f"   - Latency:   {res_dl['latency_ms']:.2f} ms")
    print(f"   - Action:    {res_dl['action_bin']}")
    print("=" * 65)

    panel_w, panel_h = 320, 320
    orig_panel = cv2.resize(bgr_224, (panel_w, panel_h))
    hog_panel = cv2.resize(cv2.cvtColor(res_ml["visual_explanation"], cv2.COLOR_RGB2BGR), (panel_w, panel_h))
    cam_panel = cv2.resize(cv2.cvtColor(res_dl["visual_explanation"], cv2.COLOR_RGB2BGR), (panel_w, panel_h))

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
    write_image(save_abs, board)
    print(f"\n[+] Visual comparison board saved to: {save_abs}")
    return save_abs


if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_image_demo(image_path=img_arg)
