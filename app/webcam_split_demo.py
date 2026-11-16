"""Live OpenCV Dual-Screen Split Webcam Demo.

Displays real-time side-by-side comparison between:
  - Left Pane: Member 1 (Classical CV: HSV+LBP+HOG + SVM)
  - Right Pane: Member 2 (Deep Learning: CNN + Real-time Grad-CAM)
"""

import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.member1_classical.predictor_ml import ClassicalPredictor
from src.member2_deeplearning.predictor_dl import DeepLearningPredictor
from src.utils import load_config, resolve_path, write_image


def draw_panel(
    frame_bgr: np.ndarray,
    title: str,
    result: dict,
    accent_color: tuple = (0, 255, 0),
) -> np.ndarray:
    """Renders diagnostic overlay and prediction text on an image pane."""
    panel = frame_bgr.copy()
    h, w = panel.shape[:2]

    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, panel, 0.25, 0, panel)

    cv2.putText(panel, title, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    class_name = result["class_name"].upper()
    conf = result["confidence"] * 100
    latency = result["latency_ms"]
    fps = 1000.0 / latency if latency > 0 else 0.0

    cv2.putText(panel, f"PRED: {class_name} ({conf:.1f}%)", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, accent_color, 2)
    cv2.putText(panel, f"{latency:.1f}ms | {fps:.1f} FPS", (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    action_bin = result.get("action_bin", "")
    if "Hữu Cơ" in action_bin or "Huu Co" in action_bin:
        bin_ascii, bin_color = "THUNG HUU CO (XANH LA)", (80, 240, 80)
    else:
        bin_ascii, bin_color = "THUNG TAI CHE (TRANG/XANH)", (255, 200, 80)

    cv2.putText(panel, f"-> {bin_ascii}", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bin_color, 2)
    return panel


def run_webcam_split_demo(cam_index: int = None):
    """Launches the real-time comparative webcam window."""
    cfg = load_config("configs/config.yaml")
    cam_cfg = cfg.get("webcam", {})
    if cam_index is None:
        cam_index = cam_cfg.get("camera_index", 0)
    frame_w = cam_cfg.get("frame_width", 640)
    frame_h = cam_cfg.get("frame_height", 480)
    roi_size = cam_cfg.get("roi_size", 300)

    svm_path = cfg.get("classical", {}).get("weight_path", "weights/classical_svm_pipeline.pkl")
    dl_path = cfg.get("deep_learning", {}).get("weight_path", "weights/mobilenetv4_best.pth")

    print("\n[*] Initializing Dual-Screen Webcam Demo...")
    print("    - Press 'q' or 'ESC' to exit")
    print("    - Press 's' to snapshot test frame")

    p_ml = ClassicalPredictor(weight_path=svm_path)
    p_dl = DeepLearningPredictor(weight_path=dl_path, fallback_mode=True)

    cap = cv2.VideoCapture(cam_index)
    use_synth = not cap.isOpened()
    if use_synth:
        print(f"[!] Camera {cam_index} unavailable. Using synthetic test mode.")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

    snap_dir = resolve_path("results/snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    snapshot_idx = 0

    try:
        while True:
            if not use_synth:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[!] Frame grab failed.")
                    break
            else:
                frame = np.full((frame_h, frame_w, 3), 40, dtype=np.uint8)
                cv2.circle(frame, (frame_w // 2, frame_h // 2), 90, (0, 165, 255), -1)
                cv2.putText(frame, "CAN / METAL", (frame_w // 2 - 60, frame_h // 2 + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                time.sleep(0.03)

            h, w = frame.shape[:2]
            x1 = max(0, (w - roi_size) // 2)
            y1 = max(0, (h - roi_size) // 2)
            x2 = min(w, x1 + roi_size)
            y2 = min(h, y1 + roi_size)

            roi_bgr = frame[y1:y2, x1:x2]
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            roi_224 = cv2.resize(roi_rgb, (224, 224), interpolation=cv2.INTER_AREA)

            res_ml = p_ml.predict(roi_224)
            res_dl = p_dl.predict(roi_224)

            viz_ml_bgr = cv2.resize(cv2.cvtColor(res_ml["visual_explanation"], cv2.COLOR_RGB2BGR), (420, 420))
            viz_dl_bgr = cv2.resize(cv2.cvtColor(res_dl["visual_explanation"], cv2.COLOR_RGB2BGR), (420, 420))

            pane_left = draw_panel(viz_ml_bgr, "TV1: HSV+LBP+HOG + SVM", res_ml, accent_color=(255, 200, 0))
            pane_right = draw_panel(viz_dl_bgr, f"TV2: {p_dl.model_name} + Grad-CAM", res_dl, accent_color=(0, 255, 128))
            split_screen = np.hstack([pane_left, pane_right])

            frame_guide = frame.copy()
            cv2.rectangle(frame_guide, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_guide, "Center Target ROI (Place waste item here)", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("CS231 - Waste Classification Dual-Screen Comparison", split_screen)
            cv2.imshow("Webcam Input Guide", cv2.resize(frame_guide, (480, 360)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("s"):
                snapshot_path = os.path.join(snap_dir, f"snapshot_{snapshot_idx:03d}.jpg")
                write_image(snapshot_path, roi_bgr)
                print(f"[+] Saved snapshot to: {snapshot_path}")
                snapshot_idx += 1
    finally:
        if not use_synth:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_split_demo()
