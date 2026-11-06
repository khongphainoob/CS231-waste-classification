"""Live OpenCV Dual-Screen Split Webcam Demo.

Displays real-time side-by-side comparison between:
  - Left Pane: Member 1 (Classical CV: HSV+LBP+HOG + SVM)
  - Right Pane: Member 2 (Deep Learning: CNN + Real-time Grad-CAM)
Includes center bounding box ROI guide (300x300) and Vietnamese action bin instructions.
"""

import os
import sys
import time
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.member1_classical.predictor_ml import ClassicalPredictor
from src.member2_deeplearning.predictor_dl import DeepLearningPredictor


def draw_panel(
    frame_bgr: np.ndarray,
    title: str,
    result: dict,
    accent_color: tuple = (0, 255, 0)
) -> np.ndarray:
    """Renders diagnostic overlay and prediction text on an image pane."""
    panel = frame_bgr.copy()
    h, w = panel.shape[:2]

    # Semi-transparent top banner
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, panel, 0.25, 0, panel)

    # Header title
    cv2.putText(panel, title, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Prediction info
    class_name = result["class_name"].upper()
    conf = result["confidence"] * 100
    latency = result["latency_ms"]
    fps = 1000.0 / latency if latency > 0 else 0

    pred_str = f"PRED: {class_name} ({conf:.1f}%)"
    cv2.putText(panel, pred_str, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, accent_color, 2)

    perf_str = f"{latency:.1f}ms | {fps:.1f} FPS"
    cv2.putText(panel, perf_str, (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Bin recommendation (Convert to clean ASCII for OpenCV FONT rendering)
    action_bin = result.get("action_bin", "")
    if "Hữu Cơ" in action_bin or "Huu Co" in action_bin:
        bin_ascii = "THUNG HUU CO (XANH LA)"
        bin_color = (80, 240, 80)
    else:
        bin_ascii = "THUNG TAI CHE (TRANG/XANH)"
        bin_color = (255, 200, 80)

    cv2.putText(panel, f"-> {bin_ascii}", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bin_color, 2)

    return panel


def run_webcam_split_demo(cam_index: int = 0):
    """Launches the real-time comparative webcam window."""
    print("\n[*] Initializing Dual-Screen Webcam Demo...")
    print("    - Press 'q' or 'ESC' to exit")
    print("    - Press 's' to snapshot test frame")

    # Load predictors
    p_ml = ClassicalPredictor(weight_path="weights/classical_svm_pipeline.pkl", fallback_mode=True)
    p_dl = DeepLearningPredictor(weight_path="weights/mobilenetv4_best.pth", fallback_mode=True)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[!] Error: Could not open camera {cam_index}.")
        print("[i] Using synthetic camera test mode.")
        use_synth = True
    else:
        use_synth = False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    roi_size = 300
    os.makedirs("results/snapshots", exist_ok=True)
    snapshot_idx = 0

    while True:
        if not use_synth:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[!] Frame grab failed.")
                break
        else:
            # Synthetic animated frame for environments without active webcam
            frame = np.full((480, 640, 3), 40, dtype=np.uint8)
            cv2.circle(frame, (320, 240), 90, (0, 165, 255), -1)
            cv2.putText(frame, "CAN / METAL", (260, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            time.sleep(0.03)

        h, w = frame.shape[:2]
        x1 = max(0, (w - roi_size) // 2)
        y1 = max(0, (h - roi_size) // 2)
        x2 = min(w, x1 + roi_size)
        y2 = min(h, y1 + roi_size)

        # Crop center ROI
        roi_bgr = frame[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi_224 = cv2.resize(roi_rgb, (224, 224))

        # Run both predictors
        res_ml = p_ml.predict(roi_224)
        res_dl = p_dl.predict(roi_224)

        # Get visual explanations (BGR for OpenCV display)
        viz_ml_bgr = cv2.cvtColor(res_ml["visual_explanation"], cv2.COLOR_RGB2BGR)
        viz_dl_bgr = cv2.cvtColor(res_dl["visual_explanation"], cv2.COLOR_RGB2BGR)

        # Resize to standard panel dimensions
        panel_w, panel_h = 420, 420
        viz_ml_panel = cv2.resize(viz_ml_bgr, (panel_w, panel_h))
        viz_dl_panel = cv2.resize(viz_dl_bgr, (panel_w, panel_h))

        # Render diagnostic text
        pane_left = draw_panel(viz_ml_panel, "TV1: HOG + LBP + SVM", res_ml, accent_color=(255, 200, 0))
        pane_right = draw_panel(viz_dl_panel, "TV2: CNN + Grad-CAM", res_dl, accent_color=(0, 255, 128))

        # Combine side-by-side
        split_screen = np.hstack([pane_left, pane_right])

        # Draw green bounding guide on original frame copy
        frame_guide = frame.copy()
        cv2.rectangle(frame_guide, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_guide,
            "Center Target ROI (Place waste item here)",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        # Show windows
        cv2.imshow("CS231 - Waste Classification Dual-Screen Comparison", split_screen)
        cv2.imshow("Webcam Input Guide", cv2.resize(frame_guide, (480, 360)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("s"):
            snapshot_path = f"results/snapshots/snapshot_{snapshot_idx:03d}.jpg"
            cv2.imwrite(snapshot_path, roi_bgr)
            print(f"[+] Saved snapshot to: {snapshot_path}")
            snapshot_idx += 1

    if not use_synth:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_split_demo()

