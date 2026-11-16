"""Streamlit Dashboard for Waste Classification.

Comparative web application for CS231 project presentation:
  - Live Webcam & Image Upload testing
  - Side-by-side Model Explainability (HOG vs Grad-CAM)
  - Real Benchmark Results, Per-Class Comparison & Disagreement Cases
  - Vietnamese municipal waste sorting rules
"""

import csv
import os
import sys

import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.base_predictor import CLASS_NAMES
from src.member1_classical.predictor_ml import ClassicalPredictor
from src.member2_deeplearning.predictor_dl import DeepLearningPredictor
from src.utils import resolve_path


st.set_page_config(
    page_title="CS231 - Waste Classification Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0px; }
    .sub-title { font-size: 1.1rem; color: #555; margin-bottom: 20px; }
    .card-classical { background-color: #f0f7ff; padding: 15px; border-radius: 10px; border-left: 5px solid #2196F3; }
    .card-dl { background-color: #f1fbf4; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50; }
    .bin-tag { font-weight: bold; padding: 4px 10px; border-radius: 5px; display: inline-block; }
    .bin-recycle { background-color: #BBDEFB; color: #0D47A1; }
    .bin-organic { background-color: #C8E6C9; color: #1B5E20; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Initializes and caches both predictors."""
    svm_abs = resolve_path("weights/classical_svm_pipeline.pkl")
    p_ml = ClassicalPredictor(weight_path=svm_abs) if os.path.exists(svm_abs) else None
    p_dl = DeepLearningPredictor(weight_path="weights/mobilenetv4_best.pth", fallback_mode=True)
    return p_ml, p_dl


def load_csv_rows(path: str):
    abs_path = resolve_path(path)
    if not os.path.exists(abs_path):
        return None
    with open(abs_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


p_ml, p_dl = load_models()

st.sidebar.title("CS231 - ĐỒ ÁN (UIT)")
st.sidebar.markdown("**Môn học:** Nhập môn Thị giác máy tính")
st.sidebar.markdown("**Đề tài:** Phân loại rác sinh hoạt đơn lẻ tại nguồn")
st.sidebar.markdown("---")
st.sidebar.markdown("### Thành viên nhóm:")
st.sidebar.markdown("- **TV1:** Classical CV (HSV + LBP + HOG + SVM)")
st.sidebar.markdown("- **TV2:** Deep Learning (CNN + Grad-CAM)")
st.sidebar.markdown("---")
input_mode = st.sidebar.radio("Nguồn ảnh đầu vào:", ["Tải ảnh lên (Upload)", "Chụp từ Webcam"])

st.markdown('<div class="main-title">♻️ Waste Classification Comparative Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hệ thống Đối sánh Trực quan giữa Thị giác Cổ điển (Hand-crafted) và Mạng Học sâu (CNN)</div>', unsafe_allow_html=True)

tabs = st.tabs(["🔍 Thử nghiệm & Đối sánh", "📊 Phân phối Xác suất", "📈 Kết quả Benchmark", "📋 Quy chuẩn Phân loại"])

input_image_rgb = None
res_ml = None
res_dl = None

with tabs[0]:
    col_input, col_models = st.columns([1, 2])

    with col_input:
        st.subheader("1. Chọn mẫu rác thử nghiệm")
        if input_mode == "Tải ảnh lên (Upload)":
            uploaded_file = st.file_uploader("Chọn ảnh rác (.jpg, .png)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                input_image_rgb = np.array(Image.open(uploaded_file).convert("RGB"))
        else:
            cam_img = st.camera_input("Chụp ảnh từ webcam (Đưa vật thể vào giữa)")
            if cam_img is not None:
                input_image_rgb = np.array(Image.open(cam_img).convert("RGB"))

        if input_image_rgb is not None:
            h, w = input_image_rgb.shape[:2]
            min_dim = min(h, w)
            sy, sx = (h - min_dim) // 2, (w - min_dim) // 2
            roi_rgb = input_image_rgb[sy:sy + min_dim, sx:sx + min_dim]
            roi_224 = cv2.resize(roi_rgb, (224, 224), interpolation=cv2.INTER_AREA)
            st.image(roi_224, caption="Vùng ROI chuẩn hóa (224x224)", use_container_width=True)
        else:
            st.info("Vui lòng tải ảnh lên hoặc bật webcam để bắt đầu đối sánh.")

    with col_models:
        st.subheader("2. Kết quả đối sánh song song")
        if input_image_rgb is not None:
            res_ml = p_ml.predict(roi_224) if p_ml is not None else None
            res_dl = p_dl.predict(roi_224)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="card-classical">', unsafe_allow_html=True)
                st.markdown("#### 🔹 Thành viên 1: Classical ML")
                if res_ml is not None:
                    st.markdown(f"**Dự đoán:** `{res_ml['class_name'].upper()}`")
                    st.progress(float(res_ml["confidence"]))
                    st.write(f"Độ tự tin: **{res_ml['confidence']*100:.1f}%** | Trễ: **{res_ml['latency_ms']:.1f} ms**")
                    bin_class = "bin-organic" if "Hữu Cơ" in res_ml["action_bin"] else "bin-recycle"
                    st.markdown(f'<span class="bin-tag {bin_class}">{res_ml["action_bin"]}</span>', unsafe_allow_html=True)
                    st.write("")
                    st.image(res_ml["visual_explanation"], caption="Giải thích: Phủ đặc trưng HOG", use_container_width=True)
                else:
                    st.warning("Chưa có weights/classical_svm_pipeline.pkl. Hãy chạy train_ml.py.")
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="card-dl">', unsafe_allow_html=True)
                st.markdown(f"#### 🟢 Thành viên 2: Deep Learning ({p_dl.model_name})")
                st.markdown(f"**Dự đoán:** `{res_dl['class_name'].upper()}`")
                st.progress(float(res_dl["confidence"]))
                st.write(f"Độ tự tin: **{res_dl['confidence']*100:.1f}%** | Trễ: **{res_dl['latency_ms']:.1f} ms**")
                bin_class = "bin-organic" if "Hữu Cơ" in res_dl["action_bin"] else "bin-recycle"
                st.markdown(f'<span class="bin-tag {bin_class}">{res_dl["action_bin"]}</span>', unsafe_allow_html=True)
                st.write("")
                st.image(res_dl["visual_explanation"], caption="Giải thích: Bản đồ nhiệt Grad-CAM", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.write("Đang chờ dữ liệu đầu vào...")

with tabs[1]:
    st.subheader("Phân phối xác suất 6 lớp (Probability Distribution)")
    if res_dl is not None:
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("**Thành viên 1 (SVM)**")
            if res_ml is not None:
                st.bar_chart(dict(zip(CLASS_NAMES, res_ml["probabilities"])))
        with p2:
            st.markdown(f"**Thành viên 2 ({p_dl.model_name})**")
            st.bar_chart(dict(zip(CLASS_NAMES, res_dl["probabilities"])))
    else:
        st.info("Hãy chọn ảnh ở Tab 1 để xem biểu đồ xác suất.")

with tabs[2]:
    st.subheader("Bảng Đối sánh Định lượng (Benchmark Matrix)")
    bench_rows = load_csv_rows("results/benchmark_comparison.csv")
    if bench_rows:
        st.table(bench_rows)
    else:
        st.info("Chưa có `results/benchmark_comparison.csv`. Chạy `python src/evaluation/benchmark.py` để tạo số liệu thực tế.")

    per_class_rows = load_csv_rows("results/benchmark_per_class.csv")
    if per_class_rows:
        st.subheader("So sánh chi tiết từng lớp (Per-class)")
        st.table(per_class_rows)

    col_cm1, col_cm2 = st.columns(2)
    cm1 = resolve_path("results/cm_classical_standard_test_set.png")
    cm2 = resolve_path("results/cm_dl_standard_test_set.png")
    if os.path.exists(cm1):
        col_cm1.image(cm1, caption="Confusion Matrix - TV1", use_container_width=True)
    if os.path.exists(cm2):
        col_cm2.image(cm2, caption="Confusion Matrix - TV2", use_container_width=True)

    ablation_img = resolve_path("results/member1/ablation.png")
    if os.path.exists(ablation_img):
        st.image(ablation_img, caption="TV1 Ablation Study (HSV / LBP / HOG)", use_container_width=True)

    case_img = resolve_path("results/case_tv1_wrong_tv2_right_standard_test_set.png")
    if os.path.exists(case_img):
        st.image(case_img, caption="Case Study: TV1 sai nhưng TV2 đúng", use_container_width=True)

with tabs[3]:
    st.subheader("Quy định Phân loại Rác tại nguồn (Luật BVMT Việt Nam)")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ### ⚪/🔵 Nhóm Rác Tái Chế
        - **Bao bì giấy & Carton (`paper`, `cardboard`):** Sách báo cũ, thùng carton dẹp gọn, giấy văn phòng.
        - **Kim loại (`metal`):** Vỏ lon nước ngọt, hộp thiếc, nắp chai nhôm.
        - **Nhựa tái chế (`plastic`):** Chai nước suối PET, chai dầu gội HDPE.
        - **Thủy tinh (`glass`):** Chai lọ thủy tinh nguyên vẹn.
        """)
    with col_b:
        st.markdown("""
        ### 🟢 Nhóm Rác Hữu Cơ & Khác
        - **Rác hữu cơ (`biological`):** Thức ăn thừa, vỏ rau củ quả, bã trà/cà phê, lá cây.
        - **Xử lý:** Chuyển sang nhà máy ủ phân compost hoặc xử lý sinh học.
        """)
