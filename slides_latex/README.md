# Hướng Dẫn Biên Dịch & Thuyết Trình Slide LaTeX Beamer (CS231 - UIT)

File mã nguồn chính: `slides_latex/main.tex`

---

## 1. Cách Biên Dịch Thành File PDF

### Cách 1: Sử dụng Overleaf (Khuyên dùng, không lo cài font/gói)
1. Truy cập [Overleaf.com](https://www.overleaf.com) $\rightarrow$ Tạo New Project $\rightarrow$ Upload Project.
2. Tải toàn bộ thư mục `slides_latex/` lên hoặc copy nội dung file `main.tex`.
3. Vào mục **Menu** (góc trên bên trái) $\rightarrow$ Chọn **Compiler:** `XeLaTeX` (hoặc `pdfLaTeX`).
4. Bấm **Recompile** để xuất ra file `main.pdf` chuẩn trình chiếu tỉ lệ 16:9.

### Cách 2: Biên dịch trực tiếp trên máy tính cá nhân
Nếu máy tính đã cài MiKTeX hoặc TeX Live:
```powershell
cd slides_latex
xelatex main.tex
```

---

## 2. Bố Cục Chiến Lược 13 Slides (Phân Bổ Thời Gian 10–12 Phút)

| Slide | Tên Nội Dung | Người Trình Bày | Thời lượng | Trọng tâm cần nhấn mạnh |
| :---: | :--- | :---: | :---: | :--- |
| **1** | Tiêu đề đồ án & Thành viên | Chung | 30s | Giới thiệu ngắn gọn tên đề tài, MSSV của 2 bạn. |
| **2** | Đặt vấn đề & Input - Output | TV1 | 1 phút | Nêu rõ giả định **Single-item** và ROI $300 \times 300$, quy định thùng rác VN. |
| **3** | Sơ đồ Kiến trúc 2 Nhánh | TV1 + TV2 | 1 phút | Chỉ vào sơ đồ TikZ phân biệt rạch ròi 2 luồng độc lập. |
| **4** | Chuẩn bị Dữ liệu & Tập VN | TV1 | 1 phút | Giải thích seed=42 để công bằng và tập 120 ảnh rác thực tế tại VN. |
| **5** | Trích xuất 3 Đặc trưng Cổ điển | TV1 | 1.5 phút | Giải thích vai trò của HSV (màu), LBP (nhám mịn), HOG (biên dạng lon/chai). |
| **6** | PCA Nén chiều & SVM/XGBoost | TV1 | 1 phút | Lý do bắt buộc dùng PCA (nén từ 2300 xuống ~80 chiều chống Overfitting). |
| **7** | Mạng Tích chập CNN | TV2 | 1.5 phút | Kiến trúc MobileNetV4 nhẹ cho nhúng, Augmentation chống lóa sáng. |
| **8** | Explainable AI với Grad-CAM | TV2 | 1 phút | Công thức toán Grad-CAM và chứng minh CNN nhìn vào nắp chai/khoen lon. |
| **9** | **Kết quả Đối sánh Định lượng** | TV2 | **2 phút** | **Slide quan trọng nhất:** So sánh bảng số liệu, CPU Latency vs Accuracy. |
| **10**| Ma trận Nhầm lẫn (Confusion Matrix)| TV1 + TV2 | 1 phút | Phân tích ca nhầm lẫn giữa Nhựa (Plastic) và Thủy tinh (Glass). |
| **11**| **Case Study: "A sai nhưng B đúng"**| TV1 + TV2 | **1.5 phút** | Đúng "gu" giảng viên CS231 UIT: Giải thích tại sao CNN thắng ML cổ điển. |
| **12**| Demo Thực tế | Chung | 1.5 phút | Mở `python app/webcam_split_demo.py` đưa vật thể vào trước webcam. |
| **13**| Kết luận & Hướng mở rộng | Chung | 30s | Chốt hạ khi nào dùng Cổ điển, khi nào dùng Deep Learning. |

