# Hướng Dẫn Thu Thập Tập Dữ Liệu Rác Thực Tế Việt Nam (`vn_real_test`)

Tập dữ liệu này gồm **120 ảnh (20 ảnh / lớp)** do nhóm tự chụp bằng điện thoại hoặc webcam máy tính.
Mục tiêu là chứng minh với Hội đồng Giảng viên môn **CS231 (UIT)** về khả năng thích ứng miền (Domain Generalization) của mô hình trên các loại rác quen thuộc tại Việt Nam.

---

## 1. Danh sách 6 Lớp & Gợi ý Mẫu vật tại Ký túc xá / Phòng học

| Lớp (Folder) | Số lượng | Gợi ý vật thể rác Việt Nam thực tế |
| :--- | :---: | :--- |
| `cardboard/` | 20 ảnh | Hộp carton giao hàng Shopee / Lazada / Tiki, hộp bánh Danisa, vỏ hộp pizza. |
| `glass/` | 20 ảnh | Chai bia Sài Gòn, chai bia Tiger xanh/nâu, lọ dầu gió, hũ sữa chua nếp cẩm thủy tinh. |
| `metal/` | 20 ảnh | Vỏ lon Coca-Cola, lon Sting đỏ/vàng, lon RedBull lùn, lon bia 333, hộp cá mòi nắp giật. |
| `paper/` | 20 ảnh | Giấy thi A4, tài liệu photo cũ, tập vở sinh viên đã dùng, hóa đơn thanh toán / vé xe bus. |
| `plastic/` | 20 ảnh | Chai nước suối Aquafina, chai C2, trà xanh Không Độ, ly trà sữa nắp màng ép, hộp cơm xốp. |
| `biological/` | 20 ảnh | Vỏ chuối, vỏ cam quýt, cùi táo, bã trà / bã cà phê, rau củ thừa từ căng tin sinh viên. |

---

## 2. Quy chuẩn Chụp ảnh Chuẩn học thuật

1. **Vật thể đơn lẻ (Single-item):** Mỗi khung hình chỉ có 1 vật thể rác chính chiếm ít nhất 30% diện tích.
2. **Khung ảnh:** Chụp góc nhìn thẳng hoặc nghiêng 45 độ, vật thể nằm ở trung tâm khung hình.
3. **Phông nền (Background):**
   - Chụp trên mặt bàn học, bàn ăn, sàn gạch men hoặc cầm trên tay.
   - Khi demo trực tiếp tại hội đồng: Sử dụng 1 tờ giấy A3 màu trắng hoặc đen để lót sau vật thể, loại bỏ 100% nhiễu nền từ trang phục hoặc người ngồi sau.
4. **Định dạng:** Lưu dạng `.jpg` hoặc `.png`, đặt tên theo cú pháp `vn_{class}_{index:02d}.jpg` (Ví dụ: `vn_metal_01.jpg`, `vn_plastic_15.jpg`).

