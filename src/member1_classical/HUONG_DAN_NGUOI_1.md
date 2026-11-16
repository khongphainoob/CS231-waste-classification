# Hướng dẫn phần Thành viên 1: Thị giác cổ điển (HSV + LBP + HOG + SVM)

Tài liệu này dùng để: (1) chạy lại toàn bộ phần TV1, (2) hiểu từng bước để trình bày, (3) chuẩn bị câu hỏi vấn đáp.
Mọi con số ở đây là **số đo thật** trên máy (CPU, Python 3.10), dữ liệu Kaggle chia với seed 42.

---

## 1. Phạm vi công việc của TV1

| File | Vai trò |
| --- | --- |
| `feature_extractor.py` | Trích đặc trưng HSV + LBP + HOG, vẽ HOG đè lên ảnh |
| `train_ml.py` | Trích đặc trưng toàn bộ dữ liệu → chọn C, gamma trên val → train → đánh giá test → lưu model |
| `predictor_ml.py` | `ClassicalPredictor` (kế thừa `BasePredictor`), dùng chung cho webcam, Streamlit, benchmark |
| `ablation.py` | Thí nghiệm: bỏ bớt từng nhóm đặc trưng thì accuracy thay đổi thế nào |
| `visualize_pipeline.py` | Vẽ 8 ô minh họa từng bước trên 1 ảnh (dùng cho slide và khi thầy hỏi) |
| `configs/config.yaml` (mục `classical`) | Toàn bộ tham số của TV1 |

---

## 2. Quy trình (nói theo đúng thứ tự này khi trình bày)

```
Ảnh RGB 224x224
   │
   ├─ B1. Tiền xử lý: chuyển HSV; chuyển xám + Gaussian 3x3 (giảm nhiễu JPEG)
   │
   ├─ B2. MÀU     : Histogram HSV 3D, 8x8x8 ô            ->  512 số
   ├─ B3. KẾT CẤU : Histogram LBP uniform P=24, R=3       ->   26 số
   ├─ B4. HÌNH DẠNG: HOG trên ảnh xám 128x128             -> 1764 số
   │
   └─ B5. Nối lại: x = [HSV | LBP | HOG]                  -> 2302 số
            │
            ├─ StandardScaler  (mỗi chiều về mean 0, std 1)
            ├─ PCA giữ 95% phương sai  (2302 -> 693 chiều)
            └─ SVM kernel RBF (C=3, gamma=0.0005)  -> xác suất 6 lớp -> thùng rác
```

Lý do có 3 nhóm đặc trưng: mỗi nhóm "nhìn" một khía cạnh khác nhau của vật liệu:
- **Màu (HSV)**: rác hữu cơ có màu xanh/vàng/nâu, carton màu nâu, giấy trắng.
- **Kết cấu (LBP)**: giấy/carton nhám, nhựa/thủy tinh láng, giấy bạc nhăn.
- **Hình dạng (HOG)**: dáng chai, miệng lon tròn, cạnh thẳng của hộp.

---

## 3. Cách chạy (từ thư mục gốc đồ án)

```bash
pip install -r requirements.txt
```

```bash
python data/download_and_split.py --raw_dir data/raw/garbage_classification
```

```bash
python src/member1_classical/train_ml.py
```

```bash
python src/member1_classical/ablation.py
```

```bash
python src/member1_classical/visualize_pipeline.py --cls glass --show
```

```bash
python src/member1_classical/train_ml.py --model xgboost
```

- Lần đầu train mất ~3 phút trên CPU (2,5 phút trích đặc trưng). Đặc trưng được cache ở `results/member1/cache/`, các lần sau chỉ mất ~1 phút. Thêm `--no_cache` để trích lại.
- Dataset tải ở Kaggle: `mostafaabla/garbage-classification` (12 lớp, gộp lại còn 6 lớp).
- Kết quả nằm trong `results/member1/`: `report_svm.txt`, `svm_tuning.csv`, `confusion_matrix_svm.png`, `wrong_examples_svm.png`, `ablation.csv`, `ablation.png`, `pipeline_*.png`.
- Notebook `notebooks/02_Member1_Classical_HOG_LBP_ML.ipynb` chạy đúng các bước trên theo thứ tự.

---

## 4. Kết quả thật

### 4.1. Dữ liệu
Train 3.360 / Val 720 / Test 689 ảnh (mỗi lớp 560 / 120 / 120; riêng **metal chỉ có 769 ảnh gốc** nên test còn 89 ảnh).

### 4.2. Chọn siêu tham số SVM trên tập validation (`svm_tuning.csv`)

| C \ gamma | scale | 0.0005 | 0.002 |
| --- | --- | --- | --- |
| 1  | 71.8% | 72.2% | 58.6% |
| 3  | 72.1% | **72.6%** | 60.0% |
| 10 | 72.4% | 72.6% | 60.0% |
| 30 | 72.4% | 72.6% | 60.0% |

→ Chọn **C = 3, gamma = 0.0005**. Gamma lớn (0.002) giảm mạnh vì mỗi điểm chỉ ảnh hưởng vùng rất hẹp → overfit. C từ 3 trở lên gần như không đổi.

### 4.3. Kết quả trên tập test (chỉ chạy 1 lần, sau khi đã chốt tham số)

| Model | Test accuracy | Macro F1 | Độ trễ / ảnh | File model |
| --- | --- | --- | --- | --- |
| **SVM RBF** | **76.5%** | **76.1%** | 31.5 ms (~32 FPS) | 23.9 MB |
| XGBoost (300 cây, depth 5) | 66.8% | 66.0% | 56.8 ms | 10.3 MB |

Từng lớp (SVM): cardboard F1 0.84 · paper 0.85 · biological 0.81 · glass 0.70 · plastic 0.69 · metal 0.67.

### 4.4. Ablation (`ablation.py`, cùng C, gamma)

| Đặc trưng | Số chiều | Test acc | Nhận xét |
| --- | --- | --- | --- |
| Chỉ HSV | 512 | 65.2% | Mạnh nhất với cardboard (79%), yếu với plastic (48%) |
| Chỉ LBP | 26 | 45.7% | Yếu khi đứng một mình, nhưng vẫn gấp ~3 lần đoán bừa (16.7%) |
| Chỉ HOG | 1764 | 65.2% | Tốt cho plastic (62%), biological (78%) |
| HSV + LBP | 538 | 68.8% | |
| HSV + HOG | 2276 | 75.3% | Màu và hình dạng **bổ sung cho nhau**: +10 điểm |
| LBP + HOG | 1790 | 66.9% | |
| **HSV + LBP + HOG** | **2302** | **76.5%** | LBP thêm +1.2 điểm (cardboard, glass) |

### 4.5. Lỗi thường gặp (ma trận nhầm lẫn)
- `plastic → glass`: 17 ảnh. Chai nhựa trong suốt và chai thủy tinh trong suốt có màu và dáng gần giống nhau.
- `metal → plastic`: 12 ảnh. Kim loại bóng phản chiếu màu xung quanh.
- `glass → biological`: 11 ảnh. Ví dụ chai thủy tinh xanh đặt trên **bãi cỏ** (xem `wrong_examples_svm.png`): histogram màu tính trên **cả ảnh** nên nền xanh lá kéo về lớp hữu cơ.
- Giấy bạc vo nhăn bị đoán là `paper`: LBP thấy kết cấu nhăn giống giấy.

---

## 5. Giải thích từng bước (nói gì khi trình bày)

**B1. Tiền xử lý.** Ảnh được đưa về 224x224 cho giống nhánh CNN. Ảnh xám được làm mờ Gaussian 3x3 vì ảnh web có nhiễu nén JPEG; LBP so sánh từng pixel nên rất nhạy với nhiễu.

**B2. Histogram HSV.** Dùng HSV thay vì RGB vì HSV tách *màu* (H), *độ đậm* (S) và *độ sáng* (V). Khi ánh sáng thay đổi chủ yếu chỉ V thay đổi, H gần như giữ nguyên. Mỗi kênh chia 8 khoảng → 8x8x8 = 512 ô, đếm số pixel rơi vào mỗi ô rồi chia tổng để ra tỉ lệ. (OpenCV: H trong 0..179, S và V trong 0..255.)

**B3. LBP.** Với mỗi pixel, lấy 24 điểm trên đường tròn bán kính 3: điểm nào ≥ tâm ghi 1, ngược lại ghi 0 → chuỗi 24 bit. Mẫu "uniform" là chuỗi có tối đa 2 lần đổi 0↔1 (tương ứng cạnh, góc, điểm phẳng); với mẫu uniform chỉ cần đếm số bit 1 (0..24) → 25 giá trị, bất biến khi xoay. Mọi mẫu không uniform gộp vào 1 nhãn → **26 ô**. Histogram 26 ô mô tả kết cấu bề mặt.
Mẹo khi chỉ vào ảnh LBP: nền trắng phẳng có mã 24 (tất cả lân cận ≥ tâm), nên nền có một màu đồng nhất.

**B4. HOG.** Tính gradient Gx, Gy → độ lớn và hướng (0–180°). Chia ảnh 128x128 thành ô 16x16 (8x8 ô), mỗi ô lập histogram 9 hướng (mỗi khoảng 20°), có trọng số theo độ lớn gradient. Gom 2x2 ô thành 1 block, trượt 1 ô → 7x7 block, mỗi block 36 số, chuẩn hóa L2-Hys để giảm ảnh hưởng của độ sáng/độ tương phản → 7x7x36 = **1764 số**.
Lý do thu nhỏ về 128: nếu tính trên 224 thì HOG có 6084 chiều, lấn át hẳn màu và LBP.

**B5. Scaler + PCA + SVM.**
- *StandardScaler*: các nhóm có thang đo khác nhau (histogram màu ~0.002, HOG tới ~0.3). SVM RBF dùng khoảng cách nên phải đưa về cùng thang đo, nếu không chiều có giá trị lớn sẽ quyết định tất cả.
- *PCA 95%*: giảm 2302 → 693 chiều, bỏ các hướng gần như không biến thiên (nhiễu), SVM chạy nhanh hơn. Scaler và PCA chỉ được **fit trên tập train** (nằm trong `Pipeline`), nên không bị lộ thông tin val/test.
- *SVM RBF*: K(x, z) = exp(−γ‖x − z‖²). C lớn → phạt lỗi nặng hơn (biên hẹp hơn); γ lớn → mỗi điểm chỉ ảnh hưởng vùng nhỏ (dễ overfit). 6 lớp được xử lý theo kiểu one-vs-one (15 bộ phân loại nhị phân). Xác suất có được nhờ Platt scaling (`probability=True`).

---

## 6. Kịch bản demo cho phần TV1 (khoảng 2–3 phút)

1. Mở `results/member1/pipeline_metal_0017.png` (hoặc chạy `visualize_pipeline.py --cls metal --show`), đi lần lượt 8 ô: ảnh gốc → ảnh xám → ảnh LBP → HOG → 3 histogram → xác suất.
2. Mở `ablation.png`: "mỗi đặc trưng một mình khoảng 65%, ghép màu + hình dạng lên 75%, thêm kết cấu lên 76,5%".
3. Mở `confusion_matrix_svm.png` + `wrong_examples_svm.png`: chỉ ca chai xanh trên cỏ → giải thích điểm yếu của histogram màu toàn ảnh.
4. Chạy webcam `python app/webcam_split_demo.py`: nửa trái là TV1, các nét vàng là HOG. **Đặt vật trên nền giấy A3 trắng**, vì dữ liệu train phần lớn là ảnh nền trắng.

---

## 7. Câu hỏi thầy hay hỏi và cách trả lời

**Tại sao dùng HSV mà không dùng RGB?**
RGB trộn màu và độ sáng chung trong cả 3 kênh; đổi ánh sáng thì cả 3 kênh đều đổi. HSV tách độ sáng ra kênh V nên màu (H) ổn định hơn khi ánh sáng thay đổi.

**Vì sao HSV 8x8x8 mà không phải nhiều ô hơn?**
16x8x8 sẽ ra 1024 chiều, nhiều ô trống, dễ nhạy nhiễu. 8 khoảng mỗi kênh đủ phân biệt các màu chính và giữ vector ngắn.

**LBP uniform là gì, sao ra 26 chiều?**
Mẫu có ≤ 2 lần chuyển 0↔1 trên vòng tròn. Với P = 24 có 25 mẫu uniform (đếm số bit 1 từ 0 đến 24) cộng 1 nhãn cho mọi mẫu còn lại → P + 2 = 26.

**Tại sao P = 24, R = 3?**
Bán kính 3 bắt được kết cấu thô hơn (sợi giấy, gợn carton) so với R = 1; P = 24 lấy đủ điểm trên vòng tròn đó. Đổi lại LBP là bước chậm nhất (~22 ms/ảnh). Hướng cải tiến: LBP nhiều tỉ lệ hoặc chia lưới để giữ thông tin vị trí.

**LBP chỉ có 45,7%, sao vẫn giữ?**
Ablation cho thấy thêm LBP vào HSV + HOG vẫn tăng 1,2 điểm (75,3 → 76,5%), rõ nhất ở cardboard và glass. Chi phí chỉ 26 chiều. Nhưng em nói thẳng là đóng góp nhỏ, và đây là một kết luận của thí nghiệm.

**HOG tính thế nào, sao ra 1764?**
128/16 = 8 ô mỗi chiều; block 2x2 ô trượt 1 ô → 7 block mỗi chiều → 49 block × (2×2 ô × 9 hướng = 36) = 1764.

**Vì sao chuẩn hóa block L2-Hys?**
Chia cho chuẩn L2 để không phụ thuộc độ tương phản, cắt các giá trị > 0.2 rồi chuẩn hóa lại để một cạnh quá mạnh không lấn át cả block.

**Vì sao cần StandardScaler trước PCA và SVM?**
PCA tìm hướng có phương sai lớn, SVM RBF dựa trên khoảng cách. Nếu không chuẩn hóa thì chiều có thang đo lớn (HOG) sẽ quyết định tất cả, histogram màu gần như bị bỏ qua.

**PCA giữ 95% là bao nhiêu chiều, vì sao chọn 95%?**
2302 → 693 chiều. 95% là mức phổ biến: bỏ 5% phương sai nhỏ (thường là nhiễu) mà vẫn giữ gần hết thông tin. 10 thành phần đầu chỉ giữ ~23% phương sai, chứng tỏ đặc trưng thủ công khá "trải đều", không nén mạnh được.

**Chọn C và gamma thế nào? Có dùng tập test để chọn không?**
Không. Thử 12 cặp (C, gamma), train trên train, chấm trên **val**, chọn cặp tốt nhất (C = 3, γ = 0.0005). Tập test chỉ dùng **1 lần** ở cuối. Bảng nằm trong `svm_tuning.csv`.

**Sao val (72,6%) thấp hơn test (76,5%)?**
Hai tập khác ảnh, mỗi tập ~700 ảnh nên sai số chuẩn khoảng ±1,6%; chênh 4 điểm vẫn là dao động bình thường giữa 2 mẫu ngẫu nhiên. Không có thông tin test nào được dùng khi chọn tham số.

**Sao SVM tốt hơn XGBoost (76,5% vs 66,8%)?**
Sau PCA, mỗi chiều là tổ hợp tuyến tính của rất nhiều đặc trưng gốc, thông tin rải đều trên 693 chiều. Cây quyết định chỉ cắt theo từng trục một nên khó khai thác; SVM RBF làm việc trên khoảng cách của toàn bộ vector nên hợp hơn với dữ liệu nhiều chiều, ít mẫu (3.360 ảnh).

**Bao nhiêu support vector? Có ý nghĩa gì?**
3.103 / 3.360 mẫu train (~92%) là support vector, nghĩa là các lớp **chồng lấn nhiều** trong không gian đặc trưng thủ công. Đây là giới hạn của đặc trưng cổ điển, cũng là lý do model nặng 23,9 MB (lưu 3.103 vector × 693 chiều + ma trận PCA 693 × 2302).

**Xác suất (confidence) lấy từ đâu?**
SVM không ra xác suất trực tiếp. `probability=True` dùng Platt scaling: học thêm hàm sigmoid trên điểm quyết định (bằng cross-validation nội bộ) để chuyển thành xác suất.

**Phần "giải thích" của TV1 là gì, có giống Grad-CAM không?**
Không giống. Các nét vàng là HOG: hướng và độ mạnh của gradient trong từng ô 16x16, tức là **đầu vào** mà model nhìn thấy. Nó không cho biết vùng nào quan trọng với quyết định như Grad-CAM. Để biết nhóm đặc trưng nào quan trọng, em dùng thí nghiệm ablation.

**Điểm yếu lớn nhất của phương pháp?**
(1) Histogram màu tính trên cả ảnh nên bị nền ảnh hưởng (chai xanh trên cỏ → hữu cơ). (2) Vật trong suốt/bóng lấy màu của nền. (3) Không có thông tin "đây là vật gì", chỉ có màu, kết cấu, cạnh. Khắc phục: tách nền (GrabCut / ngưỡng Otsu) rồi chỉ tính đặc trưng trong vùng vật thể; demo trên nền A3 trắng.

**Vì sao demo phải dùng nền trắng?**
Phần lớn ảnh train là ảnh sản phẩm nền trắng; đổi sang nền khác là *domain shift*. Đặc trưng màu toàn ảnh nhạy với nền nhất.

**Tốc độ bao nhiêu, chậm ở đâu?**
31,5 ms/ảnh (~32 FPS) trên CPU: LBP ~22 ms, HOG ~2 ms (~6 ms khi vẽ hình), HSV ~0,1 ms, Scaler + PCA + SVM ~2,6 ms.

**Code có dùng thư viện gì, tự viết phần nào?**
`cv2.calcHist` cho HSV, `skimage.feature.local_binary_pattern` và `skimage.feature.hog`, scikit-learn cho Scaler/PCA/SVM. Phần tự làm: thiết kế vector (số bin, kích thước ảnh HOG để cân bằng số chiều), quy trình chọn tham số trên val, cache đặc trưng, thí nghiệm ablation, phân tích lỗi, trực quan hóa.

**Gặp khó khăn kỹ thuật gì?**
- `cv2.imread` trên Windows trả về `None` với đường dẫn có tiếng Việt → đọc bằng `np.fromfile` + `cv2.imdecode` (hàm `read_image_rgb`).
- Lúc đầu LBP tính số bin sai (555 thay vì 26) → sửa theo đúng định nghĩa uniform P + 2.
- Lúc predict LBP bị tính 2 lần → sửa lại, độ trễ giảm từ ~53 ms xuống 31,5 ms.
- Model lưu kèm tham số trích đặc trưng để lúc predict chắc chắn giống lúc train.

---

## 8. Việc còn phải làm

- [ ] Chụp **120 ảnh rác thực tế** (20 ảnh/lớp) vào `data/vn_real_test/<lớp>/` theo `data/vn_real_test/README.md`, rồi chạy `python src/evaluation/benchmark.py` để có số liệu "VN Real" (slide đang ghi "chưa có").
- [ ] Slide 10 ("Phương pháp 1 sai nhưng 2 đúng", ví dụ chai Lavie 96,4%) là ví dụ giả định: cần thay bằng 1 ca thật sau khi TV2 train xong CNN.
- [ ] Slide 2 (sơ đồ kiến trúc TikZ) đang lỗi biên dịch từ bản gốc, cần sửa trước khi xuất PDF.
