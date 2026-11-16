"""Trích xuất đặc trưng thủ công (hand-crafted features) - Thành viên 1.

Quy trình xử lý 1 ảnh:
    Ảnh RGB 224x224
      B1. Tiền xử lý : chuyển sang HSV và ảnh xám, làm mờ Gaussian 3x3 để giảm nhiễu
      B2. Màu sắc    : histogram HSV 3 chiều  8 x 8 x 8         -> 512 số
      B3. Kết cấu    : histogram LBP uniform (P=24, R=3)        ->  26 số
      B4. Hình dạng  : HOG trên ảnh xám thu nhỏ 128x128         -> 1764 số
      B5. Nối 3 vector lại                                      -> 2302 số

Vector 2302 chiều này là đầu vào cho bước chuẩn hóa + PCA + SVM (xem train_ml.py).
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from src.utils import read_image_rgb  # noqa: F401 (re-exported for Member 1 scripts)


class WasteFeatureExtractor:
    def __init__(
        self,
        h_bins=8,
        s_bins=8,
        v_bins=8,
        lbp_points=24,
        lbp_radius=3,
        hog_size=128,
        hog_orientations=9,
        hog_pixels_per_cell=16,
        hog_cells_per_block=2,
    ):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        self.hog_size = hog_size
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block

    def get_params(self):
        """Trả về các tham số để lưu chung với model (lúc predict phải dùng đúng tham số lúc train)."""
        return {
            "h_bins": self.h_bins,
            "s_bins": self.s_bins,
            "v_bins": self.v_bins,
            "lbp_points": self.lbp_points,
            "lbp_radius": self.lbp_radius,
            "hog_size": self.hog_size,
            "hog_orientations": self.hog_orientations,
            "hog_pixels_per_cell": self.hog_pixels_per_cell,
            "hog_cells_per_block": self.hog_cells_per_block,
        }

    # ------------------------------------------------------------------
    # B1. Tiền xử lý
    # ------------------------------------------------------------------
    def preprocess(self, image_rgb):
        """Đưa ảnh về 224x224, tạo ảnh HSV và ảnh xám đã làm mờ nhẹ."""
        if image_rgb.shape[:2] != (224, 224):
            image_rgb = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        # Ảnh web/ảnh JPEG có nhiều nhiễu hạt, làm mờ nhẹ để LBP và HOG ổn định hơn
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return image_rgb, hsv, gray

    # ------------------------------------------------------------------
    # B2. Đặc trưng màu sắc: histogram HSV 3D
    # ------------------------------------------------------------------
    def color_histogram(self, hsv):
        """Đếm số pixel rơi vào từng ô (H, S, V). Chia cho tổng để ra tỉ lệ, không phụ thuộc kích thước ảnh.

        Lưu ý: trong OpenCV kênh H có giá trị 0..179, kênh S và V có giá trị 0..255.
        """
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            [self.h_bins, self.s_bins, self.v_bins],
            [0, 180, 0, 256, 0, 256],
        )
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)
        return hist.astype(np.float32)

    # ------------------------------------------------------------------
    # B3. Đặc trưng kết cấu: Local Binary Pattern
    # ------------------------------------------------------------------
    def lbp_image(self, gray):
        """Mỗi pixel so sánh với P điểm lân cận trên đường tròn bán kính R -> 1 mã LBP.

        method="uniform": chỉ giữ các mẫu có <= 2 lần chuyển 0/1 (cạnh, góc, điểm phẳng),
        các mẫu còn lại gộp chung vào 1 nhãn. Nên giá trị LBP nằm trong 0 .. P+1.
        """
        return local_binary_pattern(gray, P=self.lbp_points, R=self.lbp_radius, method="uniform")

    def lbp_histogram(self, lbp):
        """Đếm tần suất từng mã LBP trên toàn ảnh -> mô tả kết cấu (nhám, mịn, nhiều cạnh...)."""
        n_bins = self.lbp_points + 2  # P+1 mẫu uniform + 1 nhãn non-uniform
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist / (hist.sum() + 1e-7)
        return hist.astype(np.float32)

    # ------------------------------------------------------------------
    # B4. Đặc trưng hình dạng: Histogram of Oriented Gradients
    # ------------------------------------------------------------------
    def hog_features(self, gray, visualize=False):
        """Chia ảnh 128x128 thành ô 16x16 (8x8 ô), mỗi ô tính histogram 9 hướng gradient.

        Gom 2x2 ô thành 1 block rồi chuẩn hóa L2-Hys (giảm ảnh hưởng của độ sáng).
        Số block = 7 x 7, mỗi block 2*2*9 = 36 số -> 7*7*36 = 1764 số.
        Thu nhỏ ảnh về 128x128 để vector HOG không quá dài so với màu và LBP.
        """
        small = cv2.resize(gray, (self.hog_size, self.hog_size), interpolation=cv2.INTER_AREA)
        ppc = (self.hog_pixels_per_cell, self.hog_pixels_per_cell)
        cpb = (self.hog_cells_per_block, self.hog_cells_per_block)

        if visualize:
            feat, hog_img = hog(
                small, orientations=self.hog_orientations, pixels_per_cell=ppc,
                cells_per_block=cpb, block_norm="L2-Hys", visualize=True,
            )
            return feat.astype(np.float32), hog_img

        feat = hog(
            small, orientations=self.hog_orientations, pixels_per_cell=ppc,
            cells_per_block=cpb, block_norm="L2-Hys",
        )
        return feat.astype(np.float32), None

    # ------------------------------------------------------------------
    # B5. Ghép 3 nhóm đặc trưng
    # ------------------------------------------------------------------
    def extract(self, image_rgb):
        """Trả về vector đặc trưng 1 chiều (2302 số với tham số mặc định)."""
        _, hsv, gray = self.preprocess(image_rgb)
        f_color = self.color_histogram(hsv)
        f_lbp = self.lbp_histogram(self.lbp_image(gray))
        f_hog, _ = self.hog_features(gray)
        return np.concatenate([f_color, f_lbp, f_hog])

    def feature_groups(self):
        """Vị trí của từng nhóm đặc trưng trong vector (dùng cho thí nghiệm ablation)."""
        n_color = self.h_bins * self.s_bins * self.v_bins
        n_lbp = self.lbp_points + 2
        n_cells = self.hog_size // self.hog_pixels_per_cell
        n_blocks = n_cells - self.hog_cells_per_block + 1
        n_hog = n_blocks * n_blocks * self.hog_cells_per_block ** 2 * self.hog_orientations
        return {
            "HSV": slice(0, n_color),
            "LBP": slice(n_color, n_color + n_lbp),
            "HOG": slice(n_color + n_lbp, n_color + n_lbp + n_hog),
        }

    def extract_with_steps(self, image_rgb):
        """Giống extract() nhưng trả thêm ảnh trung gian của từng bước để trực quan hóa / demo."""
        image_rgb, hsv, gray = self.preprocess(image_rgb)
        f_color = self.color_histogram(hsv)
        lbp = self.lbp_image(gray)
        f_lbp = self.lbp_histogram(lbp)
        f_hog, hog_img = self.hog_features(gray, visualize=True)

        steps = {
            "rgb": image_rgb,
            "gray": gray,
            "lbp_image": lbp,
            "hog_image": hog_img,
            "f_color": f_color,
            "f_lbp": f_lbp,
            "f_hog": f_hog,
        }
        return np.concatenate([f_color, f_lbp, f_hog]), steps


def hog_overlay(image_rgb, hog_img):
    """Vẽ các đường HOG (màu vàng) đè lên ảnh gốc đã làm tối, để thấy model 'nhìn' biên dạng ở đâu."""
    hog_big = cv2.resize(hog_img.astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
    hog_big = hog_big / (hog_big.max() + 1e-7)
    hog_big = np.clip(hog_big * 2.0, 0, 1)  # tăng độ sáng các nét cho dễ nhìn

    base = (image_rgb.astype(np.float32) * 0.45)
    yellow = np.array([255, 230, 0], dtype=np.float32)
    out = base * (1 - hog_big[..., None]) + yellow * hog_big[..., None]
    return out.astype(np.uint8)
