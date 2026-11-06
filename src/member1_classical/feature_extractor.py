"""Feature Extraction for Classical CV Branch.

Combines three complementary visual descriptors:
1. HSV Color Histogram: Captures chromatic distribution (e.g., green biological waste, brown cardboard).
2. Local Binary Patterns (LBP): Captures micro-texture (e.g., rough paper/cardboard vs smooth reflective plastic/glass).
3. Histogram of Oriented Gradients (HOG): Captures geometric shape and structural contours (e.g., bottle silhouette, cans).
"""

from typing import Tuple, Optional
import cv2
import numpy as np

try:
    from skimage.feature import local_binary_pattern, hog
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


class WasteFeatureExtractor:
    """Extracts concatenated hand-crafted features from RGB waste images."""

    def __init__(
        self,
        h_bins: int = 16,
        s_bins: int = 8,
        v_bins: int = 8,
        lbp_points: int = 24,
        lbp_radius: int = 3,
        hog_orientations: int = 9,
        hog_pixels_per_cell: Tuple[int, int] = (16, 16),
        hog_cells_per_block: Tuple[int, int] = (2, 2),
    ):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block

    def extract_hsv_hist(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extracts 3D normalized HSV color histogram."""
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist(
            [image_hsv],
            [0, 1, 2],
            None,
            [self.h_bins, self.s_bins, self.v_bins],
            [0, 180, 0, 256, 0, 256],
        )
        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype(np.float32)

    def extract_lbp_hist(self, gray: np.ndarray) -> np.ndarray:
        """Extracts uniform Local Binary Pattern texture histogram."""
        if SKIMAGE_AVAILABLE:
            lbp = local_binary_pattern(
                gray, P=self.lbp_points, R=self.lbp_radius, method="uniform"
            )
            n_bins = int(self.lbp_points * (self.lbp_points - 1) + 3)
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            return hist.astype(np.float32)
        else:
            # Fast OpenCV/Numpy fallback for basic 8-neighbor LBP
            h, w = gray.shape
            lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
            center = gray[1:-1, 1:-1]
            for i, (dy, dx) in enumerate([
                (-1, -1), (-1, 0), (-1, 1),
                (0, 1), (1, 1), (1, 0),
                (1, -1), (0, -1)
            ]):
                lbp |= ((gray[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx] >= center) << i).astype(np.uint8)
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
            return hist.astype(np.float32)

    def extract_hog(
        self, gray: np.ndarray, return_viz: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extracts HOG descriptor and optional visualization image."""
        if SKIMAGE_AVAILABLE:
            if return_viz:
                features, hog_image = hog(
                    gray,
                    orientations=self.hog_orientations,
                    pixels_per_cell=self.hog_pixels_per_cell,
                    cells_per_block=self.hog_cells_per_block,
                    visualize=True,
                    feature_vector=True,
                )
                # Rescale HOG image to 0-255 uint8 RGB
                hog_viz_norm = (hog_image / (hog_image.max() + 1e-7) * 255).astype(np.uint8)
                hog_viz_rgb = cv2.cvtColor(hog_viz_norm, cv2.COLOR_GRAY2RGB)
                return features.astype(np.float32), hog_viz_rgb
            else:
                features = hog(
                    gray,
                    orientations=self.hog_orientations,
                    pixels_per_cell=self.hog_pixels_per_cell,
                    cells_per_block=self.hog_cells_per_block,
                    visualize=False,
                    feature_vector=True,
                )
                return features.astype(np.float32), None
        else:
            # OpenCV HOG fallback
            win_size = (224, 224)
            block_size = (
                self.hog_pixels_per_cell[0] * self.hog_cells_per_block[0],
                self.hog_pixels_per_cell[1] * self.hog_cells_per_block[1],
            )
            block_stride = self.hog_pixels_per_cell
            cell_size = self.hog_pixels_per_cell
            nbins = self.hog_orientations
            hog_cv = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
            features = hog_cv.compute(gray).flatten()
            
            # Simple gradient magnitude visualization
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            mag_rgb = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
            return features.astype(np.float32), mag_rgb

    def extract_features(
        self, image_rgb: np.ndarray, return_viz: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extracts concatenated (HSV + LBP + HOG) feature vector.

        Returns:
            feature_vector: 1D np.ndarray float32.
            viz_rgb: (224, 224, 3) uint8 image representing the visual explanation.
        """
        if image_rgb.shape[:2] != (224, 224):
            image_rgb = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Color (HSV)
        hsv_feat = self.extract_hsv_hist(image_rgb)

        # 2. Texture (LBP)
        lbp_feat = self.extract_lbp_hist(gray)

        # 3. Shape (HOG)
        hog_feat, hog_viz = self.extract_hog(gray, return_viz=return_viz)

        # Concatenate into single feature representation
        feature_vector = np.concatenate([hsv_feat, lbp_feat, hog_feat], axis=0)

        # Create blended visual explanation if requested
        if return_viz and hog_viz is not None:
            # Blend 50% original image + 50% HOG edge map
            viz_blended = cv2.addWeighted(image_rgb, 0.5, hog_viz, 0.5, 0)
            return feature_vector, viz_blended

        return feature_vector, None

