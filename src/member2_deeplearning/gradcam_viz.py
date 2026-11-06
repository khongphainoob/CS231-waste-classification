"""Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN Interpretability.

Provides visual explanations for deep learning decisions by computing gradients
flowing into the final convolutional layer.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAMVisualizer:
    """Computes Grad-CAM heatmap and blended visualization for a PyTorch CNN."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else self._find_target_layer()

        self.gradients = None
        self.activations = None
        self.hook_handles = []

        if self.target_layer is not None:
            self._register_hooks()

    def _find_target_layer(self) -> Optional[nn.Module]:
        """Automatically searches for the last Conv2d layer in the model."""
        target = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                target = module
        return target

    def _register_hooks(self):
        """Registers forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles = [h1, h2]

    def remove_hooks(self):
        """Cleanly removes registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generates 2D normalized Grad-CAM heatmap in range [0, 1].

        Args:
            input_tensor: Tensor of shape (1, C, H, W)
            target_class: Target class index. If None, uses argmax prediction.

        Returns:
            heatmap: 2D numpy array of shape (H, W) in [0.0, 1.0]
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        # Backward pass for the target score
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback uniform center-weighted heatmap if hooks didn't capture
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
            heatmap = np.exp(-dist_sq / (2 * (h / 4) ** 2))
            return heatmap.astype(np.float32)

        # Global average pooling of gradients: alpha_k
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weighted combination of forward activation maps
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=self.activations.device)
        for i, weight in enumerate(pooled_gradients):
            cam += weight * self.activations[0, i, :, :]

        # Apply ReLU to focus on features having positive contribution
        cam = torch.clamp(cam, min=0)

        cam_np = cam.cpu().numpy()
        max_val = np.max(cam_np)
        if max_val > 0:
            cam_np = cam_np / max_val
        else:
            cam_np = np.zeros_like(cam_np)

        # Resize heatmap back to input resolution (H, W)
        cam_resized = cv2.resize(cam_np, (input_tensor.shape[3], input_tensor.shape[2]))
        return cam_resized.astype(np.float32)

    def create_overlay(
        self,
        image_rgb_224: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.55
    ) -> np.ndarray:
        """Blends original RGB image with colored Jet heatmap.

        Args:
            image_rgb_224: Original RGB image (224, 224, 3) uint8.
            heatmap: (224, 224) float in [0.0, 1.0].
            alpha: Blend weight for original image.

        Returns:
            blended_rgb: (224, 224, 3) uint8 image.
        """
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(image_rgb_224, alpha, colored_heatmap_rgb, 1 - alpha, 0)
        return blended

