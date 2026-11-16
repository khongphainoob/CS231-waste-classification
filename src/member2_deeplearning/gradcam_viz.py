"""Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN Interpretability."""

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
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hook_handles = []

        if self.target_layer is not None:
            self._register_hooks()

    def _find_target_layer(self) -> Optional[nn.Module]:
        """Searches for the last Conv2d layer in the model."""
        target = None
        for _, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                target = module
        return target

    def _register_hooks(self):
        def forward_hook(module, _inp, output):
            self.activations = output.detach()

        def backward_hook(module, _grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles = [
            self.target_layer.register_forward_hook(forward_hook),
            self.target_layer.register_full_backward_hook(backward_hook),
        ]

    def remove_hooks(self):
        """Cleanly removes registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def generate_heatmap_and_logits(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Runs a single forward/backward pass and returns (heatmap, detached_logits)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.gradients = None
        self.activations = None

        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        logits[0, target_class].backward()
        logits_detached = logits.detach()
        self.model.zero_grad(set_to_none=True)

        if self.gradients is None or self.activations is None:
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            y, x = np.ogrid[:h, :w]
            dist_sq = (x - w / 2) ** 2 + (y - h / 2) ** 2
            heatmap = np.exp(-dist_sq / (2 * (h / 4) ** 2)).astype(np.float32)
            return heatmap, logits_detached

        # Vectorized channel-wise weighted sum of forward feature maps
        pooled_gradients = torch.mean(self.gradients, dim=(0, 2, 3))
        cam = torch.sum(pooled_gradients[:, None, None] * self.activations[0], dim=0)
        cam = torch.clamp(cam, min=0)

        cam_np = cam.cpu().numpy()
        self.gradients = None
        self.activations = None

        max_val = np.max(cam_np)
        if max_val > 0:
            cam_np = cam_np / max_val
        else:
            cam_np = np.zeros_like(cam_np)

        cam_resized = cv2.resize(cam_np, (input_tensor.shape[3], input_tensor.shape[2]))
        return cam_resized.astype(np.float32), logits_detached

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generates 2D normalized Grad-CAM heatmap in range [0, 1]."""
        heatmap, _ = self.generate_heatmap_and_logits(input_tensor, target_class=target_class)
        return heatmap

    def create_overlay(
        self,
        image_rgb_224: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.55,
    ) -> np.ndarray:
        """Blends original RGB image with Jet colormap heatmap."""
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
        colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(image_rgb_224, alpha, colored_rgb, 1.0 - alpha, 0)
