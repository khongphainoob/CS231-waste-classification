"""Actual gradient-based Grad-CAM; no synthetic/fallback heatmap."""
import cv2
import numpy as np
import torch
from torch import nn


class GradCAMVisualizer:
    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        if target_layer is None:
            # This fallback only selects an actual convolutional layer, never fakes a CAM.
            candidates = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
            if not candidates:
                raise ValueError("No convolutional layer found for Grad-CAM")
            target_layer = candidates[-1]
        self.target_layer = target_layer
        self.activations = None
        self.handle = self.target_layer.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Grad-CAM target must return 4D spatial activations")
        self.activations = output

    def remove_hooks(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        self.activations = None
        with torch.enable_grad():
            x = input_tensor.detach().clone().requires_grad_(True)
            logits = self.model(x)
            if self.activations is None:
                raise RuntimeError("Grad-CAM did not capture activations")
            if target_class is None:
                target_class = int(logits.argmax(1).item())
            gradients = torch.autograd.grad(logits[0, target_class], self.activations)[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * self.activations).sum(dim=1))[0]
            max_value = cam.max()
            cam = cam / max_value.clamp_min(1e-12)
            heatmap = cv2.resize(cam.detach().float().cpu().numpy(),
                                 (input_tensor.shape[3], input_tensor.shape[2]))
        return heatmap.astype(np.float32)

    @staticmethod
    def create_overlay(image_rgb_224, heatmap, alpha=0.55):
        colored = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(image_rgb_224, alpha, colored, 1 - alpha, 0)
