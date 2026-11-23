"""Exactly reproducible architecture selection for CS231's six-class waste classifier."""
import torch
from torch import nn

SUPPORTED_MODELS = ("light_cnn", "resnet18", "resnet50", "mobilenetv4_conv_small")


class LightWasteCNN(nn.Module):
    """Small 5-block CNN trained from scratch (no pretrained backbone)."""
    def __init__(self, num_classes=6):
        super().__init__()
        layers = []
        channels = (3, 32, 64, 128, 256, 384)
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
                       nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(384, num_classes))

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.features(x)), 1))


def build_model(model_name="light_cnn", num_classes=6, pretrained=False):
    """Never substitute an architecture if loading fails."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from {SUPPORTED_MODELS}")
    if model_name == "light_cnn":
        if pretrained:
            raise ValueError("light_cnn must train from scratch (pretrained=False).")
        return LightWasteCNN(num_classes=num_classes)
    try:
        import timm
    except ImportError as exc:
        raise ImportError("Install timm: pip install timm>=1.0.8") from exc
    try:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot create {model_name!r} (pretrained={pretrained}); "
            "check timm version / internet / weight cache. No fallback is allowed."
        ) from exc


def set_backbone_trainable(model, model_name, trainable):
    """Frozen backbone, trainable classifier during optional warmup."""
    if model_name == "light_cnn":
        return
    for p in model.parameters():
        p.requires_grad_(trainable)
    if not trainable:
        head = model.get_classifier()
        if head is None:
            raise RuntimeError(f"{model_name} has no detectable classifier")
        for p in head.parameters():
            p.requires_grad_(True)


def gradcam_target_layer(model, model_name):
    if model_name == "light_cnn":
        return model.features[-1]
    if model_name in ("resnet18", "resnet50"):
        return model.layer4[-1]
    if model_name == "mobilenetv4_conv_small":
        # timm MobileNetV4 ConvSmall is implemented as MobileNetV3, using
        # `blocks`, not `stages`. Hook the last block BEFORE global pooling.
        if hasattr(model, "blocks") and len(model.blocks):
            return model.blocks[-1]
    # Fail explicitly if implementation/version changes rather than producing fake heatmaps.
    raise ValueError(f"Cannot determine Grad-CAM target for {model_name}")
