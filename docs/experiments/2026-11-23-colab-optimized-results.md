# Optimized Colab comparison — 2026-11-23

This result supersedes the earlier one-epoch smoke comparison. It evaluates 716 held-out test images on CPU after three warmup calls; timing includes preprocessing and forward inference, excluding disk I/O, resize and Grad-CAM.

| Model | Best checkpoint epoch | Accuracy | Macro F1 | Mean latency | FPS |
|---|---:|---:|---:|---:|---:|
| LightWasteCNN | 27 | 78.911% | 78.649% | 4.951 ms | 201.972 |
| ResNet18 | 25 | **94.274%** | **94.252%** | 24.949 ms | 40.082 |
| MobileNetV4 Conv Small | 29 | 92.877% | 92.837% | 15.392 ms | 64.970 |

The `epochs` values in the CSV identify the checkpoint selected by validation, not necessarily the total configured training budget. This run improves the previous test results for ResNet18 (93.575% → 94.274%) and MobileNetV4 (89.525% → 92.877%). The held-out test set remains for final evaluation only.

Raw file: [comparison.csv](../../results/colab_20261123/three_model_comparison/comparison.csv). Normalized metadata: [metadata.json](../../results/colab_20261123/three_model_comparison/metadata.json).
