# Optimized Colab comparison — 2026-11-23

This result supersedes the earlier one-epoch smoke comparison. It evaluates 716 held-out test images on CPU after three warmup calls; timing includes preprocessing and forward inference, excluding disk I/O, resize and Grad-CAM.

| Model | Best checkpoint epoch | Accuracy | Macro F1 | Mean latency | FPS |
|---|---:|---:|---:|---:|---:|
| LightWasteCNN | 27 | 78.911% | 78.649% | 9.783 ms | 102.217 |
| ResNet18 | 25 | **94.274%** | **94.252%** | 58.061 ms | 17.223 |
| ResNet50 | 16 | **95.251%** | **95.241%** | 123.483 ms | 8.098 |
| MobileNetV4 Conv Small | 29 | 92.877% | 92.837% | 15.983 ms | 62.568 |

The `epochs` values in the CSV identify the checkpoint selected by validation, not necessarily the total configured training budget. This run improves the previous test results for ResNet18 (93.575% → 94.274%) and MobileNetV4 (89.525% → 92.877%). ResNet50 is the strongest measured candidate at 95.251% accuracy and 95.241% Macro F1. The held-out test set remains for final evaluation only.

Raw file: [comparison.csv](../../results/colab_20261123/three_model_comparison/comparison.csv). Normalized metadata: [metadata.json](../../results/colab_20261123/three_model_comparison/metadata.json).

## TTA and ensemble run

Horizontal-flip TTA raised ResNet50 to **95.670% accuracy / 95.663% Macro F1**. MobileNetV4 TTA reached 93.575% / 93.555%. The probability ensemble reached 95.391% / 95.379%, so it did not beat ResNet50 TTA in this run. TTA roughly doubles inference work; the ResNet50 TTA timing was 256.138 ms/image on CPU.

Raw TTA results: [comparison.csv](../../results/colab_20261123/tta_ensemble/comparison.csv).
