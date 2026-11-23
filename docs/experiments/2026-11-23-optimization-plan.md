# Optimization plan — 2026-11-23

The current Colab run reaches 93.575% test accuracy and 93.524% test Macro F1 with ResNet18 after 15 epochs. Its training accuracy reaches 99.071% while validation Macro F1 peaks at 94.940%, so the limiting issue is generalization rather than underfitting.

The training pipeline now supports `--label_smoothing`; the Colab content-only notebook uses 30 epochs, learning rate `0.0002`, patience `8`, and label smoothing `0.05` for all three models. Pretrained ResNet18 and MobileNetV4 keep a three-epoch frozen-backbone warm-up; LightWasteCNN remains from scratch. The best checkpoint is still selected from validation Macro F1, with validation loss as the tie-breaker.

This change does not claim a new accuracy score until the updated notebook is run. The held-out test split remains untouched for model selection. A 97–99% test score is not guaranteed and should not be manufactured by tuning against test images; train accuracy can legitimately reach that range while test performance remains lower.

Validation after the code change: 6 regression tests passed; `train_cnn.py` and `benchmark_three_models.py` compile successfully; the new `--label_smoothing` CLI option is exposed.
