# Model comparison — 2026-11-18

The three-model comparison is incomplete: only ResNet18 currently has a verified trained checkpoint.

## Code validation

`python -m pytest tests/test_dl_upgrade.py -q --junitxml=docs/experiments/2026-11-18/pytest.xml`

Result: **6 passed in 12.03 seconds**, Python 3.11.9. [JUnit report](pytest.xml) covers model output/architecture rejection, preprocessing, cross-split duplicate rejection, missing checkpoint rejection, real Grad-CAM execution, and compatibility with the current Classical inference API. These unit tests do not measure trained model accuracy.

## Colab status

[Colab notebook](https://colab.research.google.com/drive/1NndGvSzSNHBBKBbdTjz8gCiByTdxRSrs) connected to a Tesla T4 and completed dependency setup. Training and the three-model benchmark have **not completed**: the dataset bundle still needs to be available to the runtime and Drive access needs to be authorized. No GPU accuracy or latency results are claimed in this report. The repository notebook is at [04_Colab_Three_Model_Comparison.ipynb](../../../notebooks/04_Colab_Three_Model_Comparison.ipynb).

| Model | Status | Test accuracy | Test Macro F1 | CPU inference | Checkpoint size |
|---|---|---:|---:|---:|---:|
| ResNet18 | Evaluated on 716 test images | 93.6% | 93.5% | 79.3 ms/image; 12.6 FPS | 42.7 MiB |
| LightWasteCNN | No verified trained checkpoint | — | — | — | — |
| MobileNetV4 | Existing filename contains a legacy LightWasteCNN checkpoint | — | — | — | — |

ResNet18 checkpoint: epoch 13, validation Macro F1 0.9520438064568975, pretrained initialization, seed 42. Its recorded dataset fingerprint matches the current dataset.

Dataset: 3,337 training images, 715 validation images, 716 test images; six classes. Audit found no exact cross-split duplicate files or synthetic filenames. This does not establish absence of near-duplicate images.

Dataset SHA-256 manifest: `38fe0c6d1f67e073d1402f93e55ed1c5f05d5485ab1fe8d16897326a97839dba`.

Benchmark: CPU, 224×224 input, single-image inference after three warmup calls. Timed latency includes normalization and forward inference, excludes disk loading, image resizing and Grad-CAM. Timing is one local run and can vary with system load. No independent Vietnam test images were available in the integration worktree.

The file named `mobilenetv4_best.pth` reports `model_class=LightWasteCNN`, epoch 1 and `val_acc=0.85`; it lacks the trained-checkpoint marker and dataset fingerprint. The original mock-weight generator writes this exact metadata format with a hard-coded validation score. Do not report that score as measured performance.

To complete the comparison, train LightWasteCNN from scratch and pretrained MobileNetV4 on the same split, choose checkpoints using validation only, then evaluate both once on the same held-out test set. CUDA is unavailable in the selected environment; full CPU training may take hours.

Measured outputs: [benchmark CSV](resnet18/benchmark_comparison.csv), [confusion matrix](resnet18/cm_dl_standard_test_set.png).
