# CS231 — Deep Learning branch (Member 2)

This overlay works **inside** the original repository `khongphainoob/CS231-waste-classification`, without replacing the Classical CV training code. All classes and output fields follow `src/base_predictor.py`.

## 1. Installation (Windows PowerShell)

```powershell
git clone https://github.com/khongphainoob/CS231-waste-classification.git
cd CS231-waste-classification
python -m pip install -r requirements.txt
python -m pip install "timm>=1.0.8" "scikit-learn>=1.3" "pytest>=7"
```

Run all commands below from the repository root. Use Python 3.10/3.11 or another version compatible with the pinned dependency environment. `timm` pretrained checkpoints require network/cached weights; if unavailable, use `light_cnn` for an actual offline training run, not a fake fallback.

## 2. Prepare real images

Download the 12-class Kaggle Garbage Classification set referenced by the repository, then:

```powershell
python data/download_and_split.py --raw_dir data/raw/garbage-classification
python -m src.member2_deeplearning.audit_data --data data/standard_split
```

Expected folder names (`train`, `val`, `test`) each contain **exactly these names**:
`cardboard`, `glass`, `metal`, `paper`, `plastic`, `biological`.

**Important:** the original splitter samples up to 800 images per class; if any class has fewer, dataset size is smaller. Its `--samples_per_class` argument does not change its hard-coded 560/120/120 offsets. Confirm the actual counts in `dataset_audit.json` rather than assuming 4,800. The audit detects byte-for-byte duplicates across splits, not visually similar near-duplicates. For a rigorous paper, inspect perceptual duplicates and collection provenance as well. Use the **same** split for Traditional ML and DL.

`data/download_and_split.py --synthetic` is useful for wiring/debugging only. Synthetic images must never be used to claim a real test score. `--allow_synthetic` is deliberately labeled DEBUG ONLY.

## 3. Run experiments

Start with ResNet18 pretrained:

```powershell
python -m src.member2_deeplearning.train_cnn --model resnet18 --epochs 25 --freeze_epochs 3 --batch_size 32 --lr 0.0003
```

Then train the other two models:

```powershell
python -m src.member2_deeplearning.train_cnn --model light_cnn --epochs 35 --batch_size 32 --lr 0.0003
python -m src.member2_deeplearning.train_cnn --model mobilenetv4_conv_small --epochs 30 --freeze_epochs 3 --batch_size 32 --lr 0.0003
```

Use `--workers 0` if Windows multiprocessing creates a data-loader error. GPU is selected automatically. Defaults `img_size=224`, `seed=42`. `--no_pretrained` is available for ablation, not for the default Transfer Learning comparison.

Outputs per model:

- `weights/resnet18_best.pth`, `weights/light_cnn_best.pth`, or `weights/mobilenetv4_best.pth`.
- `results/dl/<model>/history.csv`, `summary.json`, `dataset_audit.json`.

Checkpoint selection: best **validation Macro F1** (lower validation loss breaks ties). Early stopping uses validation only. Do **not** use test scores to choose training hyperparameters or model selection.

## 4. Evaluate held-out TEST data

```powershell
python -m src.evaluation.benchmark --dl_only --dl_weights weights/resnet18_best.pth
```

When teammate's **trained** Classical SVM pipeline exists:

```powershell
python -m src.evaluation.benchmark --dl_weights weights/mobilenetv4_best.pth --ml_weights weights/classical_svm_pipeline.pkl
```

Optional independently collected real-world set (do not include synthetic images):

```powershell
python -m src.evaluation.benchmark --dl_weights weights/mobilenetv4_best.pth --ml_weights weights/classical_svm_pipeline.pkl --vn_dir data/vn_real_test
```

Results: `results/benchmark_comparison.csv`, `results/benchmark_per_class.csv` when comparing both branches, confusion matrix PNGs, and disagreement-case grids. Models **cannot** silently make random predictions. The benchmark measures preprocessing+classification for both branches while excluding the Grad-CAM/HOG *visualizations*. The inference model initialization and disk I/O are excluded, but GPU and CPU metrics are **not** directly comparable—rerun both branches on the same CPU for performance comparisons if you want a common hardware baseline. The default benchmark **forces Deep Learning to CPU** for a matched CPU-vs-CPU comparison with the Classical SVM. Use `--dl_device auto` or `--dl_device cuda` for separate GPU throughput measurements; state the hardware in your report.

## 5. Single-image demo / Grad-CAM

```python
from src.member2_deeplearning.predictor_dl import DeepLearningPredictor
import cv2
m = DeepLearningPredictor("weights/resnet18_best.pth")
bgr = cv2.imread("example.jpg")
rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
r = m.predict(rgb, with_explanation=True)
print(r["class_name"], r["confidence"], r["latency_ms"], r["explanation_latency_ms"])
cv2.imwrite("gradcam.png", cv2.cvtColor(r["visual_explanation"], cv2.COLOR_RGB2BGR))
m.close()
```

The pre-existing Streamlit/webcam examples still choose `weights/mobilenetv4_best.pth` by default, so train MobileNetV4 before starting them, or manually change the weight path to the chosen model. They will now refuse to run when there is no genuine trained checkpoint, rather than showing synthetic predictions.

## 6. Scientific reporting checklist

- Dataset source + license and exact image counts, class mapping, split seed.
- Same train/val/test across methods; no augmentation for val/test; exact-duplicate audit.
- Custom CNN from scratch vs ImageNet pretrained ResNet18 and MobileNetV4.
- Training hyperparameters, best validation checkpoint, seed, device, run durations.
- Final accuracy, Macro Precision/Recall/F1, confusion matrices and confusion pairs on held-out test.
- Per-class errors, Grad-CAM examples and limitations of explanations (attention maps are not causal proofs).
- Model size, inference latency measured on a matched device; separate visualization overhead.
- No metrics should be copied from README's illustrative numbers or from `generate_mock_weights.py`.

## 7. Notebook

Open `notebooks/03_Member2_DeepLearning_CNN_GradCAM.ipynb` from the repository root in Jupyter/VS Code or Google Colab with your repository mounted. Sections cover EDA, data validation, training and curves, test evaluation, and Grad-CAM. It invokes the same scripts as the CLI.
