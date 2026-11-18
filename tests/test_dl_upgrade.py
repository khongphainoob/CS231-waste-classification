from pathlib import Path
import numpy as np
import cv2
import pytest
import torch
from src.base_predictor import CLASS_NAMES
from src.member2_deeplearning.model_factory import build_model, gradcam_target_layer
from src.member2_deeplearning.dataset_aug import WasteDataset, get_transforms
from src.member2_deeplearning.audit_data import audit_dataset
from src.member2_deeplearning.predictor_dl import DeepLearningPredictor


def test_model_output_and_no_fallback():
    torch.set_num_threads(2)
    model = build_model("light_cnn", num_classes=6, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 224, 224)).shape == (1, 6)
    with pytest.raises(ValueError):
        build_model("made_up", pretrained=True)
    with pytest.raises(ValueError):
        build_model("light_cnn", pretrained=True)


def test_transforms_are_valid():
    bgr = np.full((80, 60, 3), 128, dtype=np.uint8)
    for is_train in [True, False]:
        x = get_transforms(is_train)(bgr)
        assert x.shape == (3, 224, 224) and x.dtype == torch.float32
        assert torch.isfinite(x).all()


def test_duplicate_detection_and_missing_classes(tmp_path):
    for split in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            p=tmp_path/split/cls
            p.mkdir(parents=True)
            cv2.imwrite(str(p/'im.png'), np.full((20,20,3),
                                                len(cls)*4 + len(split), dtype=np.uint8))
    # Some byte-identical classes across splits? detect and refuse.
    result = audit_dataset(tmp_path)
    assert result["n_files"]==18
    cv2.imwrite(str(tmp_path/'val'/'cardboard'/'other.png'),
                cv2.imread(str(tmp_path/'train'/'cardboard'/'im.png')))
    with pytest.raises(ValueError, match="Data leakage"):
        audit_dataset(tmp_path)


def test_reject_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        DeepLearningPredictor(str(tmp_path/"missing.pth"), fallback_mode=True)


def test_checkpoint_prediction_and_real_cam(tmp_path):
    torch.set_num_threads(2)
    m=build_model('light_cnn').eval()
    path=tmp_path/'weight.pth'
    torch.save({'checkpoint_kind':'trained','model_name':'light_cnn',
                'class_names':list(CLASS_NAMES), 'model_state_dict':m.state_dict(),
                'img_size':224},path)
    p=DeepLearningPredictor(str(path),device='cpu')
    img=np.full((224,224,3),120,dtype=np.uint8)
    result = p.predict(img,with_explanation=False)
    assert result['class_name'] in CLASS_NAMES
    assert abs(sum(result['probabilities'])-1) < 1e-5
    assert result['explanation_latency_ms']==0
    result = p.predict(img,with_explanation=True)
    assert result['visual_explanation'].shape==(224,224,3)
    assert result['explanation_latency_ms'] > 0
    assert result['latency_ms'] > 0
    p.close()


def test_benchmark_uses_current_classical_api(tmp_path):
    from types import SimpleNamespace
    from src.evaluation.benchmark import evaluate_model

    class Extractor:
        def extract(self, image):
            assert image.shape == (224, 224, 3)
            return np.zeros(3)

    class Pipeline:
        classes_ = np.array([4, 1])

        def predict_proba(self, features):
            assert features.shape == (1, 3)
            return np.array([[0.8, 0.2]])

    path = tmp_path / 'image.png'
    cv2.imwrite(str(path), np.full((30, 30, 3), 100, dtype=np.uint8))
    predictor = SimpleNamespace(extractor=Extractor(), pipeline=Pipeline())
    result = evaluate_model(predictor, [(str(path), 4)], 'Classical')
    assert result['y_pred'] == [4]
    assert result['accuracy'] == 100
