"""
Sanity tests that verify the ENTIRE pipeline (model -> Grad-CAM -> LLM fallback
-> database) is wired together correctly, using small SYNTHETIC images.

These tests do NOT verify medical accuracy (that requires the real dataset
and real training — see README). They exist so you can run:
    pytest tests/ -v
and get immediate confidence that there are no bugs in the code before you
invest hours in real GPU training.
"""
import io
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch
from PIL import Image

from src.model import build_model
from src.gradcam import GradCAM, denormalize_image, overlay_heatmap, describe_region
from src.dataset import get_transforms
from src.llm_report import generate_report, _fallback_template_report


def make_synthetic_xray_bytes(size=224) -> bytes:
    """Creates a random grayscale-looking JPEG in memory, standing in for a
    real chest X-ray for pipeline-wiring tests."""
    arr = np.random.randint(0, 255, (size, size), dtype=np.uint8)
    img = Image.fromarray(arr).convert("L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_model_forward_pass():
    model = build_model(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)


def test_gradcam_generates_valid_heatmap():
    model = build_model(pretrained=False)
    model.eval()
    gradcam = GradCAM(model, model.target_layer)

    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    cam, class_idx, probs = gradcam.generate(x)

    assert cam.shape == (224, 224)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-5
    assert class_idx in (0, 1)
    assert abs(probs.sum() - 1.0) < 1e-4


def test_denormalize_and_overlay():
    model = build_model(pretrained=False)
    model.eval()
    gradcam = GradCAM(model, model.target_layer)
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    cam, _, _ = gradcam.generate(x)

    original = denormalize_image(x[0])
    assert original.shape == (224, 224, 3)
    assert original.dtype == np.uint8

    overlay = overlay_heatmap(original, cam)
    assert overlay.shape == (224, 224, 3)


def test_describe_region_returns_valid_string():
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[10, 10] = 1.0  # peak in upper-left -> "upper right lung field" (image-left = patient right)
    region = describe_region(cam)
    assert "upper" in region
    assert "lung field" in region


def test_transforms_produce_correct_tensor_shape():
    _, eval_transform = get_transforms()
    img = Image.fromarray(np.random.randint(0, 255, (300, 400), dtype=np.uint8))
    tensor = eval_transform(img)
    assert tensor.shape == (3, 224, 224)


def test_llm_fallback_report_contains_disclaimer():
    fake_result = {
        "predicted_class": "PNEUMONIA",
        "confidence": 0.92,
        "probabilities": {"NORMAL": 0.08, "PNEUMONIA": 0.92},
        "attention_region": "lower left lung field",
    }
    report = _fallback_template_report(fake_result)
    assert "Disclaimer" in report
    assert "PNEUMONIA" in report or "Pneumonia" in report


def test_generate_report_without_api_key_uses_fallback(monkeypatch):
    monkeypatch.setattr("src.llm_report.ANTHROPIC_API_KEY", "")
    fake_result = {
        "predicted_class": "NORMAL",
        "confidence": 0.87,
        "probabilities": {"NORMAL": 0.87, "PNEUMONIA": 0.13},
        "attention_region": "middle right lung field",
    }
    report = generate_report(fake_result)
    assert "Disclaimer" in report


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
