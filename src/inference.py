"""
End-to-end inference pipeline:
  raw image bytes -> preprocessing -> model prediction -> Grad-CAM heatmap
  -> saved overlay image -> structured result dict

This module is used both by the FastAPI route and can be run standalone
for quick manual testing:
    python -m src.inference path/to/image.jpg
"""
import io
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import DEVICE, MODEL_PATH, LABELS, GRADCAM_OUTPUT_DIR
from src.model import load_trained_model
from src.dataset import get_transforms
from src.gradcam import GradCAM, denormalize_image, overlay_heatmap, describe_region

_model_cache = {}


def get_model():
    """Lazily loads and caches the trained model (singleton pattern)."""
    if "model" not in _model_cache:
        _model_cache["model"] = load_trained_model(str(MODEL_PATH), device=DEVICE)
    return _model_cache["model"]


def predict_image(image_bytes: bytes) -> dict:
    """
    Runs the full pipeline on raw image bytes and returns a JSON-serializable
    dict with the prediction, confidence, and path to the saved Grad-CAM image.
    """
    model = get_model()
    _, eval_transform = get_transforms()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

    gradcam = GradCAM(model, model.target_layer)
    cam, class_idx, probs = gradcam.generate(input_tensor)

    original = denormalize_image(input_tensor[0])
    overlay = overlay_heatmap(original, cam)

    filename = f"gradcam_{uuid.uuid4().hex[:10]}.jpg"
    out_path = GRADCAM_OUTPUT_DIR / filename
    Image.fromarray(overlay).save(out_path, quality=90)

    attention_region = describe_region(cam)

    return {
        "predicted_class": LABELS[class_idx],
        "confidence": float(probs[class_idx]),
        "probabilities": {LABELS[i]: float(p) for i, p in enumerate(probs)},
        "gradcam_image_path": str(out_path),
        "gradcam_image_filename": filename,
        "attention_region": attention_region,
        "timestamp": time.time(),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference <path_to_image>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        result = predict_image(f.read())
    print(result)
