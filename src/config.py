"""
Central configuration for the AI Medical Intelligence Platform.
All paths, hyperparameters and constants live here so nothing is hard-coded
inside the training / inference / API code.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "chest_xray" / "train"
VAL_DIR = DATA_DIR / "chest_xray" / "val"
TEST_DIR = DATA_DIR / "chest_xray" / "test"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "pneumonia_densenet121.pt"
LABELS = ["NORMAL", "PNEUMONIA"]

GRADCAM_OUTPUT_DIR = BASE_DIR / "reports" / "gradcam_outputs"
GRADCAM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = BASE_DIR / "predictions.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# ---------------------------------------------------------------------------
# Image / model hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224          # DenseNet121 default input size
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_CLASSES = 2
RANDOM_SEED = 42

# ImageNet normalization statistics (required because we fine-tune a model
# that was pretrained on ImageNet)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# LLM (Anthropic Claude) settings — used for AI-generated report narration
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
LLM_MAX_TOKENS = 700

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
import torch  # noqa: E402
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
