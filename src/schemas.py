"""Pydantic schemas — define the shape of API request/response JSON bodies."""
from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    id: int
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    attention_region: str
    gradcam_image_url: str
    llm_report: str
    created_at: datetime

    class Config:
        from_attributes = True


class HistoryItem(BaseModel):
    id: int
    original_filename: Optional[str]
    predicted_class: str
    confidence: float
    attention_region: Optional[str]
    gradcam_image_url: str
    created_at: datetime

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
