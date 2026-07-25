"""
FastAPI REST API for the AI Medical Intelligence Platform.

Endpoints:
    GET  /api/health              -> service + model status
    POST /api/predict             -> upload chest X-ray image, get prediction + Grad-CAM + LLM report
    GET  /api/history             -> paginated list of past predictions
    GET  /api/history/{id}        -> single prediction detail
    DELETE /api/history/{id}      -> delete a prediction record
    GET  /gradcam/{filename}      -> serve a saved Grad-CAM overlay image

Run with:
    uvicorn api.main:app --reload --port 8000
"""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # allow `src.*` imports

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from src.config import DEVICE, MODEL_PATH, GRADCAM_OUTPUT_DIR, BASE_DIR
from src.database import init_db, get_db, PredictionRecord
from src.schemas import PredictionResponse, HistoryItem, HealthResponse
from src.inference import predict_image, get_model
from src.llm_report import generate_report

app = FastAPI(
    title="AI Medical Intelligence Platform",
    description="Deep learning pneumonia detection with Grad-CAM explainability and LLM-generated reports.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_FILE_SIZE_MB = 10


@app.on_event("startup")
def startup_event():
    init_db()
    # Warm up the model so the first real request isn't slow (optional).
    if MODEL_PATH.exists():
        try:
            get_model()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"WARNING: could not load model at startup: {e}")
    else:
        print(f"WARNING: no trained model found at {MODEL_PATH}. "
              f"Train one first with `python -m src.train`.")


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=MODEL_PATH.exists(),
        device=DEVICE,
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400,
                             detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG or PNG image.")

    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503,
                             detail="No trained model found on the server. Train the model first "
                                    "(see README) before calling /api/predict.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")

    try:
        result = predict_image(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    llm_report = generate_report(result)

    record = PredictionRecord(
        original_filename=file.filename,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probability_normal=result["probabilities"].get("NORMAL"),
        probability_pneumonia=result["probabilities"].get("PNEUMONIA"),
        attention_region=result["attention_region"],
        gradcam_image_path=result["gradcam_image_filename"],
        llm_report=llm_report,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return PredictionResponse(
        id=record.id,
        predicted_class=record.predicted_class,
        confidence=record.confidence,
        probabilities={"NORMAL": result["probabilities"].get("NORMAL"),
                        "PNEUMONIA": result["probabilities"].get("PNEUMONIA")},
        attention_region=record.attention_region,
        gradcam_image_url=f"/gradcam/{record.gradcam_image_path}",
        llm_report=record.llm_report,
        created_at=record.created_at,
    )


@app.get("/api/history", response_model=list[HistoryItem])
def get_history(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    records = (db.query(PredictionRecord)
               .order_by(PredictionRecord.created_at.desc())
               .offset(skip).limit(limit).all())
    return [
        HistoryItem(
            id=r.id, original_filename=r.original_filename,
            predicted_class=r.predicted_class, confidence=r.confidence,
            attention_region=r.attention_region,
            gradcam_image_url=f"/gradcam/{r.gradcam_image_path}",
            created_at=r.created_at,
        ) for r in records
    ]


@app.get("/api/history/{record_id}", response_model=PredictionResponse)
def get_history_item(record_id: int, db: Session = Depends(get_db)):
    r = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Record not found.")
    return PredictionResponse(
        id=r.id, predicted_class=r.predicted_class, confidence=r.confidence,
        probabilities={"NORMAL": r.probability_normal, "PNEUMONIA": r.probability_pneumonia},
        attention_region=r.attention_region,
        gradcam_image_url=f"/gradcam/{r.gradcam_image_path}",
        llm_report=r.llm_report, created_at=r.created_at,
    )


@app.delete("/api/history/{record_id}")
def delete_history_item(record_id: int, db: Session = Depends(get_db)):
    r = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Record not found.")
    db.delete(r)
    db.commit()
    return {"deleted": record_id}


@app.get("/gradcam/{filename}")
def get_gradcam_image(filename: str):
    path = GRADCAM_OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path)


# Serve the simple HTML/JS/CSS frontend at the root URL
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
