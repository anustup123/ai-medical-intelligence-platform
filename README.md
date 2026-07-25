# AI Medical Intelligence Platform

An end-to-end deep learning system that detects **Pneumonia from chest X-ray images**,
explains its predictions with **Grad-CAM**, generates a human-readable **AI-assisted
draft report using an LLM (Claude)**, exposes everything through a **REST API**,
stores prediction history in a **database**, and ships with a simple **web UI**.

> ⚠️ **Disclaimer**: This is an educational/portfolio project, not a certified medical
> device. It must never be used for real clinical diagnosis without review by a
> licensed physician.

---

## 1. Architecture Overview

```
┌─────────────┐     upload X-ray      ┌───────────────────┐
│  Web UI      │ ───────────────────▶ │   FastAPI REST API │
│ (HTML/JS/CSS)│ ◀─────────────────── │    (api/main.py)    │
└─────────────┘   JSON response       └─────────┬──────────┘
                                                  │
                       ┌──────────────────────────┼───────────────────────────┐
                       ▼                          ▼                           ▼
              ┌─────────────────┐      ┌─────────────────────┐   ┌──────────────────────┐
              │ Deep Learning    │      │ Explainable AI       │   │ LLM Report Generator  │
              │ DenseNet121 CNN  │─────▶│ Grad-CAM              │──▶│ (Anthropic Claude API)│
              │ (src/model.py)   │      │ (src/gradcam.py)      │   │ (src/llm_report.py)   │
              └─────────────────┘      └─────────────────────┘   └──────────────────────┘
                                                  │
                                                  ▼
                                      ┌─────────────────────┐
                                      │  SQLite / PostgreSQL  │
                                      │  (src/database.py)    │
                                      └─────────────────────┘
```

**Design principle**: the deep learning model makes the prediction; Grad-CAM explains
*where* it looked; the LLM only ever sees *structured numeric output* (class, confidence,
attention region) and turns it into readable prose — it never sees the raw image and
never makes its own diagnosis. This separation keeps the system auditable and safe.

## 2. Task & Dataset

**Task**: Binary classification — `NORMAL` vs `PNEUMONIA` chest X-rays.

**Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
(Kermany et al., via Kaggle). ~5,863 JPEG images, 2 classes.

👉 **You must download this yourself** — see [`data/README.md`](data/README.md) for exact
steps (Kaggle account + `kaggle datasets download` command). It is not bundled in this
repo because of size (~1.2GB) and Kaggle's redistribution terms.

## 3. Model & Algorithms

| Component | Choice | Why |
|---|---|---|
| Backbone | DenseNet121, pretrained on ImageNet | Most widely validated CNN for chest X-ray tasks in literature (CheXNet); dense connectivity gives clean Grad-CAM maps |
| Technique | Transfer learning (fine-tune classifier head, optionally freeze conv backbone first) | Small medical datasets don't have enough data to train a CNN from scratch |
| Class imbalance | Weighted cross-entropy loss | The dataset has ~3x more PNEUMONIA than NORMAL images |
| Augmentation | Random flip, rotation, color jitter | Improves generalization on ~5k images |
| Optimizer | Adam + ReduceLROnPlateau scheduler | Standard, robust default |
| Explainability | Custom Grad-CAM (Selvaraju et al., 2017) implemented from scratch | Full transparency for your write-up; no black-box library |
| LLM | Anthropic Claude API (`claude-sonnet-4-6`) | Converts structured predictions into a readable draft report |
| API | FastAPI | Async, auto-generated OpenAPI docs at `/docs` |
| Database | SQLAlchemy ORM + SQLite (swappable to PostgreSQL via `DATABASE_URL`) | Zero-config for the assignment, production-ready path included |
| Deployment | Docker + docker-compose | Portable, reproducible |

## 4. Project Structure

```
ai-medical-intelligence-platform/
├── src/                    # Core ML/business logic
│   ├── config.py           # All paths & hyperparameters
│   ├── dataset.py          # Data loading & augmentation
│   ├── model.py             # DenseNet121 transfer-learning model
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Test-set metrics, confusion matrix, ROC curve
│   ├── gradcam.py           # Grad-CAM implementation (from scratch)
│   ├── inference.py         # Full predict pipeline (image -> result dict)
│   ├── llm_report.py         # Claude API report generation (+ offline fallback)
│   ├── database.py           # SQLAlchemy models
│   └── schemas.py            # Pydantic request/response schemas
├── api/
│   └── main.py               # FastAPI app & REST endpoints
├── frontend/                 # Plain HTML/CSS/JS single-page UI
├── tests/
│   └── test_pipeline.py      # Pipeline sanity tests (synthetic data — see §7)
├── data/README.md            # Dataset download instructions
├── models/                   # Trained model checkpoint goes here (.gitignored)
├── reports/                  # Generated evaluation plots, Grad-CAM outputs
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## 5. Setup & Installation

```bash
# 1. Clone your repo and enter it
git clone <your-repo-url>
cd ai-medical-intelligence-platform

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and paste your ANTHROPIC_API_KEY (see §8 below)

# 5. Download the dataset — see data/README.md
```

## 6. Training the Model

```bash
# Full fine-tune (recommended if you have a GPU — Colab free tier is fine)
python -m src.train --epochs 15 --batch-size 32 --lr 1e-4

# Faster CPU-only first pass: freeze the conv backbone, train classifier head only
python -m src.train --epochs 10 --batch-size 16 --freeze-backbone
```

This saves the best checkpoint (by validation accuracy) to
`models/pneumonia_densenet121.pt`, and per-epoch metrics to
`reports/training_history.json`.

Then generate the confusion matrix / ROC curve / classification report:
```bash
python -m src.evaluate
```
Outputs: `reports/confusion_matrix.png`, `reports/roc_curve.png`,
`reports/test_classification_report.json`.

**Expected performance** (typical results reported in literature for this exact
dataset/architecture combo): ~90-95% test accuracy, ~0.95+ recall on PNEUMONIA class.
Your actual numbers will depend on epochs, augmentation, and the train/val split you use
— report your real numbers from `test_classification_report.json` in your PDF report.

## 7. Running Tests

```bash
pytest tests/ -v
```
These tests use small synthetic (randomly generated) images to verify the entire
code pipeline — model forward pass, Grad-CAM heatmap generation, image
denormalization, LLM fallback reporting — is wired together correctly, **without**
requiring the real dataset or a trained model. Run these first, before investing
time in real training, to catch bugs early. (All 7 tests pass as shipped.)

## 8. LLM (Claude) API Setup — 🔑 you need to do this yourself

1. Go to https://console.anthropic.com/ and create an account.
2. Generate an API key.
3. Put it in your `.env` file: `ANTHROPIC_API_KEY=sk-ant-...`
4. That's it — `src/llm_report.py` picks it up automatically.

**If you don't set a key**, the app still works end-to-end: `llm_report.py`
automatically falls back to a deterministic, rule-based templated report so the
rest of the system (API, DB, frontend) remains fully demoable. This is documented
in the code (`_fallback_template_report`) and is a legitimate design decision to
mention in your report (graceful degradation).

## 9. Running the API + Web App

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
- Web UI: http://localhost:8000/
- Interactive API docs (Swagger): http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

### REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Service + model status |
| POST | `/api/predict` | Upload an image (`multipart/form-data`, field `file`) → prediction + Grad-CAM + LLM report |
| GET | `/api/history?skip=0&limit=20` | Paginated prediction history |
| GET | `/api/history/{id}` | Full detail of one past prediction |
| DELETE | `/api/history/{id}` | Delete a record |
| GET | `/gradcam/{filename}` | Serves a saved Grad-CAM overlay image |

Example `curl`:
```bash
curl -X POST -F "file=@sample_xray.jpg" http://localhost:8000/api/predict
```

## 10. Docker Deployment

```bash
docker build -t ai-medical-platform .
docker run -p 8000:8000 --env-file .env -v $(pwd)/models:/app/models ai-medical-platform
```
or simply:
```bash
docker compose up --build
```

## 11. Deploying to a Public URL — 🚀 you need to do this yourself

I can't create a live public URL from this sandboxed environment (no outbound
internet access here beyond package registries). Pick any of these free/low-cost
options and push this repo there:

- **Render** (https://render.com) — "New Web Service" → connect your GitHub repo →
  it auto-detects the `Dockerfile` → done. Add `ANTHROPIC_API_KEY` under
  Environment Variables.
- **Railway** (https://railway.app) — similar one-click GitHub deploy, also reads
  the `Dockerfile`.
- **Hugging Face Spaces** (Docker SDK) — free CPU tier, good for portfolio demos.
- **Fly.io** — `fly launch` auto-detects the Dockerfile.

⚠️ Note: your trained model file (`models/pneumonia_densenet121.pt`) is typically
50-100MB — too large for a normal git push in many cases. Use **Git LFS**
(`git lfs track "*.pt"`) or attach it as a **GitHub Release asset** and download it
in a startup script / `Dockerfile` step instead of committing it directly.

## 12. What You Still Need To Do Yourself

This project is fully built and tested (see §13), but 3 things require your own
accounts/hardware and can't be done from this sandbox:

1. **Download the dataset** (Kaggle account needed) — see `data/README.md`.
2. **Train the model** on real data. Free GPU option: Google Colab
   (Runtime → Change runtime type → GPU), clone this repo there, run
   `python -m src.train`. A full run (15 epochs) takes ~20-40 minutes on a
   Colab T4 GPU.
3. **Get an Anthropic API key** for the LLM report feature (§8), and/or **deploy
   to a public host** (§11) if your assignment requires a live link.

## 13. What Was Already Built & Verified

Every file in this repository is complete, working code — not a stub. Before
delivering it, the pipeline was tested end-to-end in a sandboxed environment
using a tiny synthetic dataset (since the real Kaggle dataset requires your own
account to download, and this sandbox has no GPU/internet access to Kaggle):

- ✅ All 7 unit tests pass (`pytest tests/ -v`)
- ✅ Training loop runs, computes weighted loss, saves best checkpoint
- ✅ Grad-CAM generates a valid heatmap and overlay image from a real checkpoint
- ✅ LLM fallback report generation produces a correctly structured report
- ✅ FastAPI server starts, loads the model, and serves `/api/health`
- ✅ `/api/predict` returns a full prediction + Grad-CAM image + report, and
  correctly writes a row to the database
- ✅ `/api/history` correctly retrieves the saved record
- ✅ The static frontend (`/`, `/style.css`, `/script.js`) is served correctly
  by the same FastAPI app

One real bug was found and fixed during this testing (a `.detach()` call missing
before a `.numpy()` conversion in `gradcam.py`) — this is a normal part of software
engineering and worth mentioning in your report as evidence of testing rigor.

## 14. Ethical & Safety Notes (worth including in your report)

- The system is explicitly framed as a **decision-support draft tool**, not an
  autonomous diagnostic system — the LLM prompt (`src/llm_report.py`) is
  instructed to never claim certainty and to always include a disclaimer.
- The LLM is **never given the raw image** — only structured, already-computed
  numeric output. This avoids the LLM hallucinating visual findings it can't
  actually see.
- Class-weighted loss addresses dataset imbalance to avoid a model that just
  learns to predict the majority class.
- Grad-CAM exists specifically so a clinician (or evaluator) can sanity-check
  *why* the model made a call, not just trust a black-box number.

## 15. License / Attribution

Dataset: Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by
Image-Based Deep Learning," *Cell* 172.5 (2018), distributed via Kaggle.
Grad-CAM: Selvaraju et al., ICCV 2017.
