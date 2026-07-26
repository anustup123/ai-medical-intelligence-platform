# AI Medical Intelligence Platform

An end-to-end deep learning system that detects **Pneumonia from chest X-ray images**,
explains its predictions with **Grad-CAM**, generates a human-readable **AI-assisted
draft report using an LLM (Claude)**, exposes everything through a **REST API**,
stores prediction history in a **database**, and ships with a simple **web UI**.


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

