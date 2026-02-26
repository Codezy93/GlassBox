# 🔍 GlassBox — Counterfactual Explanations Platform

> Moving beyond "Black Box" predictions to **actionable recourse** with Counterfactual Explanations.

Instead of just telling a user *why* they were rejected, GlassBox calculates the exact mathematical boundary of the model and tells them:  
**"If your credit limit was $10,000 higher and you paid on time last month, your loan would have been approved."**

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│  React Dashboard │────▶│  FastAPI Backend      │────▶│  ML Models         │
│  (Vite + React)  │     │  /predict  /explain   │     │  PyTorch DNN       │
│  What-If Sandbox │◀────│  /features            │     │  EBM (InterpretML) │
└─────────────────┘     └──────────────────────┘     └────────────────────┘
                                   │
                          ┌────────▼─────────┐
                          │  DiCE Engine       │
                          │  Counterfactuals   │
                          │  with constraints  │
                          └────────────────────┘
```

## ⚡ Quickstart

### Backend
```bash
cd backend
pip install -r requirements.txt

# 1. Download dataset & preprocess
python -m data.download_dataset
python -m data.preprocess

# 2. Train models
python -m models.train_blackbox
python -m models.train_glassbox

# 3. Benchmark
python -m models.benchmark

# 4. Start API server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| XAI Engine | DiCE (Microsoft Research) + InterpretML |
| Black-Box Model | PyTorch DNN (256→128→64→1) |
| Glassbox Model | Explainable Boosting Machine (EBM) |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Vite |
| Dataset | UCI Credit Default (30K rows, 24 features) |

## 📊 Key Features

- **Counterfactual Explanations** — "What do I need to change to get approved?"
- **Immutability Constraints** — Age, sex, education locked (no unrealistic suggestions)
- **Actionability Constraints** — Suggested changes within realistic ranges
- **Diverse Paths** — 3-4 completely different paths to approval via DPP diversity
- **What-If Sandbox** — Drag sliders and watch prediction probability update in real-time
- **Model Comparison** — Side-by-side DNN vs EBM benchmarks
