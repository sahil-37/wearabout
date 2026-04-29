# Buy Me That Look

AI-powered fashion recommendation system — upload an outfit photo and get shoppable product matches from a 994-item Myntra catalog.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![React](https://img.shields.io/badge/React-19-61DAFB)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)

---

## How it works

```
Upload outfit photo
      │
      ▼
YOLOv5 ONNX  ──────────── detects clothing items (topwear, bottomwear, footwear, eyewear, handbag)
      │
      ▼
ResNet-50 (PyTorch) ────── extracts 2048-d feature embedding
      │
      ▼
FAISS IndexFlatIP ──────── cosine similarity search across 994 catalog products
      │
      ▼
Ranked recommendations with brand, price, and Myntra buy link
```

---

## Stack

| Layer | Tech |
|---|---|
| API | FastAPI + Uvicorn |
| Detection | YOLOv5 ONNX (fashion-tuned) |
| Features | ResNet-50 ImageNet embeddings (PyTorch) |
| Search | FAISS `IndexFlatIP` — per-category + global indices |
| Frontend | React + Vite + Tailwind CSS |
| MLOps | MLflow experiment tracking, versioned catalog |
| Monitoring | Prometheus metrics |

---

## Local setup

**Requirements:** Python 3.11, Node 18+

```bash
# 1 — clone
git clone <repo-url>
cd buy_me_that_look_api

# 2 — Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3 — environment config
cp .env.example .env

# 4 — build product catalog
make build-catalog

# 5 — start API
python main.py

# 6 — start frontend (separate terminal)
make frontend-dev
```

API → `http://localhost:8000` · Frontend → `http://localhost:5173` · Docs → `http://localhost:8000/api/docs`

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/recommend` | Full outfit recommendations |
| `POST` | `/api/v1/find-similar` | Find similar items (no pose check) |
| `POST` | `/api/v1/detect-items` | Detect clothing items in image |
| `POST` | `/api/v1/extract-features` | Extract ResNet-50 feature vector |
| `GET` | `/api/docs` | Swagger UI |
| `GET` | `/metrics` | Prometheus metrics |

---

## MLOps pipeline

```bash
make retrain        # rebuild catalog → evaluate → promote if metrics improve
make retrain-dry    # evaluate only, no promotion
make evaluate       # Recall@K + Precision@K against annotated val set
make mlflow-ui      # experiment tracking UI on :5001
```

Each run logs to MLflow: feature extractor, Recall@K, Precision@K (global + per-category), delta vs baseline, catalog artifacts.

---

## Project structure

```
├── app/
│   ├── engines/        # recommendation pipeline
│   ├── models/         # YOLOv5 detector, ResNet-50 extractor, pose estimator
│   ├── routes/         # FastAPI route handlers
│   ├── middleware/     # auth, Prometheus metrics
│   └── config.py
├── frontend/           # React + Vite + Tailwind
├── scripts/
│   ├── build_catalog.py   # build feature store from product images
│   ├── evaluate.py        # Recall@K / Precision@K evaluation
│   └── retrain.py         # full MLOps retraining pipeline
├── tests/
├── data/               # gitignored — feature store + metadata
├── models/             # gitignored — ONNX weights
└── Makefile
```
