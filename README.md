# Hybrid DT+NN Air Quality API

> **Production-ready FastAPI backend for a Hybrid Decision Tree + Neural Network regression model** that predicts **CO concentration (mg/m³)** from UCI Air Quality sensor readings.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange)](https://tensorflow.org)

---

## Model Architecture

```
Raw Sensor Input (12 features)
          │
          ▼
  [StandardScaler]  ← loaded from scaler.pkl
          │
          ▼
  [Decision Tree]   ← loaded from dt_model.pkl
  max_depth=3, min_samples_leaf=1000
          │
    routes to leaf
          │
    ┌─────┴──────┐
    │            │
  Leaf NN     No NN trained
  (≥50 samples)  │
    │           DT prediction
    ▼           (fallback)
  Keras NN  ←── loaded from leaf_models/{id}.h5
  (HPO-tuned per leaf)
    │
    ▼
  CO Prediction (mg/m³)
```

| Model | Test R² | Test Adj. R² |
|---|---|---|
| Decision Tree only | 0.5636 | 0.5608 |
| **Hybrid DT + NN** | **0.8790** | **0.8782** |

---

## Project Structure

```
HybridModel/
│
├── api/
│   └── app.py              ← FastAPI app, routes, Pydantic schemas
│
├── src/
│   ├── config.py           ← Paths, constants, metadata
│   ├── model.py            ← NN builder + artifact loaders
│   ├── preprocess.py       ← Input → scaled array pipeline
│   ├── predict.py          ← HybridPredictor class
│   └── utils.py            ← Logging, JSON I/O, validation
│
├── models/                 ← Trained artifacts (not in git)
│   ├── dt_model.pkl
│   ├── scaler.pkl
│   ├── config.json
│   ├── leaf_metrics.json
│   └── leaf_models/
│       ├── 3.h5
│       ├── 4.h5  ...
│
├── notebooks/
│   ├── 4_dt(1).py          ← Original Colab notebook
│   └── 4_dt_corrected.py   ← Corrected training script
│
├── tests/
│   └── test_predict.py
│
├── main.py                 ← CLI inference test
├── requirements.txt
└── .gitignore
```

---

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Place Model Artifacts

Download the `models/` folder from Google Colab and place it in the project root:

```
HybridModel/
└── models/
    ├── dt_model.pkl
    ├── scaler.pkl
    ├── config.json
    ├── leaf_metrics.json
    └── leaf_models/
        ├── 3.h5
        ├── 4.h5
        ├── 6.h5
        ├── 7.h5
        └── 8.h5
```

---

## Running the API

```bash
uvicorn api.app:app --reload
```

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc
- **Health**: http://127.0.0.1:8000/health

---

## API Endpoints

### `GET /health`
Check model load status and performance metrics.

```json
{
  "status": "ok",
  "model_version": "1.0",
  "n_leaf_models": 5,
  "leaf_ids": [3, 4, 6, 7, 8],
  "hybrid_test_r2": 0.879
}
```

---

### `POST /predict`
Send raw sensor values — the API preprocesses and predicts.

**Request:**
```json
{
  "pt08_s1_co": 1046.0,
  "nmhc_gt": 166.0,
  "c6h6_gt": 11.9,
  "pt08_s2_nmhc": 1056.0,
  "nox_gt": 166.0,
  "pt08_s3_nox": 1056.0,
  "no2_gt": 113.0,
  "pt08_s4_no2": 1692.0,
  "pt08_s5_o3": 1268.0,
  "temperature": 13.6,
  "relative_humidity": 48.9,
  "absolute_humidity": 0.7578
}
```

**Response:**
```json
{
  "success": true,
  "decision_tree_output": {
    "leaf_id": 6,
    "prediction": 2.1
  },
  "neural_network_output": {
    "used_nn": true,
    "prediction": 1.98,
    "leaf_r2": 0.8119,
    "leaf_samples": 1767
  },
  "final_prediction": {
    "co_concentration_mg_m3": 1.98,
    "source": "neural_network"
  },
  "model_version": "1.0"
}
```

---

### `POST /predict/raw`
For advanced users with pre-scaled data. Send a 12-element float array directly.

**Feature order:** `PT08.S1(CO), NMHC(GT), C6H6(GT), PT08.S2(NMHC), NOx(GT), PT08.S3(NOx), NO2(GT), PT08.S4(NO2), PT08.S5(O3), T, RH, AH`

**Request:**
```json
{
  "features": [0.45, -1.2, 0.88, 0.33, -0.71, 1.04, 0.22, -0.5, 0.0, 1.0, 0.0, 1.0]
}
```

---

## CLI Testing

```bash
# Test with sample data
python main.py

# Test with pre-scaled features
python main.py --scaled

# Test with custom JSON input
python main.py --input '{"pt08_s1_co": 1046, "nmhc_gt": 166, ...}'
```

---

## Running Tests

```bash
# Unit tests only (no artifacts needed)
python -m pytest tests/ -v

# Include integration tests (requires models/)
python -m pytest tests/ -v -m integration
```

---

## Dataset

**UCI Air Quality Dataset (id=360)**
- Hourly averaged sensor readings from an Italian city
- 9358 instances, 12 features after preprocessing
- Target: `CO(GT)` — True hourly averaged CO concentration (mg/m³)
- Values of -200 indicate missing sensor data (replaced with column mean)

---

## Deployment Notes

- Tested on Python 3.10+, TensorFlow 2.13+
- The `negative_slope` parameter in `LeakyReLU` requires TF ≥ 2.13
- For Docker deployment, use `tensorflow/tensorflow:2.13.0` as base image
- Model artifacts are excluded from git — distribute separately
