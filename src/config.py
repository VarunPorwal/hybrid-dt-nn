"""
config.py — Central configuration for paths, constants, and metadata.
All other modules import from here — never hardcode paths elsewhere.
"""

from pathlib import Path

# ── Directory Paths ────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
MODELS_DIR      = BASE_DIR / "models"
LEAF_MODELS_DIR = MODELS_DIR / "leaf_models"
DATA_DIR        = BASE_DIR / "data"
NOTEBOOKS_DIR   = BASE_DIR / "notebooks"

# ── Artifact File Paths ────────────────────────────────────────────────────────
DT_MODEL_PATH     = MODELS_DIR / "dt_model.pkl"
SCALER_PATH       = MODELS_DIR / "scaler.pkl"
CONFIG_PATH       = MODELS_DIR / "config.json"
LEAF_METRICS_PATH = MODELS_DIR / "leaf_metrics.json"

# ── Dataset Metadata ───────────────────────────────────────────────────────────
TARGET_COLUMN  = "CO(GT)"
DATASET_NAME   = "UCI Air Quality Dataset (id=360)"

# Feature columns in EXACT training order — do not reorder
FEATURE_COLUMNS: list[str] = [
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]
FEATURE_COUNT       = len(FEATURE_COLUMNS)
CATEGORICAL_COLUMNS: list[str] = []   # No categorical columns in this dataset

# ── Model Hyperparameters ──────────────────────────────────────────────────────
DT_MAX_DEPTH         = 3
DT_MIN_SAMPLES_LEAF  = 1000
DT_RANDOM_STATE      = 42
MIN_SAMPLES_PER_LEAF = 50

# ── API Metadata ───────────────────────────────────────────────────────────────
MODEL_VERSION   = "1.0"
API_VERSION     = "1.0.0"
API_TITLE       = "Hybrid DT+NN Air Quality API"
API_DESCRIPTION = """
## Hybrid Decision Tree + Neural Network Regression Model

Predicts **CO concentration (mg/m³)** from air quality sensor readings.

### Dataset
UCI Air Quality Dataset — Hourly averaged sensor readings from an Italian city.

### Model Architecture
- **Stage 1 — Decision Tree** (max_depth=3): Routes input to a leaf node.
- **Stage 2 — Leaf Neural Network**: If the leaf has a trained NN (≥50 training samples),
  the NN refines the prediction. Otherwise, the DT prediction is used as fallback.

### Performance
| Model | Test R² | Test Adj. R² |
|---|---|---|
| Decision Tree only | 0.5636 | 0.5608 |
| **Hybrid DT + NN** | **0.8790** | **0.8782** |
"""
