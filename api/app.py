"""
app.py — FastAPI application: routes, schemas, startup lifecycle.

Run with:
    uvicorn api.app:app --reload

Swagger UI: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.config import API_TITLE, API_DESCRIPTION, API_VERSION, FEATURE_COLUMNS
from src.predict import HybridPredictor
from src.utils import setup_logger

logger = setup_logger(__name__)

# ── Global predictor instance (loaded once at startup) ────────────────────────
predictor: HybridPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, release at shutdown."""
    global predictor
    logger.info("Starting up — loading model artifacts...")
    try:
        predictor = HybridPredictor()
        logger.info("✅ Models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise
    yield
    logger.info("Shutting down API")


# ── App Initialisation ────────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ─────────────────────────────────────────────────

class AirQualityInput(BaseModel):
    """
    Raw sensor readings from the UCI Air Quality dataset.
    Send real-world values — the API handles all scaling internally.
    """
    pt08_s1_co:          float = Field(..., description="PT08.S1(CO) sensor response (nominally CO targeted)", example=1046.0)
    nmhc_gt:             float = Field(..., description="NMHC(GT) — True hourly averaged Non Metanic HydroCarbons", example=166.0)
    c6h6_gt:             float = Field(..., description="C6H6(GT) — True hourly averaged Benzene concentration (μg/m³)", example=11.9)
    pt08_s2_nmhc:        float = Field(..., description="PT08.S2(NMHC) sensor response (nominally NMHC targeted)", example=1056.0)
    nox_gt:              float = Field(..., description="NOx(GT) — True hourly averaged NOx concentration (ppb)", example=166.0)
    pt08_s3_nox:         float = Field(..., description="PT08.S3(NOx) sensor response (nominally NOx targeted)", example=1056.0)
    no2_gt:              float = Field(..., description="NO2(GT) — True hourly averaged NO2 concentration (μg/m³)", example=113.0)
    pt08_s4_no2:         float = Field(..., description="PT08.S4(NO2) sensor response (nominally NO2 targeted)", example=1692.0)
    pt08_s5_o3:          float = Field(..., description="PT08.S5(O3) sensor response (nominally O3 and NO2 targeted)", example=1268.0)
    temperature:         float = Field(..., description="T — Temperature in Celsius", example=13.6)
    relative_humidity:   float = Field(..., description="RH — Relative Humidity (%)", example=48.9)
    absolute_humidity:   float = Field(..., description="AH — Absolute Humidity", example=0.7578)

    model_config = {"json_schema_extra": {
        "example": {
            "pt08_s1_co": 1046.0, "nmhc_gt": 166.0, "c6h6_gt": 11.9,
            "pt08_s2_nmhc": 1056.0, "nox_gt": 166.0, "pt08_s3_nox": 1056.0,
            "no2_gt": 113.0, "pt08_s4_no2": 1692.0, "pt08_s5_o3": 1268.0,
            "temperature": 13.6, "relative_humidity": 48.9, "absolute_humidity": 0.7578,
        }
    }}


class RawFeaturesInput(BaseModel):
    """
    Pre-scaled feature array for advanced users who have already applied StandardScaler.
    Must be exactly 12 float values in training column order.
    """
    features: list[float] = Field(
        ...,
        description=(
            f"Pre-scaled feature array — exactly {len(FEATURE_COLUMNS)} values "
            f"in this order: {FEATURE_COLUMNS}"
        ),
        min_length=12,
        max_length=12,
    )


class PredictionResponse(BaseModel):
    success:                bool
    decision_tree_output:   dict[str, Any]
    neural_network_output:  dict[str, Any]
    final_prediction:       dict[str, Any]
    model_version:          str


class HealthResponse(BaseModel):
    status:          str
    model_version:   str
    n_leaf_models:   int
    leaf_ids:        list[int]
    dt_leaves:       int
    dt_depth:        int
    feature_count:   int
    target:          str
    hybrid_test_r2:  float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"], summary="Welcome")
async def root() -> dict[str, str]:
    return {
        "message": "Hybrid DT+NN Air Quality API",
        "docs":    "http://127.0.0.1:8000/docs",
        "version": API_VERSION,
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["General"],
    summary="Model health and status",
)
async def health() -> dict[str, Any]:
    """Returns model load status, version, and performance metrics."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    return predictor.health()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Full pipeline — raw sensor values → CO prediction",
)
async def predict(input_data: AirQualityInput) -> dict[str, Any]:
    """
    Send raw air quality sensor readings.
    The API handles all preprocessing (scaling) internally.

    Returns structured prediction including:
    - DT leaf routing info
    - NN prediction details
    - Final CO concentration (mg/m³)
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        raw = input_data.model_dump()
        result = predictor.predict(raw)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post(
    "/predict/raw",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Raw inference — pre-scaled 12-element array → CO prediction",
)
async def predict_raw(input_data: RawFeaturesInput) -> dict[str, Any]:
    """
    For advanced users who have already applied StandardScaler externally.

    Send a 12-element array of pre-scaled values in exact training feature order:
    PT08.S1(CO), NMHC(GT), C6H6(GT), PT08.S2(NMHC), NOx(GT), PT08.S3(NOx),
    NO2(GT), PT08.S4(NO2), PT08.S5(O3), T, RH, AH

    The API skips preprocessing and runs the model directly.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        result = predictor.predict_scaled(input_data.features)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Raw prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
