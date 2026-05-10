"""
preprocess.py — Preprocessing pipeline for inference.

Transforms raw user input into the scaled numpy array expected by the model.
No fitting happens here — only transforming using saved artifacts.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from src.config import FEATURE_COLUMNS, FEATURE_COUNT
from src.utils import setup_logger

logger = setup_logger(__name__)

# Maps clean API field names → exact dataset column names (training order)
FIELD_TO_COLUMN: dict[str, str] = {
    "pt08_s1_co":          "PT08.S1(CO)",
    "nmhc_gt":             "NMHC(GT)",
    "c6h6_gt":             "C6H6(GT)",
    "pt08_s2_nmhc":        "PT08.S2(NMHC)",
    "nox_gt":              "NOx(GT)",
    "pt08_s3_nox":         "PT08.S3(NOx)",
    "no2_gt":              "NO2(GT)",
    "pt08_s4_no2":         "PT08.S4(NO2)",
    "pt08_s5_o3":          "PT08.S5(O3)",
    "temperature":         "T",
    "relative_humidity":   "RH",
    "absolute_humidity":   "AH",
}


def input_dict_to_array(raw_input: dict[str, float]) -> np.ndarray:
    """
    Convert a raw API input dict to a (1, 12) numpy array in training feature order.

    Args:
        raw_input: Dict with clean field names (e.g. "pt08_s1_co", "temperature").

    Returns:
        numpy array of shape (1, 12), unscaled.

    Raises:
        ValueError: If any required field is missing.
    """
    missing = [k for k in FIELD_TO_COLUMN if k not in raw_input]
    if missing:
        raise ValueError(f"Missing input fields: {missing}")

    # Build feature vector in exact training column order
    values = []
    for col in FEATURE_COLUMNS:
        # Reverse-lookup: find which API field maps to this column
        api_key = next(k for k, v in FIELD_TO_COLUMN.items() if v == col)
        values.append(float(raw_input[api_key]))

    return np.array(values, dtype=np.float64).reshape(1, -1)


def scale_input(X_raw: np.ndarray, scaler: Any) -> np.ndarray:
    """
    Apply the saved StandardScaler to a raw input array.

    Args:
        X_raw:  numpy array of shape (1, 12), unscaled.
        scaler: Loaded sklearn StandardScaler (fit on training data).

    Returns:
        Scaled numpy array of shape (1, 12).
    """
    return scaler.transform(X_raw)


def preprocess_input(raw_input: dict[str, float], scaler: Any) -> np.ndarray:
    """
    Full preprocessing pipeline: raw dict → scaled array.

    Args:
        raw_input: Dict of API input fields.
        scaler:    Loaded StandardScaler artifact.

    Returns:
        Scaled numpy array (1, 12) ready for model inference.
    """
    X_raw    = input_dict_to_array(raw_input)
    X_scaled = scale_input(X_raw, scaler)
    logger.debug(f"Preprocessed input — raw: {X_raw.flatten()[:3]}... scaled: {X_scaled.flatten()[:3]}...")
    return X_scaled


def validate_raw_array(features: list[float]) -> np.ndarray:
    """
    Validate and reshape a pre-scaled raw feature array for /predict/raw endpoint.

    Args:
        features: List of 12 already-scaled float values.

    Returns:
        numpy array of shape (1, 12).

    Raises:
        ValueError: If feature count doesn't match.
    """
    if len(features) != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} features, got {len(features)}. "
            f"Feature order: {FEATURE_COLUMNS}"
        )
    return np.array(features, dtype=np.float64).reshape(1, -1)
