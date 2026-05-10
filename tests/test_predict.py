"""
test_predict.py — Unit tests for the inference pipeline.

Run with:
    python -m pytest tests/ -v
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import input_dict_to_array, validate_raw_array, FIELD_TO_COLUMN
from src.config import FEATURE_COUNT, FEATURE_COLUMNS


# ── Sample Data ───────────────────────────────────────────────────────────────

SAMPLE_INPUT = {
    "pt08_s1_co":        1046.0,
    "nmhc_gt":           166.0,
    "c6h6_gt":           11.9,
    "pt08_s2_nmhc":      1056.0,
    "nox_gt":            166.0,
    "pt08_s3_nox":       1056.0,
    "no2_gt":            113.0,
    "pt08_s4_no2":       1692.0,
    "pt08_s5_o3":        1268.0,
    "temperature":       13.6,
    "relative_humidity": 48.9,
    "absolute_humidity": 0.7578,
}

SAMPLE_SCALED = [0.45, -1.2, 0.88, 0.33, -0.71, 1.04, 0.22, -0.5, 0.0, 1.0, 0.0, 1.0]


# ── Preprocessing Tests ───────────────────────────────────────────────────────

def test_input_dict_to_array_shape():
    arr = input_dict_to_array(SAMPLE_INPUT)
    assert arr.shape == (1, FEATURE_COUNT), f"Expected (1, {FEATURE_COUNT}), got {arr.shape}"


def test_input_dict_to_array_dtype():
    arr = input_dict_to_array(SAMPLE_INPUT)
    assert arr.dtype == np.float64


def test_input_dict_to_array_values():
    arr = input_dict_to_array(SAMPLE_INPUT)
    # PT08.S1(CO) is first column — should equal sample value
    assert arr[0, 0] == pytest.approx(1046.0)
    # Temperature (T) is 10th column (index 9)
    assert arr[0, 9] == pytest.approx(13.6)


def test_input_dict_missing_field():
    bad_input = {k: v for k, v in SAMPLE_INPUT.items() if k != "temperature"}
    with pytest.raises(ValueError, match="Missing input fields"):
        input_dict_to_array(bad_input)


def test_validate_raw_array_correct():
    arr = validate_raw_array(SAMPLE_SCALED)
    assert arr.shape == (1, FEATURE_COUNT)


def test_validate_raw_array_wrong_length():
    with pytest.raises(ValueError, match="Expected 12 features"):
        validate_raw_array([1.0, 2.0, 3.0])  # too short


def test_feature_column_count():
    assert len(FEATURE_COLUMNS) == 12


def test_field_to_column_mapping_complete():
    # Every feature column must have a corresponding API field
    mapped_columns = set(FIELD_TO_COLUMN.values())
    for col in FEATURE_COLUMNS:
        assert col in mapped_columns, f"Column '{col}' has no API field mapping"


# ── Integration Test (requires models/ artifacts) ─────────────────────────────

@pytest.mark.integration
def test_full_prediction():
    """Requires trained artifacts in models/. Skip if not available."""
    try:
        from src.predict import HybridPredictor
        predictor = HybridPredictor()
    except Exception:
        pytest.skip("Model artifacts not available")

    result = predictor.predict(SAMPLE_INPUT)
    assert result["success"] is True
    assert "final_prediction" in result
    assert "co_concentration_mg_m3" in result["final_prediction"]
    assert isinstance(result["final_prediction"]["co_concentration_mg_m3"], float)


@pytest.mark.integration
def test_raw_prediction():
    """Requires trained artifacts in models/. Skip if not available."""
    try:
        from src.predict import HybridPredictor
        predictor = HybridPredictor()
    except Exception:
        pytest.skip("Model artifacts not available")

    result = predictor.predict_scaled(SAMPLE_SCALED)
    assert result["success"] is True
    assert result["model_version"] == "1.0"
