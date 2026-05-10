"""
predict.py — Core inference pipeline: HybridPredictor class.

Loads all artifacts once at startup, then serves predictions efficiently.
Hybrid logic: DT routes to leaf → NN predicts (or DT fallback).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    DT_MODEL_PATH, SCALER_PATH, LEAF_MODELS_DIR,
    CONFIG_PATH, LEAF_METRICS_PATH, MODEL_VERSION,
)
from src.model import load_dt_model, load_scaler, load_leaf_models
from src.preprocess import preprocess_input, validate_raw_array
from src.utils import setup_logger, load_json, check_all_artifacts

logger = setup_logger(__name__)


class HybridPredictor:
    """
    Production inference engine for the Hybrid DT+NN model.

    Usage:
        predictor = HybridPredictor()
        result = predictor.predict(raw_input_dict)
    """

    def __init__(self) -> None:
        logger.info("Initialising HybridPredictor — loading all artifacts...")

        # Validate all files exist before loading
        check_all_artifacts(DT_MODEL_PATH, SCALER_PATH, LEAF_MODELS_DIR, CONFIG_PATH)

        # Load artifacts
        self.dt           = load_dt_model(DT_MODEL_PATH)
        self.scaler       = load_scaler(SCALER_PATH)
        self.leaf_models  = load_leaf_models(LEAF_MODELS_DIR)
        self.config       = load_json(CONFIG_PATH)
        self.leaf_metrics = load_json(LEAF_METRICS_PATH) if LEAF_METRICS_PATH.exists() else {}

        logger.info(
            f"HybridPredictor ready — "
            f"{len(self.leaf_models)} leaf NNs | "
            f"version {MODEL_VERSION}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, raw_input: dict[str, float]) -> dict[str, Any]:
        """
        Full pipeline: raw field dict → preprocess → DT → NN → result.

        Args:
            raw_input: Dict with clean API field names and raw (unscaled) values.

        Returns:
            Structured prediction dict.
        """
        X_scaled = preprocess_input(raw_input, self.scaler)
        return self._run_hybrid(X_scaled)

    def predict_scaled(self, features: list[float]) -> dict[str, Any]:
        """
        Raw inference endpoint: pre-scaled feature array → DT → NN → result.

        Args:
            features: List of 12 already-scaled float values.

        Returns:
            Structured prediction dict.
        """
        X_scaled = validate_raw_array(features)
        return self._run_hybrid(X_scaled)

    def health(self) -> dict[str, Any]:
        """Return model status and metadata for /health endpoint."""
        return {
            "status":        "ok",
            "model_version": MODEL_VERSION,
            "n_leaf_models": len(self.leaf_models),
            "leaf_ids":      sorted(self.leaf_models.keys()),
            "dt_leaves":     self.dt.get_n_leaves(),
            "dt_depth":      self.dt.get_depth(),
            "feature_count": self.config.get("feature_count"),
            "target":        self.config.get("target_column"),
            "hybrid_test_r2": self.config.get("hybrid_test_r2"),
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run_hybrid(self, X_scaled: np.ndarray) -> dict[str, Any]:
        """Core hybrid routing logic: DT leaf → NN or fallback."""
        # Stage 1: Decision Tree
        leaf_id      = int(self.dt.apply(X_scaled)[0])
        dt_prediction = float(self.dt.predict(X_scaled)[0])

        # Stage 2: Leaf NN (if available)
        used_nn = leaf_id in self.leaf_models
        if used_nn:
            nn_model    = self.leaf_models[leaf_id]
            final_value = float(nn_model.predict(X_scaled, verbose=0)[0, 0])
        else:
            final_value = dt_prediction

        # Leaf metrics (if available)
        leaf_meta = self.leaf_metrics.get(str(leaf_id), {})

        return {
            "success": True,
            "decision_tree_output": {
                "leaf_id":    leaf_id,
                "prediction": round(dt_prediction, 4),
            },
            "neural_network_output": {
                "used_nn":    used_nn,
                "prediction": round(final_value, 4) if used_nn else None,
                "leaf_r2":    leaf_meta.get("r2"),
                "leaf_samples": leaf_meta.get("num_samples"),
            },
            "final_prediction": {
                "co_concentration_mg_m3": round(final_value, 4),
                "source": "neural_network" if used_nn else "decision_tree_fallback",
            },
            "model_version": MODEL_VERSION,
        }
