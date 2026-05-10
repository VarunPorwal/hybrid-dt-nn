"""
model.py — Neural Network architecture builder and artifact loaders.

The build_model() function exactly replicates the architecture used during
training in Colab. This ensures loaded .h5 weights are fully compatible.
"""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any

import numpy as np

from src.utils import setup_logger, validate_artifact_exists

logger = setup_logger(__name__)


# ── NN Architecture Builder ────────────────────────────────────────────────────

def build_model(
    optimizer: str = "adam",
    activation: str = "relu",
    hidden_layers: int = 1,
    units: int = 64,
) -> Any:
    """
    Build a Keras Sequential regression model.

    This is an EXACT replica of the training-time build_model() from Colab.
    Architecture must match saved .h5 weights precisely.

    Args:
        optimizer:     One of adam | adamw | rmsprop | nadam
        activation:    One of relu | leakyrelu | elu | tanh | swish
        hidden_layers: Number of hidden dense layers (1–5)
        units:         Neurons per hidden layer (8 | 16 | 32 | 64 | 128)

    Returns:
        Compiled keras.Sequential model.
    """
    # Lazy import to avoid TF loading at module import time
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential()
    for _ in range(hidden_layers):
        if activation == "leakyrelu":
            model.add(layers.Dense(units))
            model.add(layers.LeakyReLU(negative_slope=0.1))
        else:
            model.add(layers.Dense(units, activation=activation))
    model.add(layers.Dense(1))  # single regression output
    model.compile(optimizer=optimizer, loss="mse")
    return model


# ── Artifact Loaders ───────────────────────────────────────────────────────────

def load_dt_model(path: Path) -> Any:
    """Load the sklearn DecisionTreeRegressor from a .pkl file."""
    validate_artifact_exists(path, "Decision Tree")
    dt = joblib.load(path)
    logger.info(f"Decision Tree loaded — leaves: {dt.get_n_leaves()}, depth: {dt.get_depth()}")
    return dt


def load_scaler(path: Path) -> Any:
    """Load the sklearn StandardScaler from a .pkl file."""
    validate_artifact_exists(path, "StandardScaler")
    scaler = joblib.load(path)
    logger.info(f"Scaler loaded — n_features: {scaler.n_features_in_}")
    return scaler


def load_leaf_models(leaf_models_dir: Path) -> dict[int, Any]:
    """
    Load all Keras leaf NN models from a directory.

    Expects files named by leaf ID: 3.h5, 4.h5, 6.h5, ...

    Returns:
        Dict mapping leaf_id (int) -> loaded keras.Model
    """
    import tensorflow as tf

    h5_files = sorted(leaf_models_dir.glob("*.h5"))
    if not h5_files:
        raise RuntimeError(f"No .h5 files found in {leaf_models_dir}")

    leaf_models: dict[int, Any] = {}
    for h5_path in h5_files:
        leaf_id = int(h5_path.stem)
        leaf_models[leaf_id] = tf.keras.models.load_model(str(h5_path), compile=False)
        logger.debug(f"  Leaf {leaf_id} NN loaded from {h5_path.name}")

    logger.info(f"Leaf NNs loaded — count: {len(leaf_models)}, ids: {sorted(leaf_models)}")
    return leaf_models
