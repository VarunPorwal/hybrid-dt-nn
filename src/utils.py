"""
utils.py — Shared utility functions: logging, JSON I/O, validation helpers.
"""

import json
import logging
from pathlib import Path
from typing import Any


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a structured console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler  = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and return as dict. Raises FileNotFoundError with clear message."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Make sure you have placed the trained artifacts in the models/ directory."
        )
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save a dict as a formatted JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def validate_artifact_exists(path: Path, label: str) -> None:
    """Raise a clear RuntimeError if a required artifact file is missing."""
    if not path.exists():
        raise RuntimeError(
            f"Missing artifact [{label}]: {path}\n"
            "Run the training script in Colab and place the models/ folder here."
        )


def check_all_artifacts(
    dt_path: Path,
    scaler_path: Path,
    leaf_models_dir: Path,
    config_path: Path,
) -> None:
    """Validate that all required model artifacts are present before loading."""
    validate_artifact_exists(dt_path,     "Decision Tree model")
    validate_artifact_exists(scaler_path, "StandardScaler")
    validate_artifact_exists(config_path, "config.json")
    if not leaf_models_dir.exists() or not any(leaf_models_dir.glob("*.h5")):
        raise RuntimeError(
            f"No leaf NN models found in: {leaf_models_dir}\n"
            "Expected files like: 3.h5, 4.h5, 6.h5, ..."
        )
