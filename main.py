"""
main.py — CLI entry point for quick local inference testing.

Usage:
    python main.py
    python main.py --scaled  (use pre-scaled values)
"""

import argparse
import json
import sys

from src.predict import HybridPredictor
from src.utils import setup_logger

logger = setup_logger("main")

# Default sample input — real values from the UCI Air Quality dataset
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

# Sample pre-scaled values (for --scaled mode)
SAMPLE_SCALED = [
    0.45, -1.2, 0.88, 0.33, -0.71, 1.04,
    0.22, -0.5, 0.0, 1.0, 0.0, 1.0,
]


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid DT+NN — Local Inference Test"
    )
    parser.add_argument(
        "--scaled", action="store_true",
        help="Use pre-scaled raw array instead of raw field values"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="JSON string of input fields (overrides default sample)"
    )
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  Hybrid DT+NN — Local Inference")
    print("="*55)

    try:
        predictor = HybridPredictor()
    except Exception as e:
        print(f"\n❌ Failed to load models: {e}")
        sys.exit(1)

    if args.scaled:
        print("\nMode: /predict/raw (pre-scaled features)")
        print(f"Input: {SAMPLE_SCALED}")
        result = predictor.predict_scaled(SAMPLE_SCALED)
    else:
        raw = SAMPLE_INPUT
        if args.input:
            try:
                raw = json.loads(args.input)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON input: {e}")
                sys.exit(1)
        print("\nMode: /predict (raw field values)")
        print(f"Input: {json.dumps(raw, indent=2)}")
        result = predictor.predict(raw)

    print("\n── Result ──────────────────────────────────────")
    print(json.dumps(result, indent=2))
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
