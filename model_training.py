"""Train the loan default model.

Usage:
    python model_training.py --data dataset/accepted_2007_to_2018Q4.csv
"""

from __future__ import annotations

import argparse
import json

from src.loan_default_risk.modeling import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a loan default risk model.")
    parser.add_argument("--data", required=True, help="Path to the LendingClub CSV file.")
    parser.add_argument(
        "--model-output",
        default="artifacts/loan_default_pipeline.joblib",
        help="Where to save the trained pipeline artifact.",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="Optional row limit for quick tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train(args.data, args.model_output, sample_size=args.sample_size)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
