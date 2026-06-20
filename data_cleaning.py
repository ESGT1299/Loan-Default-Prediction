"""Create a cleaned modeling dataset from the raw LendingClub CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.loan_default_risk.data import load_originated_loans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LendingClub origination features.")
    parser.add_argument("--data", required=True, help="Path to the raw LendingClub CSV.")
    parser.add_argument("--output", default="dataset/cleaned_originated_loans.csv")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional row limit for quick tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_originated_loans(args.data, sample_size=args.sample_size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Saved cleaned origination dataset to {output_path}")


if __name__ == "__main__":
    main()
