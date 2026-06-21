"""Generate a lightweight exploratory report from LendingClub data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loan_default_risk.data import load_originated_loans
from src.loan_default_risk.features import DEFAULT_LABEL, TARGET_COLUMN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore the LendingClub modeling dataset.")
    parser.add_argument("--data", required=True, help="CSV file or folder containing accepted loans.")
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--output-dir", default="artifacts/eda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_originated_loans(args.data, sample_size=args.sample_size)
    default_rate = (data[TARGET_COLUMN] == DEFAULT_LABEL).mean()
    summary = pd.DataFrame(
        {
            "metric": ["rows", "columns", "default_rate", "start_date", "end_date"],
            "value": [
                len(data),
                len(data.columns),
                default_rate,
                data["issue_d"].min().strftime("%Y-%m"),
                data["issue_d"].max().strftime("%Y-%m"),
            ],
        }
    )
    summary.to_csv(output_dir / "dataset_summary.csv", index=False)

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(data=data, x="loan_amnt", bins=35, ax=axes[0], color="#2563eb")
    axes[0].set(title="Loan Amount Distribution", xlabel="Loan Amount")

    outcome_counts = data[TARGET_COLUMN].value_counts().rename_axis("Outcome").reset_index(name="Loans")
    sns.barplot(data=outcome_counts, x="Outcome", y="Loans", ax=axes[1], color="#0f766e")
    axes[1].set(title="Resolved Loan Outcomes", xlabel="")

    figure.tight_layout()
    figure.savefig(output_dir / "portfolio_overview.png", dpi=180)
    plt.close(figure)
    print(summary.to_string(index=False))
    print(f"Saved EDA outputs to {output_dir}")


if __name__ == "__main__":
    main()
