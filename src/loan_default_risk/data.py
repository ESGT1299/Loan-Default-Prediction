"""Data loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import (
    DEFAULT_LABEL,
    DEFAULT_STATUSES,
    NON_DEFAULT_LABEL,
    NON_DEFAULT_STATUSES,
    ORIGINATION_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
    validate_feature_policy,
)


def normalize_percent(value):
    """Convert values like '13.56%' to floats while preserving numeric inputs."""
    if pd.isna(value):
        return value
    if isinstance(value, str):
        return float(value.strip().replace("%", ""))
    return value


def make_binary_target(status: str) -> str | None:
    """Map resolved loan statuses to a binary risk target."""
    if status in DEFAULT_STATUSES:
        return DEFAULT_LABEL
    if status in NON_DEFAULT_STATUSES:
        return NON_DEFAULT_LABEL
    return None


def resolve_accepted_loans_path(path: str | Path) -> Path:
    """Resolve either the raw accepted loans CSV file or a directory containing it."""
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        matches = sorted(candidate for candidate in path.rglob("accepted_2007_to_2018Q4.csv") if candidate.is_file())
        if not matches:
            matches = sorted(candidate for candidate in path.rglob("accepted_2007_to_2018q4.csv") if candidate.is_file())
        if matches:
            return matches[0]
    raise FileNotFoundError(
        "Accepted loans CSV not found. Pass either the CSV file or the folder that contains it."
    )


def load_originated_loans(csv_path: str | Path, sample_size: int | None = None) -> pd.DataFrame:
    """Load LendingClub data and keep only origination-time features."""
    csv_path = resolve_accepted_loans_path(csv_path)

    usecols = list(dict.fromkeys(ORIGINATION_FEATURES + [TARGET_COLUMN]))
    read_options = {
        "usecols": lambda col: col in usecols,
        "low_memory": False,
    }
    if sample_size:
        random_generator = np.random.default_rng(42)
        sampled_chunks = []
        for chunk in pd.read_csv(csv_path, chunksize=100_000, **read_options):
            chunk["_sample_key"] = random_generator.random(len(chunk))
            sampled_chunks.append(chunk)
            combined = pd.concat(sampled_chunks, ignore_index=True)
            sampled_chunks = [combined.nsmallest(sample_size, "_sample_key")]
        data = sampled_chunks[0].drop(columns="_sample_key")
    else:
        data = pd.read_csv(csv_path, **read_options)
    missing = sorted(set(usecols) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data[TARGET_COLUMN] = data[TARGET_COLUMN].map(make_binary_target)
    data = data.dropna(subset=[TARGET_COLUMN]).copy()
    data["issue_d"] = pd.to_datetime(data["issue_d"], format="%b-%Y", errors="coerce")
    data = data.dropna(subset=["issue_d"])

    for column in ["int_rate", "revol_util"]:
        if column in data.columns:
            data[column] = data[column].map(normalize_percent)

    validate_feature_policy([col for col in data.columns if col != TARGET_COLUMN])
    return data


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and target."""
    features = [col for col in MODEL_FEATURES if col in data.columns]
    validate_feature_policy(features)
    return data[features], data[TARGET_COLUMN]
