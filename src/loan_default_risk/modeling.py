"""Model training, calibration, evaluation, and explainability."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import load_originated_loans, split_features_target
from .features import DATE_COLUMN, DEFAULT_LABEL, NON_DEFAULT_LABEL


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build a preprocessing and random forest pipeline."""
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=25,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def temporal_split(
    data: pd.DataFrame,
    train_fraction: float = 0.70,
    calibration_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split loans by complete issue months into train, calibration, and test sets."""
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("Train and calibration fractions must leave data for testing.")

    ordered = data.sort_values(DATE_COLUMN).reset_index(drop=True)
    monthly_counts = ordered.groupby(DATE_COLUMN).size().sort_index()
    cumulative_share = monthly_counts.cumsum() / monthly_counts.sum()

    calibration_start = cumulative_share[cumulative_share >= train_fraction].index[0]
    test_start = cumulative_share[
        cumulative_share >= train_fraction + calibration_fraction
    ].index[0]

    train = ordered[ordered[DATE_COLUMN] < calibration_start].copy()
    calibration = ordered[
        (ordered[DATE_COLUMN] >= calibration_start)
        & (ordered[DATE_COLUMN] < test_start)
    ].copy()
    test = ordered[ordered[DATE_COLUMN] >= test_start].copy()

    if train.empty or calibration.empty or test.empty:
        raise ValueError("Temporal split produced an empty dataset partition.")

    return train, calibration, test


def get_default_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    """Return probabilities for the explicit default class."""
    classes = list(model.classes_)
    default_index = classes.index(DEFAULT_LABEL)
    return model.predict_proba(X)[:, default_index]


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Select a threshold that maximizes default-class F1 on calibration data."""
    y_default = (y_true == DEFAULT_LABEL).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_default, probabilities)
    if not len(thresholds):
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1_values))])


def _probability_scorer(estimator, X: pd.DataFrame, y: pd.Series) -> float:
    """ROC-AUC scorer with default explicitly treated as the positive class."""
    y_default = (y == DEFAULT_LABEL).astype(int)
    return roc_auc_score(y_default, get_default_probabilities(estimator, X))


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate discrimination, calibration, threshold behavior, and score distribution."""
    probabilities = get_default_probabilities(model, X_test)
    y_default = (y_test == DEFAULT_LABEL).astype(int)
    predictions = np.where(probabilities >= threshold, DEFAULT_LABEL, NON_DEFAULT_LABEL)

    fraction_positive, mean_predicted = calibration_curve(
        y_default,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )
    report = classification_report(
        y_test,
        predictions,
        labels=[DEFAULT_LABEL, NON_DEFAULT_LABEL],
        output_dict=True,
        zero_division=0,
    )
    score_frame = pd.DataFrame(
        {
            "actual": y_test.reset_index(drop=True),
            "default_probability": probabilities,
        }
    )

    metrics = {
        "roc_auc": roc_auc_score(y_default, probabilities),
        "pr_auc": average_precision_score(y_default, probabilities),
        "brier_score": brier_score_loss(y_default, probabilities),
        "threshold": threshold,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=[DEFAULT_LABEL, NON_DEFAULT_LABEL],
        ).tolist(),
        "class_order": [DEFAULT_LABEL, NON_DEFAULT_LABEL],
        "calibration_curve": {
            "mean_predicted_probability": mean_predicted.tolist(),
            "observed_default_rate": fraction_positive.tolist(),
        },
    }
    return metrics, score_frame


def explain_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> list[dict]:
    """Calculate original-feature permutation importance on a test subset."""
    sample_size = min(2_000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    y_sample = y_test.loc[X_sample.index]
    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring=_probability_scorer,
        n_repeats=3,
        random_state=42,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    ).sort_values("importance", ascending=False)
    return importance.head(15).to_dict(orient="records")


def _date_range(frame: pd.DataFrame) -> dict:
    return {
        "start": frame[DATE_COLUMN].min().strftime("%Y-%m"),
        "end": frame[DATE_COLUMN].max().strftime("%Y-%m"),
        "rows": len(frame),
    }


def train(
    csv_path: str | Path,
    model_path: str | Path = "artifacts/loan_default_pipeline.joblib",
    sample_size: int | None = None,
) -> dict:
    """Train chronologically, calibrate probabilities, evaluate, and persist artifacts."""
    data = load_originated_loans(csv_path, sample_size=sample_size)
    train_data, calibration_data, test_data = temporal_split(data)

    X_train, y_train = split_features_target(train_data)
    X_calibration, y_calibration = split_features_target(calibration_data)
    X_test, y_test = split_features_target(test_data)

    base_model = build_pipeline(X_train)
    base_model.fit(X_train, y_train)

    calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
    calibrated_model.fit(X_calibration, y_calibration)

    calibration_probabilities = get_default_probabilities(calibrated_model, X_calibration)
    threshold = select_threshold(y_calibration, calibration_probabilities)
    metrics, score_frame = evaluate_model(calibrated_model, X_test, y_test, threshold)
    feature_importance = explain_model(calibrated_model, X_test, y_test)

    metrics["split_summary"] = {
        "train": _date_range(train_data),
        "calibration": _date_range(calibration_data),
        "test": _date_range(test_data),
        "default_rate": float((data["loan_status"] == DEFAULT_LABEL).mean()),
        "total_rows": len(data),
    }

    artifact = {
        "model": calibrated_model,
        "features": X_train.columns.tolist(),
        "metrics": metrics,
        "feature_importance": feature_importance,
        "score_sample": score_frame.sample(min(5_000, len(score_frame)), random_state=42),
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    return metrics
