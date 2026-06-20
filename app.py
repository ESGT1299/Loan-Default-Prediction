"""Streamlit app for loan default risk scoring."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.loan_default_risk.features import DEFAULT_LABEL
from src.loan_default_risk.modeling import get_default_probabilities

ARTIFACT_PATH = Path("artifacts/loan_default_pipeline.joblib")

FIELD_LABELS = {
    "loan_amnt": "Loan Amount",
    "term": "Loan Term",
    "int_rate": "Interest Rate (%)",
    "installment": "Monthly Installment",
    "grade": "Credit Grade",
    "sub_grade": "Credit Subgrade",
    "emp_length": "Employment Length",
    "home_ownership": "Home Ownership",
    "annual_inc": "Annual Income",
    "verification_status": "Income Verification",
    "issue_d": "Loan Issue Date",
    "purpose": "Loan Purpose",
    "dti": "Debt-to-Income Ratio (%)",
    "delinq_2yrs": "Delinquencies in Last 2 Years",
    "earliest_cr_line": "Earliest Credit Line",
    "fico_range_low": "FICO Score Low",
    "fico_range_high": "FICO Score High",
    "inq_last_6mths": "Recent Credit Inquiries",
    "open_acc": "Open Credit Accounts",
    "pub_rec": "Public Records",
    "revol_bal": "Revolving Balance",
    "revol_util": "Revolving Utilization (%)",
    "total_acc": "Total Credit Accounts",
    "initial_list_status": "Initial Listing Status",
    "application_type": "Application Type",
    "mort_acc": "Mortgage Accounts",
    "pub_rec_bankruptcies": "Public Record Bankruptcies",
}

HELP_TEXT = {
    "loan_amnt": "Requested loan principal.",
    "term": "Length of the loan contract.",
    "int_rate": "Annual interest rate at origination.",
    "installment": "Expected monthly payment.",
    "grade": "LendingClub credit grade from A to G.",
    "sub_grade": "More granular grade, such as B3 or C2.",
    "emp_length": "Borrower's reported employment history.",
    "home_ownership": "Borrower's housing status.",
    "annual_inc": "Borrower's stated annual income.",
    "verification_status": "Whether income was verified by the platform.",
    "issue_d": "Approximate month the loan was issued.",
    "purpose": "Main reason for the loan.",
    "dti": "Monthly debt payments divided by monthly income.",
    "delinq_2yrs": "Number of recent credit delinquencies.",
    "earliest_cr_line": "Oldest credit account month.",
    "fico_range_low": "Lower bound of the FICO score range.",
    "fico_range_high": "Upper bound of the FICO score range.",
    "inq_last_6mths": "Credit pulls in the last six months.",
    "open_acc": "Currently open credit lines.",
    "pub_rec": "Derogatory public records.",
    "revol_bal": "Outstanding revolving credit balance.",
    "revol_util": "Percent of revolving credit currently used.",
    "total_acc": "Total number of credit accounts.",
    "initial_list_status": "Whether the loan was listed as whole or fractional.",
    "application_type": "Individual or joint loan application.",
    "mort_acc": "Number of mortgage accounts.",
    "pub_rec_bankruptcies": "Public bankruptcy records.",
}

PROFILES = {
    "Typical borrower": {
        "loan_amnt": 15000,
        "term": " 36 months",
        "int_rate": 13.5,
        "installment": 510,
        "grade": "C",
        "sub_grade": "C2",
        "emp_length": "10+ years",
        "home_ownership": "MORTGAGE",
        "annual_inc": 72000,
        "verification_status": "Source Verified",
        "issue_d": "Jan-2016",
        "purpose": "debt_consolidation",
        "dti": 18.5,
        "delinq_2yrs": 0,
        "earliest_cr_line": "Jan-2004",
        "fico_range_low": 680,
        "fico_range_high": 684,
        "inq_last_6mths": 1,
        "open_acc": 11,
        "pub_rec": 0,
        "revol_bal": 14000,
        "revol_util": 48.0,
        "total_acc": 26,
        "initial_list_status": "w",
        "application_type": "Individual",
        "mort_acc": 2,
        "pub_rec_bankruptcies": 0,
    },
    "Lower-risk example": {
        "loan_amnt": 10000,
        "term": " 36 months",
        "int_rate": 7.9,
        "installment": 313,
        "grade": "A",
        "sub_grade": "A4",
        "emp_length": "10+ years",
        "home_ownership": "MORTGAGE",
        "annual_inc": 115000,
        "verification_status": "Verified",
        "issue_d": "Jan-2016",
        "purpose": "home_improvement",
        "dti": 9.5,
        "delinq_2yrs": 0,
        "earliest_cr_line": "Jun-1998",
        "fico_range_low": 740,
        "fico_range_high": 744,
        "inq_last_6mths": 0,
        "open_acc": 9,
        "pub_rec": 0,
        "revol_bal": 6500,
        "revol_util": 22.0,
        "total_acc": 31,
        "initial_list_status": "w",
        "application_type": "Individual",
        "mort_acc": 3,
        "pub_rec_bankruptcies": 0,
    },
    "Higher-risk example": {
        "loan_amnt": 28000,
        "term": " 60 months",
        "int_rate": 22.0,
        "installment": 775,
        "grade": "E",
        "sub_grade": "E2",
        "emp_length": "< 1 year",
        "home_ownership": "RENT",
        "annual_inc": 52000,
        "verification_status": "Not Verified",
        "issue_d": "Jan-2016",
        "purpose": "debt_consolidation",
        "dti": 31.0,
        "delinq_2yrs": 2,
        "earliest_cr_line": "Sep-2011",
        "fico_range_low": 660,
        "fico_range_high": 664,
        "inq_last_6mths": 4,
        "open_acc": 18,
        "pub_rec": 1,
        "revol_bal": 31000,
        "revol_util": 84.0,
        "total_acc": 21,
        "initial_list_status": "f",
        "application_type": "Individual",
        "mort_acc": 0,
        "pub_rec_bankruptcies": 1,
    },
}

CATEGORICAL_OPTIONS = {
    "term": [" 36 months", " 60 months"],
    "grade": list("ABCDEFG"),
    "sub_grade": [
        f"{grade}{number}"
        for grade in "ABCDEFG"
        for number in range(1, 6)
    ],
    "emp_length": [
        "< 1 year",
        "1 year",
        "2 years",
        "3 years",
        "4 years",
        "5 years",
        "6 years",
        "7 years",
        "8 years",
        "9 years",
        "10+ years",
    ],
    "home_ownership": ["RENT", "MORTGAGE", "OWN", "OTHER"],
    "verification_status": ["Not Verified", "Source Verified", "Verified"],
    "purpose": [
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "major_purchase",
        "small_business",
        "medical",
        "car",
        "other",
    ],
    "initial_list_status": ["w", "f"],
    "application_type": ["Individual", "Joint App"],
}

OPTION_LABELS = {
    "debt_consolidation": "Debt Consolidation",
    "credit_card": "Credit Card Refinancing",
    "home_improvement": "Home Improvement",
    "major_purchase": "Major Purchase",
    "small_business": "Small Business",
    "medical": "Medical Expenses",
    "car": "Vehicle Purchase",
    "other": "Other",
    "w": "Whole Loan",
    "f": "Fractional Loan",
    "MORTGAGE": "Mortgage",
    "RENT": "Rent",
    "OWN": "Own",
    "OTHER": "Other",
}

BASIC_FIELDS = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "annual_inc",
    "home_ownership",
    "verification_status",
    "purpose",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "revol_util",
]

ADVANCED_FIELDS = [
    "emp_length",
    "issue_d",
    "earliest_cr_line",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "total_acc",
    "initial_list_status",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
]


def label_for(field: str) -> str:
    """Return a human-readable field label."""
    return FIELD_LABELS.get(field, field.replace("_", " ").title())


def risk_band(probability: float, threshold: float) -> tuple[str, str]:
    """Map default probability into a simple analyst-facing risk band."""
    if probability < threshold * 0.75:
        return "Lower estimated risk", "success"
    if probability < threshold * 1.25:
        return "Moderate estimated risk", "warning"
    return "Higher estimated risk", "error"


def select_index(options: list, value) -> int:
    """Return the selected index for Streamlit selectboxes."""
    return options.index(value) if value in options else 0


def render_input(field: str, defaults: dict) -> object:
    """Render one friendly input and return its value."""
    label = label_for(field)
    help_text = HELP_TEXT.get(field)
    default_value = defaults.get(field)

    if field in CATEGORICAL_OPTIONS:
        options = CATEGORICAL_OPTIONS[field]
        return st.selectbox(
            label,
            options=options,
            index=select_index(options, default_value),
            format_func=lambda value: OPTION_LABELS.get(value, value),
            help=help_text,
        )

    if field in {"issue_d", "earliest_cr_line"}:
        return st.text_input(label, value=str(default_value), help=help_text)

    if isinstance(default_value, int):
        return st.number_input(label, value=int(default_value), step=1, help=help_text)

    return st.number_input(label, value=float(default_value), step=0.1, help=help_text)


st.set_page_config(page_title="Loan Default Risk", layout="wide")
st.title("Loan Default Risk Scoring")
st.caption(
    "Professional portfolio project for credit-risk-style modeling. "
    "This score is not a real approval or denial decision."
)

if not ARTIFACT_PATH.exists():
    st.warning(
        "Model artifact not found. Train the model first with: "
        '`python model_training.py --data "datos de kaggle" --sample-size 50000`'
    )
    st.stop()

artifact = joblib.load(ARTIFACT_PATH)
model = artifact["model"]
features = artifact["features"]
metrics = artifact.get("metrics", {})
feature_importance = artifact.get("feature_importance", [])
score_sample = artifact.get("score_sample", pd.DataFrame())
decision_threshold = float(metrics.get("threshold", 0.5))

with st.sidebar:
    st.header("Model Snapshot")
    if metrics:
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        st.caption("Ranking quality: higher is better; 0.50 is close to random.")
        st.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")
        st.caption("Useful when defaults are less common than non-defaults.")
        st.metric("Brier Score", f"{metrics.get('brier_score', 0):.3f}")
        st.caption("Probability calibration error: lower is better.")
    st.divider()
    st.write("Use the profiles below as realistic starting points. You do not need to manually fill every field.")

scoring_page, performance_page = st.tabs(["Score a Loan", "Model Performance"])

with scoring_page:
    st.subheader("Borrower and Loan Profile")
    profile_name = st.selectbox("Start with an example profile", list(PROFILES.keys()))
    defaults = PROFILES[profile_name].copy()

    st.info(
        "You can score a loan immediately using the selected example profile. "
        "Adjust a few fields to see how the estimated risk changes."
    )

    basic_tab, advanced_tab, explanation_tab = st.tabs(
        ["Main Inputs", "Advanced Inputs", "How to Read the Score"]
    )

    values = defaults.copy()
    with basic_tab:
        basic_columns = st.columns(2)
        for index, field in enumerate(BASIC_FIELDS):
            if field in features:
                with basic_columns[index % 2]:
                    values[field] = render_input(field, values)

    with advanced_tab:
        st.caption("These fields come from the original LendingClub application data.")
        advanced_columns = st.columns(2)
        for index, field in enumerate(ADVANCED_FIELDS):
            if field in features:
                with advanced_columns[index % 2]:
                    values[field] = render_input(field, values)

    with explanation_tab:
        st.write(
            "The model estimates the probability that a loan ends in a default-like outcome, "
            "using information available around origination."
        )
        st.write(
            f"The operating threshold is {decision_threshold:.1%}. It was selected on a separate "
            "calibration period to balance precision and recall for defaults."
        )
        st.write(
            "The model was trained on earlier loans, calibrated on a later period, and tested on "
            "the newest loans. This better represents how a model would face future applications."
        )

    input_df = pd.DataFrame([{feature: values.get(feature) for feature in features}])

    if st.button("Score Loan", type="primary"):
        default_probability = float(get_default_probabilities(model, input_df)[0])
        band_label, band_style = risk_band(default_probability, decision_threshold)

        st.subheader("Risk Estimate")
        metric_column, band_column = st.columns([1, 2])
        with metric_column:
            st.metric("Estimated Default Probability", f"{default_probability:.1%}")
        with band_column:
            getattr(st, band_style)(band_label)

        st.progress(min(max(default_probability, 0.0), 1.0))
        if default_probability >= decision_threshold:
            st.warning(
                "This score is above the model's analysis threshold. A real workflow would send "
                "the case for additional review, not automatically reject it."
            )
        else:
            st.success(
                "This score is below the model's analysis threshold. It still requires the normal "
                "credit policy and verification process."
            )

with performance_page:
    st.subheader("Out-of-Time Model Evaluation")
    summary = metrics.get("split_summary", {})
    metric_columns = st.columns(4)
    metric_columns[0].metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    metric_columns[1].metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")
    metric_columns[2].metric("Brier Score", f"{metrics.get('brier_score', 0):.3f}")
    metric_columns[3].metric("Default Threshold", f"{decision_threshold:.1%}")

    st.caption(
        "These metrics come from the newest holdout period, which was not used to train or calibrate the model."
    )

    if summary:
        split_rows = []
        for split_name in ["train", "calibration", "test"]:
            split = summary.get(split_name, {})
            split_rows.append(
                {
                    "Dataset": split_name.title(),
                    "Period": f"{split.get('start', '')} to {split.get('end', '')}",
                    "Rows": split.get("rows", 0),
                }
            )
        st.dataframe(pd.DataFrame(split_rows), hide_index=True, use_container_width=True)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Confusion Matrix")
        matrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        confusion = pd.DataFrame(
            matrix,
            index=["Actual Default", "Actual Non-Default"],
            columns=["Predicted Default", "Predicted Non-Default"],
        )
        st.dataframe(confusion, use_container_width=True)
        default_report = metrics.get("classification_report", {}).get(DEFAULT_LABEL, {})
        st.caption(
            f"Default precision: {default_report.get('precision', 0):.1%} | "
            f"Default recall: {default_report.get('recall', 0):.1%}"
        )

    with chart_right:
        st.markdown("#### Probability Calibration")
        calibration = metrics.get("calibration_curve", {})
        calibration_frame = pd.DataFrame(
            {
                "Predicted Probability": calibration.get("mean_predicted_probability", []),
                "Observed Default Rate": calibration.get("observed_default_rate", []),
            }
        )
        if not calibration_frame.empty:
            calibration_frame["Perfect Calibration"] = calibration_frame["Predicted Probability"]
            st.line_chart(
                calibration_frame.set_index("Predicted Probability"),
                y=["Observed Default Rate", "Perfect Calibration"],
            )

    st.markdown("#### Most Influential Model Features")
    if feature_importance:
        importance_frame = pd.DataFrame(feature_importance)
        importance_frame["Feature"] = importance_frame["feature"].map(label_for)
        st.bar_chart(
            importance_frame.head(10).set_index("Feature")["importance"],
            horizontal=True,
        )
        st.caption(
            "Permutation importance measures how much out-of-time ROC-AUC falls when each feature "
            "is shuffled. It describes the model globally, not a causal relationship."
        )

    st.markdown("#### Score Distribution")
    if isinstance(score_sample, pd.DataFrame) and not score_sample.empty:
        distribution = score_sample.copy()
        distribution["Outcome"] = distribution["actual"].map(
            {DEFAULT_LABEL: "Default", "non_default": "Non-Default"}
        )
        distribution["Score Range"] = pd.cut(
            distribution["default_probability"],
            bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            include_lowest=True,
        ).astype(str)
        counts = (
            distribution.groupby(["Score Range", "Outcome"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        st.bar_chart(counts)
