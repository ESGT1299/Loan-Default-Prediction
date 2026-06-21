"""Feature policy for origination-time loan default prediction."""

TARGET_COLUMN = "loan_status"
DATE_COLUMN = "issue_d"
DEFAULT_LABEL = "default"
NON_DEFAULT_LABEL = "non_default"

DEFAULT_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}

NON_DEFAULT_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}

# These columns are generally known after the loan has been issued or serviced.
# Using them can make the model look excellent while failing in a real approval setting.
LEAKAGE_COLUMNS = {
    "collection_recovery_fee",
    "debt_settlement_flag",
    "deferral_term",
    "hardship_flag",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "last_pymnt_amnt",
    "last_pymnt_d",
    "next_pymnt_d",
    "out_prncp",
    "out_prncp_inv",
    "recoveries",
    "settlement_amount",
    "settlement_date",
    "settlement_percentage",
    "settlement_status",
    "settlement_term",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_int",
    "total_rec_late_fee",
    "total_rec_prncp",
}

# Conservative subset of LendingClub columns available at or near origination.
ORIGINATION_FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "purpose",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
]

MODEL_FEATURES = [feature for feature in ORIGINATION_FEATURES if feature != DATE_COLUMN]


def validate_feature_policy(features: list[str]) -> None:
    """Raise when the proposed model feature set includes leakage columns."""
    leaked = sorted(set(features).intersection(LEAKAGE_COLUMNS))
    if leaked:
        raise ValueError(f"Leakage columns cannot be used as features: {leaked}")
