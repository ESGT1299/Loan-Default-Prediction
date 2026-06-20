# Model Card: Loan Default Risk Model

## Model Summary

This model estimates the probability that an originated LendingClub loan ends in a default-like resolved status. It is a portfolio project demonstrating leakage-aware feature design, temporal validation, probability calibration, model evaluation, and Streamlit deployment.

## Intended Use

- Educational and portfolio analysis
- Technical interviews and project demonstrations
- Exploration of credit-risk-style modeling workflows
- Comparison of borrower and loan profiles

## Out-of-Scope Use

- Automated loan approval or denial
- Pricing, lending limits, or adverse action notices
- Production underwriting
- Decisions involving protected classes
- Financial, legal, or credit advice

## Data

- Source: LendingClub accepted loans dataset from Kaggle
- Historical coverage: 2007-2018
- Target:
  - `default`: Charged Off or Default
  - `non_default`: Fully Paid
- Unresolved statuses such as Current are excluded.
- The final artifact uses 59,738 resolved loans from a uniform 100,000-row sample across the source file.

## Feature Policy

Only fields available near origination are used. Post-origination fields such as payments, recoveries, settlement information, remaining principal, and last payment dates are explicitly blocked to reduce target leakage.

The loan issue date is used to create chronological datasets but is not used as a model predictor.

## Methodology

- Preprocessing:
  - median imputation for numeric variables
  - most-frequent imputation for categorical variables
  - one-hot encoding with unknown-category handling
- Estimator: class-weighted Random Forest
- Probability calibration: sigmoid calibration
- Validation:
  - training: 2007-07 to 2016-03
  - calibration: 2016-03 to 2017-02
  - test: 2017-02 to 2018-12
- Decision threshold: selected on calibration data to maximize default-class F1

## Out-of-Time Performance

| Metric | Value |
|---|---:|
| ROC-AUC | 0.702 |
| PR-AUC | 0.368 |
| Brier score | 0.153 |
| Threshold | 0.270 |
| Default precision | 0.338 |
| Default recall | 0.617 |
| Default F1 | 0.436 |

These values reflect the saved artifact trained on June 17, 2026. Results may change when the training sample or model configuration changes.

## Interpretation

The output is a calibrated model estimate based on historical patterns. A score above the selected threshold indicates that the profile should receive additional analytical review in this demonstration workflow. It is not an automatic rejection rule.

Permutation importance in the dashboard shows global model sensitivity. It does not establish causality and does not explain a specific applicant by itself.

## Limitations and Risks

- LendingClub data may not represent other lenders, products, countries, or current economic conditions.
- Historical lending decisions can contain selection bias because only accepted loans have observed repayment outcomes.
- The dataset can reflect historical social and institutional inequities.
- Protected attributes are not intentionally modeled, but other variables may act as proxies.
- Calibration can deteriorate over time under economic or policy changes.
- The model has not undergone legal, compliance, fairness, stress, or production validation.
- The 100,000-row source sample is suitable for portfolio demonstration but is not the full dataset.

## Required Controls Before Production

- Full data lineage and feature availability review
- Fair lending and subgroup performance analysis
- Adverse action reason methodology
- Independent model validation
- Monitoring for drift, calibration, and stability
- Data security and access controls
- Human review and escalation policy
- Legal and compliance approval

