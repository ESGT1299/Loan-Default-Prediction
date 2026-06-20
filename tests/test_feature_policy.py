import unittest

import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from src.loan_default_risk.data import make_binary_target, normalize_percent, resolve_accepted_loans_path
from src.loan_default_risk.features import (
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    ORIGINATION_FEATURES,
    validate_feature_policy,
)
from src.loan_default_risk.modeling import select_threshold, temporal_split


class FeaturePolicyTests(unittest.TestCase):
    def test_origin_features_do_not_include_known_leakage(self):
        self.assertFalse(set(ORIGINATION_FEATURES).intersection(LEAKAGE_COLUMNS))

    def test_numeric_features_are_part_of_origin_feature_policy(self):
        self.assertTrue(set(NUMERIC_FEATURES).issubset(ORIGINATION_FEATURES))

    def test_validate_feature_policy_rejects_leakage(self):
        with self.assertRaises(ValueError):
            validate_feature_policy(["loan_amnt", "recoveries", "total_pymnt"])

    def test_target_mapping_keeps_only_resolved_outcomes(self):
        self.assertEqual(make_binary_target("Charged Off"), "default")
        self.assertEqual(make_binary_target("Fully Paid"), "non_default")
        self.assertIsNone(make_binary_target("Current"))

    def test_percent_parser(self):
        self.assertEqual(normalize_percent("13.56%"), 13.56)
        self.assertEqual(normalize_percent(8.5), 8.5)

    def test_resolves_nested_kaggle_csv_folder(self):
        with TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "accepted_2007_to_2018q4.csv"
            nested.mkdir()
            csv_path = nested / "accepted_2007_to_2018Q4.csv"
            csv_path.write_text("loan_status\n", encoding="utf-8")
            self.assertEqual(resolve_accepted_loans_path(tmpdir), csv_path)

    def test_temporal_split_preserves_chronological_order(self):
        data = pd.DataFrame(
            {
                "issue_d": pd.date_range("2010-01-01", periods=20, freq="MS"),
                "loan_status": ["default", "non_default"] * 10,
            }
        )
        train, calibration, test = temporal_split(data)
        self.assertLess(train["issue_d"].max(), calibration["issue_d"].min())
        self.assertLess(calibration["issue_d"].max(), test["issue_d"].min())
        self.assertEqual(len(train) + len(calibration) + len(test), len(data))

    def test_threshold_is_a_valid_probability(self):
        target = pd.Series(["default", "default", "non_default", "non_default"])
        probabilities = np.array([0.9, 0.6, 0.4, 0.1])
        threshold = select_threshold(target, probabilities)
        self.assertGreaterEqual(threshold, 0)
        self.assertLessEqual(threshold, 1)


if __name__ == "__main__":
    unittest.main()
