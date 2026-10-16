#!/usr/bin/env python3
"""
Automated Unit & Integrity Test Suite for CarePing OS v2.0
"""

import unittest
import numpy as np
from train_and_evaluate_model import (
    load_and_sanitize_relational_data,
    build_design_matrix,
    ScratchOLSRegressor,
    ScratchRidgeRegressor,
    simulate_priority_aging_scheduler,
)


class TestCarePingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patients, cls.departments, cls.encounters, cls.tests, cls.df, cls.forensic = (
            load_and_sanitize_relational_data()
        )
        cls.X, cls.feature_names = build_design_matrix(cls.df)
        cls.y = cls.df["wait_minutes"].values

    def test_01_dataset_record_counts_and_none_string_trap(self):
        self.assertEqual(len(self.patients), 2000)
        self.assertEqual(len(self.departments), 12)
        self.assertEqual(len(self.encounters), 8000)
        self.assertEqual(len(self.tests), 5000)
        self.assertEqual(self.patients.isna().sum().sum(), 0)
        self.assertEqual(self.forensic["none_string_trap_patients_preserved"], 747)

    def test_02_forensic_timestamp_and_lwbs_anomalies(self):
        self.assertEqual(self.forensic["tests_inside_encounter_window"], 2)
        self.assertEqual(self.forensic["lwbs_ghost_service_records_sanitized"], 242)
        self.assertEqual(self.forensic["unseen_registered_patients_count"], 45)
        self.assertTrue((self.df.loc[self.df["is_lwbs"] == 1, "clean_service_minutes"] == 0).all())

    def test_03_from_scratch_ols_and_ridge_accuracy_and_speed(self):
        ols = ScratchOLSRegressor().fit(self.X, self.y)
        ridge = ScratchRidgeRegressor(alpha=2.0).fit(self.X, self.y)
        preds_ols = ols.predict(self.X)
        mae_ols = float(np.mean(np.abs(self.y - preds_ols)))
        r2_ols = 1.0 - float(np.sum((self.y - preds_ols) ** 2)) / float(
            np.sum((self.y - np.mean(self.y)) ** 2)
        )
        self.assertGreater(r2_ols, 0.515)
        self.assertLess(mae_ols, 14.0)
        self.assertLess(ols.fit_time_ms, 100.0)
        self.assertLess(ridge.fit_time_ms, 50.0)

    def test_04_priority_aging_scheduler_eliminates_starvation(self):
        sim = simulate_priority_aging_scheduler(self.df, wait_cap_min=45.0, boarding_cut_pct=0.50)
        self.assertLess(sim["simulated_low_triage_wait_min"], sim["baseline_low_triage_wait_min"])
        self.assertGreater(sim["annual_bed_hours_reclaimed"], 3400)
        self.assertGreater(sim["projected_walkouts_prevented"], 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
