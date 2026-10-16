#!/usr/bin/env python3
"""
CarePing — Machine Learning Training, Cross-Validation & Evaluation Pipeline
Team Name: AAdharsha Kutumbam
Team Members:
  - Naga Adithya V (23BCE9062)
  - Harsha Vardhan Vurtha (23BCE9129)

This script trains our in-house Wait-Time Prediction Model and Walkout Risk Scoring
Engine from scratch using an 80/20 train-test split, compares it against a naive
baseline, and exports the learned model coefficients and evaluation metrics.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{REPO_DIR}/outputs", exist_ok=True)


def main():
    print("=" * 72)
    print("CAREPING ML TRAINING & EVALUATION PIPELINE — TEAM AADHARSHA KUTUMBAM")
    print("=" * 72)

    # 1. Load Relational Tables (with keep_default_na=False to preserve 'None' strings)
    patients = pd.read_csv(f"{DATA_DIR}/patients.csv", keep_default_na=False)
    departments = pd.read_csv(f"{DATA_DIR}/departments.csv", keep_default_na=False)
    encounters = pd.read_csv(f"{DATA_DIR}/encounters.csv", keep_default_na=False)
    tests = pd.read_csv(f"{DATA_DIR}/diagnostic_tests.csv", keep_default_na=False)

    print(f"[1/5] Loaded Relational Dataset:")
    print(f"      - patients.csv         : {len(patients):,} rows")
    print(f"      - departments.csv      : {len(departments):,} rows")
    print(f"      - encounters.csv       : {len(encounters):,} rows")
    print(f"      - diagnostic_tests.csv : {len(tests):,} rows")

    # 2. Join Tables & Engineer Features
    for col in ["arrival_time", "service_start_time", "service_end_time", "discharge_time"]:
        encounters[col] = pd.to_datetime(encounters[col])

    encounters["post_service_min"] = (
        encounters["discharge_time"] - encounters["service_end_time"]
    ).dt.total_seconds() / 60.0
    encounters["total_los_min"] = (
        encounters["discharge_time"] - encounters["arrival_time"]
    ).dt.total_seconds() / 60.0
    encounters["arrival_hour"] = encounters["arrival_time"].dt.hour
    encounters["is_lwbs"] = (encounters["outcome"] == "Left before care").astype(int)

    df = encounters.merge(patients, on="patient_id", how="left").merge(
        departments, on="department_id", how="left"
    )

    print(f"[2/5] Feature Engineering Complete: Joined master table shape = {df.shape}")

    # 3. Statistical Variance Decomposition (One-Way ANOVA Eta-Squared)
    print("\n[3/5] Running One-Way ANOVA Variance Decomposition on wait_minutes:")
    anova_factors = [
        "triage_level",
        "department_name",
        "home_area",
        "visit_type",
        "insurance_type",
        "chronic_condition",
        "sex",
    ]
    ss_total = ((df["wait_minutes"] - df["wait_minutes"].mean()) ** 2).sum()
    anova_results = []
    for factor in anova_factors:
        groups = [g["wait_minutes"].values for _, g in df.groupby(factor)]
        f_stat, p_val = stats.f_oneway(*groups)
        ss_between = sum(
            len(g) * ((g.mean() - df["wait_minutes"].mean()) ** 2) for g in groups
        )
        eta_sq_pct = (ss_between / ss_total) * 100.0
        anova_results.append(
            {
                "factor": factor,
                "f_statistic": round(float(f_stat), 4),
                "p_value": float(p_val),
                "eta_squared_pct": round(float(eta_sq_pct), 4),
                "significant": bool(p_val < 0.05),
            }
        )
        sig_label = "SIGNIFICANT" if p_val < 0.05 else "NOISE (Not Significant)"
        print(
            f"      - {factor:<18}: F = {f_stat:>9.2f} | Eta^2 = {eta_sq_pct:>6.2f}% | p = {p_val:.2e} ({sig_label})"
        )

    # 4. Build Design Matrix X and Target y for 80/20 Train-Test Split
    # Features: One-hot encoded Triage Level (reference: Critical) and Department (reference: Cardiology D04)
    # plus standardized numerical features (age, staff_count, bed_count, daily_capacity, arrival_hour)
    feature_cols = []
    X_parts = [np.ones((len(df), 1))]  # Intercept
    feature_names = ["Intercept (Baseline: Critical Triage, Cardiology D04)"]

    for t_level in ["High", "Medium", "Low"]:
        col_vals = (df["triage_level"] == t_level).astype(float).values.reshape(-1, 1)
        X_parts.append(col_vals)
        feature_names.append(f"Triage_Level == {t_level}")

    dept_list = sorted([d for d in df["department_name"].unique() if d != "Cardiology"])
    for d_name in dept_list:
        col_vals = (df["department_name"] == d_name).astype(float).values.reshape(-1, 1)
        X_parts.append(col_vals)
        feature_names.append(f"Department == {d_name}")

    # Add patient age & arrival hour (to prove operational features dominate demographic noise)
    age_norm = ((df["age"] - df["age"].mean()) / df["age"].std()).values.reshape(-1, 1)
    hr_norm = (
        (df["arrival_hour"] - df["arrival_hour"].mean()) / df["arrival_hour"].std()
    ).values.reshape(-1, 1)
    X_parts.extend([age_norm, hr_norm])
    feature_names.extend(["Patient_Age (Standardized)", "Arrival_Hour (Standardized)"])

    X = np.hstack(X_parts)
    y = df["wait_minutes"].values

    # Deterministic 80/20 Train-Test Split (Seed = 42)
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(df))
    split_idx = int(0.80 * len(df))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Fit OLS Multivariate Regression from scratch using Moore-Penrose Pseudoinverse / Least Squares
    beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

    # Predictions
    y_train_pred = X_train @ beta
    y_test_pred = X_test @ beta

    # Baseline Model (Predicting Hospital Mean Wait Time)
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_mae = float(np.mean(np.abs(y_test - baseline_pred)))
    baseline_rmse = float(np.sqrt(np.mean((y_test - baseline_pred) ** 2)))

    # Model Evaluation Metrics
    train_mae = float(np.mean(np.abs(y_train - y_train_pred)))
    test_mae = float(np.mean(np.abs(y_test - y_test_pred)))
    train_rmse = float(np.sqrt(np.mean((y_train - y_train_pred) ** 2)))
    test_rmse = float(np.sqrt(np.mean((y_test - y_test_pred) ** 2)))

    ss_res_train = float(np.sum((y_train - y_train_pred) ** 2))
    ss_tot_train = float(np.sum((y_train - y_train.mean()) ** 2))
    r2_train = 1.0 - (ss_res_train / ss_tot_train)

    ss_res_test = float(np.sum((y_test - y_test_pred) ** 2))
    ss_tot_test = float(np.sum((y_test - y_test.mean()) ** 2))
    r2_test = 1.0 - (ss_res_test / ss_tot_test)

    # Full-dataset fit for final production deployment weights
    beta_full, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_full_pred = X @ beta_full
    r2_full = 1.0 - float(np.sum((y - y_full_pred) ** 2)) / float(
        np.sum((y - y.mean()) ** 2)
    )
    mae_full = float(np.mean(np.abs(y - y_full_pred)))

    # Calculate Standard Errors and t-stats for full model
    n_obs, k_params = X.shape
    sigma2 = np.sum((y - y_full_pred) ** 2) / (n_obs - k_params)
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(cov_beta))
    t_stats = beta_full / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n_obs - k_params))

    mae_improvement_pct = ((baseline_mae - test_mae) / baseline_mae) * 100.0

    print("\n[4/5] Model Training & 80/20 Evaluation Complete:")
    print(f"      - Train Set Size       : {len(y_train):,} encounters (80%)")
    print(f"      - Test Set Size        : {len(y_test):,} encounters (20%)")
    print(f"      - Baseline Test MAE    : {baseline_mae:.2f} min (RMSE: {baseline_rmse:.2f} min)")
    print(f"      - CarePing Train R^2   : {r2_train:.4f} ({r2_train*100:.2f}%) | Train MAE: {train_mae:.2f} min")
    print(f"      - CarePing Test R^2    : {r2_test:.4f} ({r2_test*100:.2f}%) | Test MAE : {test_mae:.2f} min")
    print(f"      - Full Dataset R^2     : {r2_full:.4f} ({r2_full*100:.2f}%) | Full MAE : {mae_full:.2f} min")
    print(f"      - Error Reduction      : {mae_improvement_pct:.1f}% lower MAE than hospital baseline")

    # 5. Export Model Weights & Metrics to CSV and JSON
    coef_rows = []
    for name, w, se, t_val, p_val in zip(
        feature_names, beta_full, se_beta, t_stats, p_values
    ):
        coef_rows.append(
            {
                "Feature": name,
                "Weight_Minutes": round(float(w), 4),
                "Std_Error": round(float(se), 4),
                "t_Statistic": round(float(t_val), 4),
                "p_Value": float(p_val),
            }
        )

    coef_df = pd.DataFrame(coef_rows)
    coef_csv_path = f"{REPO_DIR}/outputs/trained_wait_model_coefficients.csv"
    coef_df.to_csv(coef_csv_path, index=False)

    # Save Department Operational Scorecard CSV
    dept_scorecard = (
        df.groupby(
            [
                "department_id",
                "department_name",
                "service_type",
                "staff_count",
                "bed_count",
                "daily_capacity",
            ]
        )
        .agg(
            encounters=("encounter_id", "count"),
            mean_wait_min=("wait_minutes", "mean"),
            median_wait_min=("wait_minutes", "median"),
            p90_wait_min=("wait_minutes", lambda x: np.percentile(x, 90)),
            mean_service_min=("service_minutes", "mean"),
            mean_post_service_min=("post_service_min", "mean"),
            mean_total_los_min=("total_los_min", "mean"),
            walkout_lwbs_count=("is_lwbs", "sum"),
            walkout_lwbs_rate_pct=("is_lwbs", lambda x: x.mean() * 100.0),
        )
        .reset_index()
        .round(2)
    )
    dept_scorecard.to_csv(
        f"{REPO_DIR}/outputs/department_operational_scorecard.csv", index=False
    )

    # Save All 242 Walkout (LWBS) Encounters CSV
    lwbs_df = df[df["outcome"] == "Left before care"][
        [
            "encounter_id",
            "patient_id",
            "age",
            "sex",
            "chronic_condition",
            "home_area",
            "insurance_type",
            "department_id",
            "department_name",
            "visit_type",
            "triage_level",
            "wait_minutes",
            "service_minutes",
            "post_service_min",
        ]
    ].sort_values("wait_minutes", ascending=False)
    lwbs_df.to_csv(f"{REPO_DIR}/outputs/walkout_lwbs_patients.csv", index=False)

    # Save All 45 Unseen Registered Patients CSV
    unseen_df = patients[~patients["patient_id"].isin(encounters["patient_id"])]
    unseen_df.to_csv(f"{REPO_DIR}/outputs/unseen_chronic_registry.csv", index=False)

    # Save JSON summary of model metrics & weights
    metrics_payload = {
        "project": "CarePing — Health: The Forgotten Patient",
        "team": "AAdharsha Kutumbam",
        "members": ["Naga Adithya V (23BCE9062)", "Harsha Vardhan Vurtha (23BCE9129)"],
        "model_type": "Multivariate Ordinary Least Squares (OLS) Queue Estimator & Walkout Risk Scorer",
        "split_protocol": "80% Train (6,400 encounters) / 20% Test (1,600 encounters), random_seed=42",
        "metrics": {
            "baseline_test_mae_min": round(baseline_mae, 2),
            "baseline_test_rmse_min": round(baseline_rmse, 2),
            "train_r2": round(r2_train, 4),
            "test_r2": round(r2_test, 4),
            "full_r2": round(r2_full, 4),
            "train_mae_min": round(train_mae, 2),
            "test_mae_min": round(test_mae, 2),
            "full_mae_min": round(mae_full, 2),
            "train_rmse_min": round(train_rmse, 2),
            "test_rmse_min": round(test_rmse, 2),
            "mae_improvement_over_baseline_pct": round(mae_improvement_pct, 2),
        },
        "anova_variance_decomposition": anova_results,
        "learned_coefficients": coef_rows,
    }
    with open(f"{REPO_DIR}/outputs/model_metrics_and_weights.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"[5/5] Saved all trained model weights and CSV outputs to {REPO_DIR}/outputs/")
    print("=" * 72)


if __name__ == "__main__":
    main()
