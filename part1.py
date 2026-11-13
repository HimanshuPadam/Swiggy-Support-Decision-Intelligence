"""
IIT Mandi - Minor in AI & Data Science
Capstone Project: Swiggy - Support Decision Intelligence
Part 1: Statistical Inference and Regression Analysis

Author: Student
Dataset: data/records.csv
Output: outputs/part1_results.txt

This script implements:
  - Task 1.1: Estimation (Mean, Sample SD, Standard Error, 95% Confidence Interval)
  - Task 1.2: Hypothesis Testing (One-sample two-sided Z-test)
  - Task 1.3: Sample Size Determination (Margin of error formula)
  - Task 1.4: Simple Linear Regression (OLS covariance/variance, TSS/ESS/RSS decomposition, R2, RMSE, MAE)
  - Task 1.5: Gradient Descent Optimization (Batch GD on standardized feature, loss trajectory, parameter recovery)
  - Task 1.6: Residual Analysis & Log-Log Transformation (Heteroscedasticity inspection, stabilizing variance)
  - Task 1.7: Multiple Linear Regression (Normal equation via np.linalg.solve, R2 and Adjusted R2)

All formulas are implemented from first principles using standard library and NumPy primitives.
No ready-made regression black-boxes (such as polyfit, cov, corrcoef, lstsq) are used.
"""

import os
import math
import csv
import numpy as np


def load_dataset(csv_path: str):
    """
    Reads the dataset from a CSV file using Python's standard csv module.
    Returns:
        dict: A dictionary mapping column names to NumPy float arrays.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at relative path: {csv_path}")

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("CSV file has no header row.")

        data = {col: [] for col in fieldnames}
        for row in reader:
            for col in fieldnames:
                data[col].append(float(row[col]) if col != "record_id" else row[col])

    # Convert numeric columns to NumPy arrays
    arrays = {}
    for col in fieldnames:
        if col != "record_id":
            arrays[col] = np.array(data[col], dtype=np.float64)
        else:
            arrays[col] = data[col]

    return arrays


def format_value(val):
    """
    Helper to format results for output text:
      - Integers as plain integers
      - Floats rounded to 4 decimal places
      - Strings (like 'yes'/'no') as strings
    """
    if isinstance(val, (int, np.integer)):
        return str(val)
    elif isinstance(val, (float, np.floating)):
        return f"{val:.4f}"
    else:
        return str(val)


def main():
    # ---------------------------------------------------------
    # 0. Setup and Data Ingestion
    # ---------------------------------------------------------
    data_path = os.path.join("data", "records.csv")
    data = load_dataset(data_path)

    # Primary continuous target: resolution_minutes (value)
    # Primary continuous predictor: account_value_inr_thousands (X4)
    y = data["resolution_minutes"]
    x4 = data["account_value_inr_thousands"]
    x1 = data["customer_tenure_months"]
    x2 = data["earlier_ticket_count_90d"]
    x3 = data["sla_consumed_ratio"]
    x5 = data["satisfaction_score"]
    x6 = data["reopened_flag"]
    x7 = data["self_service_first_flag"]

    n = len(y)
    results = {}

    print("=" * 70)
    print("IIT Mandi Capstone: Swiggy Support Decision Intelligence - Part 1")
    print("=" * 70)

    # ---------------------------------------------------------
    # Task 1.1 — Estimation
    # ---------------------------------------------------------
    # Sample mean: y_bar = (1/n) * sum(y_i)
    # Sample SD: s = sqrt( (1 / (n - 1)) * sum((y_i - y_bar)^2) )
    # Standard Error: SE = SD / sqrt(n)
    # 95% Confidence Interval: [y_bar - 1.96 * SE, y_bar + 1.96 * SE]
    mean_val = float(np.sum(y) / n)
    sd_val = float(math.sqrt(np.sum((y - mean_val) ** 2) / (n - 1)))
    se_val = float(sd_val / math.sqrt(n))
    ci95_low = float(mean_val - 1.96 * se_val)
    ci95_high = float(mean_val + 1.96 * se_val)

    results["n"] = n
    results["mean_value"] = mean_val
    results["sd_value"] = sd_val
    results["se_value"] = se_val
    results["ci95_low"] = ci95_low
    results["ci95_high"] = ci95_high

    print(f"\n[Task 1.1 Estimation]")
    print(f"  Sample size (n)        : {results['n']}")
    print(f"  Mean (mean_value)      : {results['mean_value']:.4f}")
    print(f"  Sample SD (sd_value)   : {results['sd_value']:.4f}")
    print(f"  Standard Error (se)    : {results['se_value']:.4f}")
    print(f"  95% CI                 : [{results['ci95_low']:.4f}, {results['ci95_high']:.4f}]")

    # ---------------------------------------------------------
    # Task 1.2 — Hypothesis Testing
    # ---------------------------------------------------------
    # H0: mu = 8.0  vs  H1: mu != 8.0
    # Two-sided Z-test: z = (y_bar - mu_0) / SE
    # Decision rule: Reject H0 at alpha = 0.05 if |z| > 1.96
    mu_0 = 8.0
    z_stat = float((mean_val - mu_0) / se_val)
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

    results["z_stat"] = z_stat
    results["reject_h0"] = reject_h0

    print(f"\n[Task 1.2 Hypothesis Testing]")
    print(f"  H0: mu = 8.0 vs H1: mu != 8.0")
    print(f"  Z-statistic            : {results['z_stat']:.4f}")
    print(f"  Reject H0 (|z| > 1.96) : {results['reject_h0']}")

    # ---------------------------------------------------------
    # Task 1.3 — Sample Size Determination
    # ---------------------------------------------------------
    # Margin of error E = 0.5 minutes
    # Formula: n_required = ceil( (1.96 * SD / E)^2 )
    E = 0.5
    n_required = int(math.ceil(((1.96 * sd_val) / E) ** 2))

    results["n_required"] = n_required

    print(f"\n[Task 1.3 Sample Size Planning]")
    print(f"  Margin of error (E)    : {E}")
    print(f"  Required n (n_required): {results['n_required']}")

    # ---------------------------------------------------------
    # Task 1.4 — Simple Linear Regression
    # ---------------------------------------------------------
    # Regress y (resolution_minutes) on X4 (account_value_inr_thousands)
    # Using sample covariance and variance formulas with divisor n - 1:
    #   s_xx = (1 / (n - 1)) * sum((x - x_bar)^2)
    #   s_xy = (1 / (n - 1)) * sum((x - x_bar) * (y - y_bar))
    #   slope = s_xy / s_xx
    #   intercept = y_bar - slope * x_bar
    x_mean = float(np.sum(x4) / n)
    s_xx = float(np.sum((x4 - x_mean) ** 2) / (n - 1))
    s_xy = float(np.sum((x4 - x_mean) * (y - mean_val)) / (n - 1))

    reg_slope = float(s_xy / s_xx)
    reg_intercept = float(mean_val - reg_slope * x_mean)

    # Fitted values and residuals
    y_fitted = reg_intercept + reg_slope * x4
    residuals = y - y_fitted

    # Total Sum of Squares (TSS), Explained Sum of Squares (ESS), Residual Sum of Squares (RSS)
    tss = float(np.sum((y - mean_val) ** 2))
    ess = float(np.sum((y_fitted - mean_val) ** 2))
    rss = float(np.sum(residuals ** 2))

    # Check ANOVA decomposition identity: TSS = ESS + RSS
    tss_equals = "yes" if abs(tss - (ess + rss)) < 1e-3 else "no"

    r2 = float(ess / tss)
    rmse = float(math.sqrt(rss / n))
    mae = float(np.sum(np.abs(residuals)) / n)

    results["reg_slope"] = reg_slope
    results["reg_intercept"] = reg_intercept
    results["tss"] = tss
    results["ess"] = ess
    results["rss"] = rss
    results["tss_equals_ess_plus_rss"] = tss_equals
    results["r2"] = r2
    results["rmse"] = rmse
    results["mae"] = mae

    print(f"\n[Task 1.4 Simple Linear Regression]")
    print(f"  Model                  : y_hat = {reg_intercept:.4f} + {reg_slope:.4f} * X4")
    print(f"  TSS                    : {results['tss']:.4f}")
    print(f"  ESS                    : {results['ess']:.4f}")
    print(f"  RSS                    : {results['rss']:.4f}")
    print(f"  TSS == ESS + RSS?      : {results['tss_equals_ess_plus_rss']}")
    print(f"  R^2                    : {results['r2']:.4f}")
    print(f"  RMSE                   : {results['rmse']:.4f}")
    print(f"  MAE                    : {results['mae']:.4f}")

    # ---------------------------------------------------------
    # Task 1.5 — Gradient Descent Optimization
    # ---------------------------------------------------------
    # Standardise X4 using its sample standard deviation (divisor n - 1):
    #   z_i = (x4_i - x_bar) / s_x
    # Model: y_pred = m * z + c
    # Loss: Mean Squared Error J(m, c) = (1 / (2 * n)) * sum((y_pred - y)^2)
    # Gradients:
    #   dJ/dm = (1 / n) * sum((y_pred - y) * z)
    #   dJ/dc = (1 / n) * sum(y_pred - y)
    # Updates:
    #   m <- m - lr * dJ/dm
    #   c <- c - lr * dJ/dc
    # 500 full-batch updates with lr = 0.1, starting from m = 0, c = 0
    sx = math.sqrt(s_xx)
    z = (x4 - x_mean) / sx

    m = 0.0
    c = 0.0
    lr = 0.1
    iterations = 500

    gd_costs = {}

    for step in range(iterations + 1):
        y_pred = m * z + c
        err = y_pred - y
        # Cost at current step
        cost = float((1.0 / (2.0 * n)) * np.sum(err ** 2))

        if step in [0, 1, 10, 100, 500]:
            gd_costs[step] = cost

        if step == iterations:
            break

        # Compute batch gradients
        grad_m = float((1.0 / n) * np.sum(err * z))
        grad_c = float((1.0 / n) * np.sum(err))

        # Update parameters
        m -= lr * grad_m
        c -= lr * grad_c

    # Convert parameters back to original unstandardized scale:
    # y_pred = m * ((x - x_bar) / s_x) + c = (m / s_x) * x + (c - (m / s_x) * x_bar)
    gd_slope = float(m / sx)
    gd_intercept = float(c - gd_slope * x_mean)
    gd_slope_gap = float(abs(gd_slope - reg_slope))

    results["gd_J0"] = gd_costs[0]
    results["gd_J1"] = gd_costs[1]
    results["gd_J10"] = gd_costs[10]
    results["gd_J100"] = gd_costs[100]
    results["gd_J500"] = gd_costs[500]
    results["gd_slope"] = gd_slope
    results["gd_intercept"] = gd_intercept
    results["gd_slope_gap"] = gd_slope_gap

    print(f"\n[Task 1.5 Gradient Descent (500 steps, lr=0.1)]")
    print(f"  J(0)                   : {results['gd_J0']:.4f}")
    print(f"  J(1)                   : {results['gd_J1']:.4f}")
    print(f"  J(10)                  : {results['gd_J10']:.4f}")
    print(f"  J(100)                 : {results['gd_J100']:.4f}")
    print(f"  J(500)                 : {results['gd_J500']:.4f}")
    print(f"  Recovered slope        : {results['gd_slope']:.4f}")
    print(f"  Recovered intercept    : {results['gd_intercept']:.4f}")
    print(f"  Slope gap (|GD - OLS|) : {results['gd_slope_gap']:.6f} (check: < 0.001)")

    # ---------------------------------------------------------
    # Task 1.6 — Residual Analysis & Log-Log Transformation
    # ---------------------------------------------------------
    # Heteroscedasticity check on raw simple linear regression:
    # 1. Stable ascending sort by fitted values y_fitted
    # 2. Split residuals into lower (first 400) and upper (last 400) halves
    # 3. Calculate sample SD of each half (divisor m - 1 = 399) and their ratio
    sort_order_raw = np.argsort(y_fitted, kind="stable")
    sorted_resids_raw = residuals[sort_order_raw]

    res_lower_raw = sorted_resids_raw[:400]
    res_upper_raw = sorted_resids_raw[400:]

    mean_res_low = np.sum(res_lower_raw) / 400.0
    mean_res_up = np.sum(res_upper_raw) / 400.0

    resid_sd_lower_raw = float(math.sqrt(np.sum((res_lower_raw - mean_res_low) ** 2) / (400 - 1)))
    resid_sd_upper_raw = float(math.sqrt(np.sum((res_upper_raw - mean_res_up) ** 2) / (400 - 1)))
    resid_sd_ratio_raw = float(resid_sd_upper_raw / resid_sd_lower_raw)

    results["resid_sd_lower_raw"] = resid_sd_lower_raw
    results["resid_sd_upper_raw"] = resid_sd_upper_raw
    results["resid_sd_ratio_raw"] = resid_sd_ratio_raw

    # Log-Log Regression: ln(y) on ln(X4)
    # y_log = beta_0 + beta_1 * x_log
    log_y = np.log(y)
    log_x4 = np.log(x4)

    lx_mean = float(np.sum(log_x4) / n)
    ly_mean = float(np.sum(log_y) / n)

    s_lxx = float(np.sum((log_x4 - lx_mean) ** 2) / (n - 1))
    s_lxy = float(np.sum((log_x4 - lx_mean) * (log_y - ly_mean)) / (n - 1))

    loglog_slope = float(s_lxy / s_lxx)
    loglog_intercept = float(ly_mean - loglog_slope * lx_mean)

    log_y_fitted = loglog_intercept + loglog_slope * log_x4
    log_residuals = log_y - log_y_fitted

    tss_log = float(np.sum((log_y - ly_mean) ** 2))
    rss_log = float(np.sum(log_residuals ** 2))
    loglog_r2 = float(1.0 - (rss_log / tss_log))
    loglog_rmse = float(math.sqrt(rss_log / n))

    # Stable sort by fitted values of log-log model
    sort_order_log = np.argsort(log_y_fitted, kind="stable")
    sorted_resids_log = log_residuals[sort_order_log]

    res_lower_log = sorted_resids_log[:400]
    res_upper_log = sorted_resids_log[400:]

    mean_res_low_log = np.sum(res_lower_log) / 400.0
    mean_res_up_log = np.sum(res_upper_log) / 400.0

    resid_sd_lower_log = float(math.sqrt(np.sum((res_lower_log - mean_res_low_log) ** 2) / (400 - 1)))
    resid_sd_upper_log = float(math.sqrt(np.sum((res_upper_log - mean_res_up_log) ** 2) / (400 - 1)))
    resid_sd_ratio_log = float(resid_sd_upper_log / resid_sd_lower_log)

    results["loglog_intercept"] = loglog_intercept
    results["loglog_slope"] = loglog_slope
    results["loglog_r2"] = loglog_r2
    results["loglog_rmse"] = loglog_rmse
    results["resid_sd_lower_log"] = resid_sd_lower_log
    results["resid_sd_upper_log"] = resid_sd_upper_log
    results["resid_sd_ratio_log"] = resid_sd_ratio_log

    print(f"\n[Task 1.6 Residual Analysis & Log-Log Transformation]")
    print(f"  Raw Residual SD Lower  : {results['resid_sd_lower_raw']:.4f}")
    print(f"  Raw Residual SD Upper  : {results['resid_sd_upper_raw']:.4f}")
    print(f"  Raw SD Ratio (Upper/Low: {results['resid_sd_ratio_raw']:.4f} (heteroscedastic)")
    print(f"  Log-Log Intercept      : {results['loglog_intercept']:.4f}")
    print(f"  Log-Log Slope          : {results['loglog_slope']:.4f}")
    print(f"  Log-Log R^2            : {results['loglog_r2']:.4f}")
    print(f"  Log-Log RMSE           : {results['loglog_rmse']:.4f}")
    print(f"  Log Residual SD Lower  : {results['resid_sd_lower_log']:.4f}")
    print(f"  Log Residual SD Upper  : {results['resid_sd_upper_log']:.4f}")
    print(f"  Log SD Ratio (Upper/Low: {results['resid_sd_ratio_log']:.4f} (homoscedastic ~ 1.0)")

    # ---------------------------------------------------------
    # Task 1.7 — Multiple Linear Regression
    # ---------------------------------------------------------
    # Regress ln(value) on 8 design columns in exact order:
    #   1. Constant (ones)
    #   2. ln(X4)
    #   3. X1 (customer_tenure_months)
    #   4. X2 (earlier_ticket_count_90d)
    #   5. X3 (sla_consumed_ratio)
    #   6. X5 (satisfaction_score)
    #   7. X6 (reopened_flag)
    #   8. X7 (self_service_first_flag)
    #
    # Normal Equation:
    #   (X^T * X) * beta = X^T * y
    # Solved using numpy.linalg.solve(X^T * X, X^T * y)
    X = np.column_stack([
        np.ones(n, dtype=np.float64),
        log_x4,
        x1,
        x2,
        x3,
        x5,
        x6,
        x7
    ])

    XtX = X.T @ X
    Xty = X.T @ log_y

    beta = np.linalg.solve(XtX, Xty)

    for i in range(8):
        results[f"mr_beta_{i}"] = float(beta[i])

    mr_y_fitted = X @ beta
    mr_residuals = log_y - mr_y_fitted
    mr_rss = float(np.sum(mr_residuals ** 2))

    mr_r2 = float(1.0 - (mr_rss / tss_log))

    # Adjusted R-squared formula:
    # R^2_adj = 1 - [ (1 - R^2) * (n - 1) / (n - p - 1) ]
    # where p is the number of slope predictors (excluding intercept).
    p_mr = 7  # 7 slope predictors: ln(X4), X1, X2, X3, X5, X6, X7
    mr_adj_r2 = float(1.0 - (1.0 - mr_r2) * (n - 1) / (n - p_mr - 1))

    # Adjusted R-squared for loglog model (p = 1 predictor):
    p_loglog = 1
    loglog_adj_r2 = float(1.0 - (1.0 - loglog_r2) * (n - 1) / (n - p_loglog - 1))

    results["mr_r2"] = mr_r2
    results["mr_adj_r2"] = mr_adj_r2
    results["loglog_adj_r2"] = loglog_adj_r2

    print(f"\n[Task 1.7 Multiple Regression]")
    for i in range(8):
        print(f"  beta_{i} (mr_beta_{i})   : {results[f'mr_beta_{i}']:.4f}")
    print(f"  Multiple R^2 (mr_r2)   : {results['mr_r2']:.4f}")
    print(f"  Multiple Adj R^2       : {results['mr_adj_r2']:.4f}")
    print(f"  Log-Log Adj R^2        : {results['loglog_adj_r2']:.4f}")

    # ---------------------------------------------------------
    # Verification & Checkpoints
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERIFICATION CHECKPOINTS")
    print("=" * 70)
    assert results["n"] == 800, f"Expected n=800, got {results['n']}"
    assert abs(results["mean_value"] - 8.9382) < 0.001, (
        f"Expected mean_value=8.9382, got {results['mean_value']:.4f}"
    )
    assert results["gd_slope_gap"] < 0.001, (
        f"Expected gd_slope_gap < 0.001, got {results['gd_slope_gap']:.6f}"
    )
    print("✓ Checkpoint 1: n = 800 (Passed)")
    print("✓ Checkpoint 2: mean_value = 8.9382 within normal rounding (Passed)")
    print(f"✓ Checkpoint 3: gd_slope_gap = {results['gd_slope_gap']:.6f} < 0.001 (Passed)")

    # ---------------------------------------------------------
    # Save Results to outputs/part1_results.txt
    # ---------------------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    output_file_path = os.path.join("outputs", "part1_results.txt")

    # Define the exact required key order matching Part 1 specifications
    required_keys = [
        # Task 1.1
        "n", "mean_value", "sd_value", "se_value", "ci95_low", "ci95_high",
        # Task 1.2
        "z_stat", "reject_h0",
        # Task 1.3
        "n_required",
        # Task 1.4
        "reg_slope", "reg_intercept", "tss", "ess", "rss",
        "tss_equals_ess_plus_rss", "r2", "rmse", "mae",
        # Task 1.5
        "gd_J0", "gd_J1", "gd_J10", "gd_J100", "gd_J500",
        "gd_slope", "gd_intercept", "gd_slope_gap",
        # Task 1.6
        "resid_sd_lower_raw", "resid_sd_upper_raw", "resid_sd_ratio_raw",
        "loglog_intercept", "loglog_slope", "loglog_r2", "loglog_rmse",
        "resid_sd_lower_log", "resid_sd_upper_log", "resid_sd_ratio_log",
        # Task 1.7
        "mr_beta_0", "mr_beta_1", "mr_beta_2", "mr_beta_3",
        "mr_beta_4", "mr_beta_5", "mr_beta_6", "mr_beta_7",
        "mr_r2", "mr_adj_r2", "loglog_adj_r2"
    ]

    # Verify all keys are present
    missing_keys = [k for k in required_keys if k not in results]
    if missing_keys:
        raise KeyError(f"Missing required output keys: {missing_keys}")

    with open(output_file_path, mode="w", encoding="utf-8") as out_f:
        for key in required_keys:
            val_str = format_value(results[key])
            out_f.write(f"{key} = {val_str}\n")

    print(f"\nAll {len(required_keys)} results successfully saved to: {output_file_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

