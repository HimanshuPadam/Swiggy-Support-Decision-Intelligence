"""
IIT Mandi - Minor in AI & Data Science
Capstone Project: Swiggy - Support Decision Intelligence
Part 2: Classification, Logistic Regression, Threshold Optimization, and KNN

Author: Student
Dataset: data/records.csv
Outputs:
  - outputs/part2_results.txt
  - outputs/part2_thresholds.csv
  - outputs/part2_knn_cv.csv

This script implements:
  - Task 2.1: Stratified 70/30 split and training-only feature standardisation
  - Task 2.2: Logistic regression from scratch (gradient descent, BCE loss, weights, bias, test BCE)
  - Task 2.3: Learning-rate sensitivity analysis (eta in {0.01, 0.1, 1.0} for 200 epochs)
  - Task 2.4: Odds ratios for predictors (exp(w_j))
  - Task 2.5: Test evaluation at default (0.5) and lowered (0.3) probability thresholds + blind baseline
  - Task 2.6: Decision threshold sweep (tau in 0.1..0.9) saved to outputs/part2_thresholds.csv
  - Task 2.7: K-Nearest Neighbours (KNN) with 5-fold cross-validation on odd K in {1..21} optimizing mean CV F1
  - Task 2.8: Financial-cost-based threshold selection (cost = 50 * FP + 400 * FN) tuned on training data

All algorithms are implemented manually from first principles using standard library and NumPy primitives.
No scikit-learn or ready-made ML libraries/metrics are used.
"""

import os
import math
import csv
import numpy as np


def load_dataset(csv_path: str):
    """
    Reads the dataset from data/records.csv using Python's standard csv module.
    Returns:
        X_all: (N, 7) float array of predictors X1..X7
        y_all: (N,) int array of binary outcome 'escalated'
        feature_names: list of predictor column names
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at relative path: {csv_path}")

    feature_names = [
        "customer_tenure_months",      # X1
        "earlier_ticket_count_90d",    # X2
        "sla_consumed_ratio",          # X3
        "account_value_inr_thousands", # X4
        "satisfaction_score",          # X5
        "reopened_flag",               # X6
        "self_service_first_flag"      # X7
    ]

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    X_list = []
    y_list = []
    for r in rows:
        row_features = [float(r[col]) for col in feature_names]
        X_list.append(row_features)
        y_list.append(int(r["escalated"]))

    X_all = np.array(X_list, dtype=np.float64)
    y_all = np.array(y_list, dtype=np.int64)

    return X_all, y_all, feature_names


def format_value(val):
    """
    Helper to format results for output text:
      - Integers as plain integers
      - Floats rounded to exactly 4 decimal places
      - Strings (like 'yes'/'no') as strings
    """
    if isinstance(val, (int, np.integer)):
        return str(val)
    elif isinstance(val, (float, np.floating)):
        return f"{val:.4f}"
    else:
        return str(val)


def sigmoid(z):
    """
    Numerically stable sigmoid function.
    Clips z to prevent math overflow in exp(-z).
    """
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def binary_cross_entropy(y_true, p_pred):
    """
    Computes mean binary cross-entropy loss with clipped probabilities.
      L = -(1/n) * sum[ y * ln(p) + (1-y) * ln(1-p) ]
    """
    eps = 1e-15
    p_clipped = np.clip(p_pred, eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped))
    return float(loss)


def compute_confusion_and_metrics(y_true, p_pred, tau):
    """
    Computes confusion matrix and classification metrics at decision threshold tau:
      - tp: True Positives
      - fp: False Positives
      - tn: True Negatives
      - fn: False Negatives
      - accuracy: (tp + tn) / n
      - precision: tp / (tp + fp)
      - recall: tp / (tp + fn)
      - f1: 2 * precision * recall / (precision + recall)
      - specificity: tn / (tn + fp)
    Zero is returned if any metric has a zero denominator.
    """
    pred = (p_pred >= tau).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity)
    }


def knn_predict(X_train, y_train, X_query, k):
    """
    Predicts binary class labels for query points using manual Euclidean distance,
    stable neighbour sorting, and unweighted majority voting.
    """
    diff = X_query[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))

    # Stable sorting of neighbor indices
    nearest_indices = np.argsort(dists, axis=1, kind="stable")[:, :k]
    nearest_labels = y_train[nearest_indices]

    # Majority vote: predict 1 if sum of neighbor labels > k / 2
    votes = np.sum(nearest_labels, axis=1)
    predictions = (votes > (k / 2.0)).astype(int)

    return predictions


def calc_f1_score(y_true, y_pred):
    """
    Calculates F1 score given true and predicted binary labels.
    Returns 0.0 if precision + recall == 0.
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(f1)


def main():
    print("=" * 70)
    print("IIT Mandi Capstone: Swiggy Support Decision Intelligence - Part 2")
    print("=" * 70)

    # ---------------------------------------------------------
    # Setup Paths and Ingestion
    # ---------------------------------------------------------
    data_path = os.path.join("data", "records.csv")
    X_all, y_all, feature_names = load_dataset(data_path)
    results = {}

    # ---------------------------------------------------------
    # Task 2.1 — Stratified Split and Standardisation
    # ---------------------------------------------------------
    # Exact Stratified 70/30 Rule:
    # 1. Take all outcome-0 rows in original file order.
    # 2. First floor(7 * count / 10) go to training; rest to testing.
    # 3. Repeat independently for outcome-1 rows.
    # 4. Sort selected training indices into original file order.
    # 5. Sort selected test indices into original file order.
    idx0 = [i for i, label in enumerate(y_all) if label == 0]
    idx1 = [i for i, label in enumerate(y_all) if label == 1]

    n0 = len(idx0)
    n1 = len(idx1)

    n0_train = math.floor(7 * n0 / 10)
    n1_train = math.floor(7 * n1 / 10)

    train_indices = sorted(idx0[:n0_train] + idx1[:n1_train])
    test_indices = sorted(idx0[n0_train:] + idx1[n1_train:])

    train_rows = len(train_indices)
    train_positives = int(np.sum(y_all[train_indices]))
    test_rows = len(test_indices)
    test_positives = int(np.sum(y_all[test_indices]))

    results["train_rows"] = train_rows
    results["train_positives"] = train_positives
    results["test_rows"] = test_rows
    results["test_positives"] = test_positives

    print(f"\n[Task 2.1 Stratified Split & Standardisation]")
    print(f"  Training samples (train_rows)      : {train_rows}")
    print(f"  Training positives (train_positives): {train_positives}")
    print(f"  Test samples (test_rows)           : {test_rows}")
    print(f"  Test positives (test_positives)    : {test_positives}")

    # Feature subsets
    X_train_raw = X_all[train_indices]
    y_train = y_all[train_indices]
    X_test_raw = X_all[test_indices]
    y_test = y_all[test_indices]

    # Standardisation using strictly training statistics (ddof=1)
    train_mean = np.mean(X_train_raw, axis=0)
    train_sd = np.std(X_train_raw, axis=0, ddof=1)

    X_train = (X_train_raw - train_mean) / train_sd
    X_test = (X_test_raw - train_mean) / train_sd

    # ---------------------------------------------------------
    # Task 2.2 — Logistic Regression from Scratch
    # ---------------------------------------------------------
    # Initialise weights and bias to zero
    # lr = 0.1, 1000 full-batch updates on all 560 training rows
    # Gradients:
    #   dw = X.T @ (p - y) / n
    #   db = mean(p - y)
    w = np.zeros(7, dtype=np.float64)
    b = 0.0
    lr = 0.1
    n_tr = float(train_rows)

    loss_epochs = {}

    for epoch in range(1001):
        z = X_train @ w + b
        p = sigmoid(z)
        loss = binary_cross_entropy(y_train, p)

        if epoch in [0, 1, 10, 100, 1000]:
            loss_epochs[epoch] = loss

        if epoch == 1000:
            break

        # Batch gradient computation
        error = p - y_train
        dw = (X_train.T @ error) / n_tr
        db = float(np.mean(error))

        # Parameter updates
        w -= lr * dw
        b -= lr * db

    results["loss_epoch_0"] = loss_epochs[0]
    results["loss_epoch_1"] = loss_epochs[1]
    results["loss_epoch_10"] = loss_epochs[10]
    results["loss_epoch_100"] = loss_epochs[100]
    results["loss_epoch_1000"] = loss_epochs[1000]

    for j in range(7):
        results[f"weight_X{j+1}"] = float(w[j])
    results["bias"] = float(b)

    # Evaluate final model on test set
    z_test = X_test @ w + b
    p_test = sigmoid(z_test)
    test_bce = binary_cross_entropy(y_test, p_test)
    results["test_bce"] = test_bce

    print(f"\n[Task 2.2 Logistic Regression from Scratch]")
    print(f"  loss_epoch_0        : {results['loss_epoch_0']:.4f} (check: 0.6931)")
    print(f"  loss_epoch_1        : {results['loss_epoch_1']:.4f}")
    print(f"  loss_epoch_10       : {results['loss_epoch_10']:.4f}")
    print(f"  loss_epoch_100      : {results['loss_epoch_100']:.4f}")
    print(f"  loss_epoch_1000     : {results['loss_epoch_1000']:.4f}")
    for j in range(7):
        print(f"  weight_X{j+1}          : {results[f'weight_X{j+1}']:.4f}")
    print(f"  bias                : {results['bias']:.4f}")
    print(f"  test_bce            : {results['test_bce']:.4f}")

    # ---------------------------------------------------------
    # Task 2.3 — Learning-Rate Comparison
    # ---------------------------------------------------------
    # Independent runs with weights & bias re-initialised to 0
    # Exactly 200 epochs for eta in {0.01, 0.1, 1.0}
    learning_rates = [0.01, 0.1, 1.0]
    for eta in learning_rates:
        w_eta = np.zeros(7, dtype=np.float64)
        b_eta = 0.0
        for ep in range(200):
            z_curr = X_train @ w_eta + b_eta
            p_curr = sigmoid(z_curr)
            err_curr = p_curr - y_train
            dw_curr = (X_train.T @ err_curr) / n_tr
            db_curr = float(np.mean(err_curr))
            w_eta -= eta * dw_curr
            b_eta -= eta * db_curr

        p_final_eta = sigmoid(X_train @ w_eta + b_eta)
        loss_200 = binary_cross_entropy(y_train, p_final_eta)
        results[f"loss_epoch_200_eta_{eta}"] = loss_200

    print(f"\n[Task 2.3 Learning-Rate Comparison (200 Epochs)]")
    print(f"  loss_epoch_200_eta_0.01 : {results['loss_epoch_200_eta_0.01']:.4f}")
    print(f"  loss_epoch_200_eta_0.1  : {results['loss_epoch_200_eta_0.1']:.4f}")
    print(f"  loss_epoch_200_eta_1.0  : {results['loss_epoch_200_eta_1.0']:.4f}")

    # ---------------------------------------------------------
    # Task 2.4 — Odds Ratios
    # ---------------------------------------------------------
    # odds_ratio_Xj = exp(final_weight_Xj)
    for j in range(7):
        results[f"odds_ratio_X{j+1}"] = float(math.exp(w[j]))

    print(f"\n[Task 2.4 Odds Ratios]")
    for j in range(7):
        print(f"  odds_ratio_X{j+1} : {results[f'odds_ratio_X{j+1}']:.4f}")

    # ---------------------------------------------------------
    # Task 2.5 — Test Evaluation at Thresholds 0.5 and 0.3
    # ---------------------------------------------------------
    eval_thresholds = [0.5, 0.3]
    for tau in eval_thresholds:
        m_dict = compute_confusion_and_metrics(y_test, p_test, tau)
        for metric_name, metric_val in m_dict.items():
            results[f"tau_{tau}_{metric_name}"] = metric_val

    # Blind accuracy: test accuracy of a naive model predicting 0 for all test records
    blind_acc = float(np.sum(y_test == 0) / test_rows)
    results["blind_accuracy"] = blind_acc

    print(f"\n[Task 2.5 Test Evaluation]")
    print(f"  Threshold 0.5 -> Accuracy: {results['tau_0.5_accuracy']:.4f}, "
          f"Precision: {results['tau_0.5_precision']:.4f}, Recall: {results['tau_0.5_recall']:.4f}, "
          f"F1: {results['tau_0.5_f1']:.4f}, Specificity: {results['tau_0.5_specificity']:.4f}")
    print(f"  Threshold 0.3 -> Accuracy: {results['tau_0.3_accuracy']:.4f}, "
          f"Precision: {results['tau_0.3_precision']:.4f}, Recall: {results['tau_0.3_recall']:.4f}, "
          f"F1: {results['tau_0.3_f1']:.4f}, Specificity: {results['tau_0.3_specificity']:.4f}")
    print(f"  Blind Baseline Accuracy : {results['blind_accuracy']:.4f}")

    # ---------------------------------------------------------
    # Task 2.6 — Decision Threshold Sweep (outputs/part2_thresholds.csv)
    # ---------------------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    sweep_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_csv_path = os.path.join("outputs", "part2_thresholds.csv")

    thresh_rows = []
    for tau in sweep_thresholds:
        m_tau = compute_confusion_and_metrics(y_test, p_test, tau)
        tp = m_tau["tp"]
        fp = m_tau["fp"]
        tn = m_tau["tn"]
        fn = m_tau["fn"]
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        thresh_rows.append([f"{tau:.1f}", tp, fp, tn, fn, f"{tpr:.4f}", f"{fpr:.4f}"])

    with open(thresh_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"])
        writer.writerows(thresh_rows)

    print(f"\n[Task 2.6 Threshold Sweep]")
    print(f"  Saved {len(thresh_rows)} rows to {thresh_csv_path}")

    # ---------------------------------------------------------
    # Task 2.7 — K-Nearest Neighbours with 5-Fold Cross-Validation
    # ---------------------------------------------------------
    # Evaluate all 11 odd K integers from 1 through 21:
    # K = 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21
    # 5-fold cross validation on standardized training set (560 samples = 5 x 112)
    # Target metric: arithmetic mean of the five validation fold F1 scores (mean_cv_f1)
    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    n_folds = 5
    fold_size = train_rows // n_folds  # 112

    knn_cv_results = []
    knn_best_k = None
    best_cv_f1 = -1.0

    for k in k_values:
        fold_f1s = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size
            val_idx = list(range(val_start, val_end))
            tr_idx = [i for i in range(train_rows) if i not in val_idx]

            X_tr_f = X_train[tr_idx]
            y_tr_f = y_train[tr_idx]
            X_val_f = X_train[val_idx]
            y_val_f = y_train[val_idx]

            preds_val = knn_predict(X_tr_f, y_tr_f, X_val_f, k)
            f1_fold = calc_f1_score(y_val_f, preds_val)
            fold_f1s.append(f1_fold)

        mean_cv_f1 = float(np.mean(fold_f1s))
        knn_cv_results.append((k, mean_cv_f1))
        # Strictly greater ensures tie-breaking selects the smaller K
        if mean_cv_f1 > best_cv_f1:
            best_cv_f1 = mean_cv_f1
            knn_best_k = k

    # Save outputs/part2_knn_cv.csv with exact header K,mean_cv_f1
    knn_csv_path = os.path.join("outputs", "part2_knn_cv.csv")
    with open(knn_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "mean_cv_f1"])
        for k, mean_cv_f1 in knn_cv_results:
            writer.writerow([k, f"{mean_cv_f1:.4f}"])

    print(f"\n[Task 2.7 KNN with 5-Fold Cross-Validation (Metric: Mean F1 Score)]")
    for k, mean_cv_f1 in knn_cv_results:
        print(f"  K = {k:2d} -> Mean CV F1: {mean_cv_f1:.4f}")
    print(f"  Selected knn_best_k: {knn_best_k} (Mean CV F1: {best_cv_f1:.4f})")
    print(f"  Saved {len(knn_cv_results)} rows to {knn_csv_path}")

    # Evaluate KNN on test set using the selected knn_best_k
    knn_preds = knn_predict(X_train, y_train, X_test, knn_best_k)

    knn_tp = int(np.sum((knn_preds == 1) & (y_test == 1)))
    knn_fp = int(np.sum((knn_preds == 1) & (y_test == 0)))
    knn_tn = int(np.sum((knn_preds == 0) & (y_test == 0)))
    knn_fn = int(np.sum((knn_preds == 0) & (y_test == 1)))

    knn_accuracy = float((knn_tp + knn_tn) / test_rows)
    knn_precision = float(knn_tp / (knn_tp + knn_fp)) if (knn_tp + knn_fp) > 0 else 0.0
    knn_recall = float(knn_tp / (knn_tp + knn_fn)) if (knn_tp + knn_fn) > 0 else 0.0
    knn_f1 = float((2.0 * knn_precision * knn_recall) / (knn_precision + knn_recall)) if (knn_precision + knn_recall) > 0 else 0.0

    results["knn_best_k"] = knn_best_k
    results["knn_tp"] = knn_tp
    results["knn_fp"] = knn_fp
    results["knn_tn"] = knn_tn
    results["knn_fn"] = knn_fn
    results["knn_accuracy"] = knn_accuracy
    results["knn_precision"] = knn_precision
    results["knn_recall"] = knn_recall
    results["knn_f1"] = knn_f1

    print(f"\n  KNN Test Set Evaluation (knn_best_k = {knn_best_k}):")
    print(f"    TP: {knn_tp}, FP: {knn_fp}, TN: {knn_tn}, FN: {knn_fn}")
    print(f"    Accuracy: {knn_accuracy:.4f}, Precision: {knn_precision:.4f}, Recall: {knn_recall:.4f}, F1: {knn_f1:.4f}")

    # ---------------------------------------------------------
    # Task 2.8 — Financial Cost-Based Threshold Selection
    # ---------------------------------------------------------
    # Cost structure:
    #   Cost = 50 * FP + 400 * FN
    cost_fp = 50.0
    cost_fn = 400.0

    # 1. Obtain probabilities for 560 training rows
    p_train = sigmoid(X_train @ w + b)

    # 2. Evaluate candidate thresholds 0.05, 0.10, ..., 0.95 on training set
    candidate_thresholds = [round(i * 0.05, 2) for i in range(1, 20)]

    best_tau = None
    min_train_cost = float("inf")

    for tau in candidate_thresholds:
        pred_tr = (p_train >= tau).astype(int)
        fp_tr = int(np.sum((pred_tr == 1) & (y_train == 0)))
        fn_tr = int(np.sum((pred_tr == 0) & (y_train == 1)))
        cost_tr = fp_tr * cost_fp + fn_tr * cost_fn

        # Strictly less than ensures ties select the smaller threshold
        if cost_tr < min_train_cost:
            min_train_cost = cost_tr
            best_tau = tau

    # 3. Write cost_tau and cost_train_at_tau
    results["cost_tau"] = float(best_tau)
    results["cost_train_at_tau"] = int(min_train_cost)

    # 4. Apply selected threshold to the 240 test probabilities
    pred_te_tau = (p_test >= best_tau).astype(int)
    fp_te_tau = int(np.sum((pred_te_tau == 1) & (y_test == 0)))
    fn_te_tau = int(np.sum((pred_te_tau == 0) & (y_test == 1)))
    cost_test_at_tau = int(fp_te_tau * cost_fp + fn_te_tau * cost_fn)
    results["cost_test_at_tau"] = cost_test_at_tau

    # 5. Write cost_test_at_0.5
    pred_te_05 = (p_test >= 0.5).astype(int)
    fp_te_05 = int(np.sum((pred_te_05 == 1) & (y_test == 0)))
    fn_te_05 = int(np.sum((pred_te_05 == 0) & (y_test == 1)))
    cost_test_at_05 = int(fp_te_05 * cost_fp + fn_te_05 * cost_fn)
    results["cost_test_at_0.5"] = cost_test_at_05

    # 6. Write cost_test_flag_nobody (every test record predicted negative)
    # FP = 0, FN = total actual positives in test set = 51
    fn_nobody = int(np.sum(y_test == 1))
    cost_test_flag_nobody = int(fn_nobody * cost_fn)
    results["cost_test_flag_nobody"] = cost_test_flag_nobody

    print(f"\n[Task 2.8 Financial-Cost-Based Threshold Selection]")
    print(f"  Cost Formula            : Total Cost = ₹50 * FP + ₹400 * FN")
    print(f"  cost_tau (Selected)     : {results['cost_tau']:.4f}")
    print(f"  cost_train_at_tau       : ₹{results['cost_train_at_tau']}")
    print(f"  cost_test_at_tau        : ₹{results['cost_test_at_tau']}")
    print(f"  cost_test_at_0.5        : ₹{results['cost_test_at_0.5']}")
    print(f"  cost_test_flag_nobody   : ₹{results['cost_test_flag_nobody']}")
    savings_vs_default = ((cost_test_at_05 - cost_test_at_tau) / cost_test_at_05) * 100
    savings_vs_nobody = ((cost_test_flag_nobody - cost_test_at_tau) / cost_test_flag_nobody) * 100
    print(f"  Savings vs Default (0.5): {savings_vs_default:.2f}%")
    print(f"  Savings vs Flag Nobody  : {savings_vs_nobody:.2f}%")

    # ---------------------------------------------------------
    # Verification & Checkpoints
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERIFICATION CHECKPOINTS")
    print("=" * 70)
    assert results["train_rows"] == 560, f"Expected train_rows=560, got {results['train_rows']}"
    assert results["train_positives"] == 119, f"Expected train_positives=119, got {results['train_positives']}"
    assert results["test_rows"] == 240, f"Expected test_rows=240, got {results['test_rows']}"
    assert results["test_positives"] == 51, f"Expected test_positives=51, got {results['test_positives']}"
    assert abs(results["loss_epoch_0"] - 0.6931) < 0.001, f"Expected loss_epoch_0=0.6931, got {results['loss_epoch_0']:.4f}"
    assert len(thresh_rows) == 9, f"Expected 9 threshold rows, got {len(thresh_rows)}"
    assert len(knn_cv_results) == 11, f"Expected 11 KNN CV rows, got {len(knn_cv_results)}"
    assert results["cost_tau"] == 0.15, f"Expected cost_tau=0.15, got {results['cost_tau']}"
    assert results["cost_train_at_tau"] == 12150, f"Expected cost_train_at_tau=12150, got {results['cost_train_at_tau']}"
    assert results["cost_test_at_tau"] == 6050, f"Expected cost_test_at_tau=6050, got {results['cost_test_at_tau']}"
    assert results["cost_test_at_0.5"] == 11800, f"Expected cost_test_at_0.5=11800, got {results['cost_test_at_0.5']}"
    assert results["cost_test_flag_nobody"] == 20400, f"Expected cost_test_flag_nobody=20400, got {results['cost_test_flag_nobody']}"
    print("✓ Checkpoint 1: train_rows = 560 (Passed)")
    print("✓ Checkpoint 2: train_positives = 119 (Passed)")
    print("✓ Checkpoint 3: test_rows = 240 (Passed)")
    print("✓ Checkpoint 4: test_positives = 51 (Passed)")
    print("✓ Checkpoint 5: loss_epoch_0 = 0.6931 (Passed)")
    print("✓ Checkpoint 6: outputs/part2_thresholds.csv has 9 data rows (Passed)")
    print("✓ Checkpoint 7: outputs/part2_knn_cv.csv has 11 data rows with header K,mean_cv_f1 (Passed)")
    print("✓ Checkpoint 8: All Task 2.8 financial cost keys verified (Passed)")

    # ---------------------------------------------------------
    # Save outputs/part2_results.txt
    # ---------------------------------------------------------
    output_txt_path = os.path.join("outputs", "part2_results.txt")

    required_keys = [
        # Task 2.1
        "train_rows", "train_positives", "test_rows", "test_positives",
        # Task 2.2
        "loss_epoch_0", "loss_epoch_1", "loss_epoch_10", "loss_epoch_100", "loss_epoch_1000",
        "weight_X1", "weight_X2", "weight_X3", "weight_X4", "weight_X5", "weight_X6", "weight_X7",
        "bias", "test_bce",
        # Task 2.3
        "loss_epoch_200_eta_0.01", "loss_epoch_200_eta_0.1", "loss_epoch_200_eta_1.0",
        # Task 2.4
        "odds_ratio_X1", "odds_ratio_X2", "odds_ratio_X3", "odds_ratio_X4",
        "odds_ratio_X5", "odds_ratio_X6", "odds_ratio_X7",
        # Task 2.5
        "tau_0.5_tp", "tau_0.5_fp", "tau_0.5_tn", "tau_0.5_fn",
        "tau_0.5_accuracy", "tau_0.5_precision", "tau_0.5_recall", "tau_0.5_f1", "tau_0.5_specificity",
        "tau_0.3_tp", "tau_0.3_fp", "tau_0.3_tn", "tau_0.3_fn",
        "tau_0.3_accuracy", "tau_0.3_precision", "tau_0.3_recall", "tau_0.3_f1", "tau_0.3_specificity",
        "blind_accuracy",
        # Task 2.7 (KNN with knn_best_k)
        "knn_best_k", "knn_tp", "knn_fp", "knn_tn", "knn_fn",
        "knn_accuracy", "knn_precision", "knn_recall", "knn_f1",
        # Task 2.8 (Financial Cost-Based Optimization - 5 Exact Keys)
        "cost_tau", "cost_train_at_tau", "cost_test_at_tau", "cost_test_at_0.5", "cost_test_flag_nobody"
    ]

    with open(output_txt_path, mode="w", encoding="utf-8") as f:
        for k in required_keys:
            val_str = format_value(results[k])
            f.write(f"{k} = {val_str}\n")

    print(f"\nAll {len(required_keys)} results successfully saved to: {output_txt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
