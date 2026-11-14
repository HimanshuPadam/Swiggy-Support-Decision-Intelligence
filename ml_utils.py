"""
Shared machine learning utilities for IIT Mandi Capstone: Swiggy Support Decision Intelligence.
Shared between Part 2 and Part 4.
This module defines pure functions and contains no top-level execution code.
"""

import os
import math
import csv
import numpy as np


FEATURE_NAMES = [
    "customer_tenure_months",      # X1
    "earlier_ticket_count_90d",    # X2
    "sla_consumed_ratio",          # X3
    "account_value_inr_thousands", # X4
    "satisfaction_score",          # X5
    "reopened_flag",               # X6
    "self_service_first_flag"      # X7
]


def load_dataset(csv_path: str):
    """
    Reads the dataset from data/records.csv using Python's standard csv module.
    Returns:
        X_all: (N, 7) float array of predictors X1..X7
        y_all: (N,) int array of binary outcome 'escalated'
        feature_names: list of feature names
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at relative path: {csv_path}")

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    X_list = []
    y_list = []

    for r in rows:
        row_feat = [float(r[col]) for col in FEATURE_NAMES]
        X_list.append(row_feat)
        y_list.append(int(r["escalated"]))

    X_all = np.array(X_list, dtype=np.float64)
    y_all = np.array(y_list, dtype=np.int64)

    return X_all, y_all, FEATURE_NAMES


def stratified_split(y_all):
    """
    Exact Stratified 70/30 Rule from Part 2 Task 2.1:
    1. Take all outcome-0 rows in original file order.
    2. First floor(7 * count / 10) go to training; rest to testing.
    3. Repeat independently for outcome-1 rows.
    4. Sort selected training indices into original file order.
    5. Sort selected test indices into original file order.
    Returns:
        train_indices: list of ints
        test_indices: list of ints
    """
    idx0 = [i for i, label in enumerate(y_all) if label == 0]
    idx1 = [i for i, label in enumerate(y_all) if label == 1]

    n0 = len(idx0)
    n1 = len(idx1)

    n0_train = math.floor(7 * n0 / 10)
    n1_train = math.floor(7 * n1 / 10)

    train_indices = sorted(idx0[:n0_train] + idx1[:n1_train])
    test_indices = sorted(idx0[n0_train:] + idx1[n1_train:])

    return train_indices, test_indices


def standardise_features(X_train_raw, X_test_raw):
    """
    Standardises predictors strictly using training set statistics.
    Uses sample standard deviation (ddof=1).
    Returns:
        X_train: (N_train, 7)
        X_test: (N_test, 7)
        train_mean: (7,)
        train_sd: (7,)
    """
    train_mean = np.mean(X_train_raw, axis=0)
    train_sd = np.std(X_train_raw, axis=0, ddof=1)

    X_train = (X_train_raw - train_mean) / train_sd
    X_test = (X_test_raw - train_mean) / train_sd

    return X_train, X_test, train_mean, train_sd


def sigmoid(z):
    """Numerically stable sigmoid function."""
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def binary_cross_entropy(y_true, p_pred):
    """Calculates binary cross-entropy loss."""
    eps = 1e-15
    p_clipped = np.clip(p_pred, eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped))
    return float(loss)


def train_logistic_regression(X_train, y_train, lr=0.1, epochs=1000):
    """
    Trains logistic regression model using full-batch gradient descent.
    Exact configuration from Part 2 Task 2.2:
      - Initialise weights and bias to 0
      - Learning rate = 0.1
      - 1000 full-batch updates
    Returns:
        w: (7,) float array
        b: float
    """
    n_tr = float(len(X_train))
    w = np.zeros(X_train.shape[1], dtype=np.float64)
    b = 0.0

    for epoch in range(epochs):
        z = X_train @ w + b
        p = sigmoid(z)
        dw = (X_train.T @ (p - y_train)) / n_tr
        db = float(np.mean(p - y_train))
        w -= lr * dw
        b -= lr * db

    return w, b


def predict_probabilities(X, w, b):
    """Computes predicted probabilities using logistic model."""
    z = X @ w + b
    return sigmoid(z)


def knn_predict(X_train, y_train, X_query, k):
    """
    K-Nearest Neighbours classifier using squared Euclidean distance.
    Tie-breaking in distance uses stable sorting (preserves earlier training row index).
    Majority voting predicts 1 if strictly more than half neighbours are positive.
    """
    diff = X_query[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)

    nn_indices = np.argsort(dists, axis=1, kind="stable")[:, :k]
    nn_labels = y_train[nn_indices]

    ones_count = np.sum(nn_labels == 1, axis=1)
    zeros_count = k - ones_count

    return np.where(ones_count > zeros_count, 1, 0)


def calc_f1_score(y_true, y_pred):
    """Calculates F1 score manually."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    denom = (2.0 * tp + fp + fn)
    if denom == 0:
        return 0.0
    return float((2.0 * tp) / denom)


def format_value(val):
    """Formats output values: integers as integers, floats to 4 decimal places."""
    if isinstance(val, (int, np.integer)):
        return str(val)
    elif isinstance(val, (float, np.floating)):
        return f"{val:.4f}"
    else:
        return str(val)

