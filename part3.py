"""
IIT Mandi - Minor in AI & Data Science
Capstone Project: Swiggy - Support Decision Intelligence
Part 3: Unsupervised Learning, Clustering, and Dimensionality Reduction

Author: Student
Dataset: data/records.csv
Outputs:
  - outputs/part3_results.txt
  - outputs/part3_segments.csv
  - outputs/part3_hierarchical.csv

This script implements:
  - Task 3.1: K-Means from scratch (K in {2..6}) on standardized features X1–X5
  - Task 3.2: Elbow method for optimal K selection
  - Task 3.3: Profiling three segments (K=3) ordered by ascending mean of X3 -> A, B, C
  - Task 3.4: Hierarchical clustering from scratch (single, complete, average linkage) on E1–E6
  - Task 3.5: Principal Component Analysis (PCA) on standardized 800x5 matrix

All algorithms are implemented manually from first principles using standard library and NumPy primitives.
No scikit-learn or ready-made clustering/PCA functions are used.
"""

import os
import math
import csv
import numpy as np


def load_dataset(csv_path: str):
    """
    Reads the dataset from data/records.csv using Python's standard csv module.
    Returns:
        X_5: (N, 5) float array of predictors X1..X5
        y_outcome: (N,) int array of binary outcome 'escalated'
        y_value: (N,) float array of continuous target 'resolution_minutes'
        feature_names: list of feature names X1..X5
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at relative path: {csv_path}")

    feature_names = [
        "customer_tenure_months",      # X1
        "earlier_ticket_count_90d",    # X2
        "sla_consumed_ratio",          # X3
        "account_value_inr_thousands", # X4
        "satisfaction_score"           # X5
    ]

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    X_list = []
    y_outcome_list = []
    y_value_list = []

    for r in rows:
        row_feat = [float(r[col]) for col in feature_names]
        X_list.append(row_feat)
        y_outcome_list.append(int(r["escalated"]))
        y_value_list.append(float(r["resolution_minutes"]))

    X_5 = np.array(X_list, dtype=np.float64)
    y_outcome = np.array(y_outcome_list, dtype=np.int64)
    y_value = np.array(y_value_list, dtype=np.float64)

    return X_5, y_outcome, y_value, feature_names


def format_value(val):
    """
    Helper to format results for output text:
      - Integers as plain integers
      - Floats rounded to exactly 4 decimal places
      - Strings as strings
    """
    if isinstance(val, (int, np.integer)):
        return str(val)
    elif isinstance(val, (float, np.floating)):
        return f"{val:.4f}"
    else:
        return str(val)


def run_kmeans(X, K):
    """
    K-Means clustering from scratch:
      - Initial centroids: row indices 0, floor(N/K), 2*floor(N/K), ..., (K-1)*floor(N/K)
      - Assign each row to nearest centroid using squared Euclidean distance.
      - Ties go to lower-numbered centroid.
      - Update each centroid to the mean of its assigned members.
      - Empty centroid remains unchanged.
      - Stop at first pass where no row changes cluster.
    Returns:
        final_centroids: (K, D) array
        final_assignments: (N,) array
        wcss: float, sum of squared distances to final centroids
    """
    N = len(X)
    step = math.floor(N / K)
    init_indices = [i * step for i in range(K)]
    centroids = X[init_indices].copy()

    prev_assignments = None

    while True:
        # Squared Euclidean distance from each row to each centroid: (N, K)
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)

        # Ties go to lower-numbered centroid: np.argmin returns first minimal index
        assignments = np.argmin(sq_dists, axis=1)

        if prev_assignments is not None and np.array_equal(assignments, prev_assignments):
            break

        prev_assignments = assignments.copy()

        # Update centroids to mean of assigned members
        for k in range(K):
            members = X[assignments == k]
            if len(members) > 0:
                centroids[k] = np.mean(members, axis=0)

    # Calculate WCSS in standardized space
    diff_final = X - centroids[assignments]
    wcss = float(np.sum(diff_final ** 2))

    return centroids, assignments, wcss


def run_hierarchical(dist_matrix, elements, linkage_type):
    """
    Agglomerative Hierarchical Clustering from scratch for E1..E6:
      - single: minimum pairwise distance
      - complete: maximum pairwise distance
      - average: mean pairwise distance
    At each step, merges the pair of clusters with smallest linkage distance.
    Returns list of merge dictionaries.
    """
    n_elements = len(elements)
    clusters = [[i] for i in range(n_elements)]
    merges = []

    for merge_no in range(1, n_elements):
        min_dist = float("inf")
        best_pair = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c1 = clusters[i]
                c2 = clusters[j]
                pair_dists = [dist_matrix[p1, p2] for p1 in c1 for p2 in c2]

                if linkage_type == "single":
                    d = min(pair_dists)
                elif linkage_type == "complete":
                    d = max(pair_dists)
                elif linkage_type == "average":
                    d = sum(pair_dists) / len(pair_dists)

                if d < min_dist:
                    min_dist = d
                    best_pair = (i, j)

        i, j = best_pair
        c1, c2 = clusters[i], clusters[j]

        # cluster_a must be the cluster whose smallest member number is smaller
        min_m1 = min(c1)
        min_m2 = min(c2)
        if min_m1 < min_m2:
            cA, cB = c1, c2
        else:
            cA, cB = c2, c1

        nameA = "+".join(sorted([elements[m] for m in cA]))
        nameB = "+".join(sorted([elements[m] for m in cB]))

        merges.append({
            "linkage": linkage_type,
            "merge_no": merge_no,
            "cluster_a": nameA,
            "cluster_b": nameB,
            "height": float(min_dist)
        })

        new_cluster = sorted(c1 + c2)
        clusters = [c for idx, c in enumerate(clusters) if idx not in (i, j)]
        clusters.append(new_cluster)

    return merges


def main():
    print("=" * 70)
    print("IIT Mandi Capstone: Swiggy Support Decision Intelligence - Part 3")
    print("=" * 70)

    # ---------------------------------------------------------
    # Setup Paths and Data Ingestion
    # ---------------------------------------------------------
    data_path = os.path.join("data", "records.csv")
    X_raw, y_outcome, y_value, feature_names = load_dataset(data_path)
    N = len(X_raw)
    results = {}

    # ---------------------------------------------------------
    # Task 3.1 — K-Means from Scratch
    # ---------------------------------------------------------
    # Standardise each column using its mean and sample standard deviation (ddof=1) over all 800 rows
    train_means = np.mean(X_raw, axis=0)
    train_sds = np.std(X_raw, axis=0, ddof=1)
    X_std = (X_raw - train_means) / train_sds

    kmeans_models = {}
    print(f"\n[Task 3.1 K-Means from Scratch (K in 2..6)]")
    for K in [2, 3, 4, 5, 6]:
        centroids, assignments, wcss = run_kmeans(X_std, K)
        kmeans_models[K] = {
            "centroids": centroids,
            "assignments": assignments,
            "wcss": wcss
        }
        results[f"wcss_k{K}"] = wcss
        print(f"  K = {K} -> WCSS: {wcss:.4f}")

    # Checkpoint validation
    assert abs(results["wcss_k2"] - 2262.6753) < 0.001, (
        f"Expected wcss_k2=2262.6753, got {results['wcss_k2']:.4f}"
    )
    print(f"  ✓ Checkpoint verified: wcss_k2 = {results['wcss_k2']:.4f}")

    # ---------------------------------------------------------
    # Task 3.2 — Elbow Method
    # ---------------------------------------------------------
    # Smallest K in {3, 4, 5} for which:
    #   WCSS(K) - WCSS(K+1) < 0.5 * (WCSS(K-1) - WCSS(K))
    elbow_k = None
    for K in [3, 4, 5]:
        drop_curr = kmeans_models[K]["wcss"] - kmeans_models[K + 1]["wcss"]
        drop_prev = kmeans_models[K - 1]["wcss"] - kmeans_models[K]["wcss"]
        if drop_curr < 0.5 * drop_prev:
            elbow_k = K
            break

    results["elbow_k"] = elbow_k
    print(f"\n[Task 3.2 Elbow Method]")
    print(f"  Identified elbow_k: {elbow_k}")

    # ---------------------------------------------------------
    # Task 3.3 — Profile Three Segments (K = 3)
    # ---------------------------------------------------------
    # Regardless of elbow_k, use K=3 assignments
    k3_assignments = kmeans_models[3]["assignments"]

    # Order clusters by ascending original (unstandardised) mean of X3 (sla_consumed_ratio, index 2)
    # Name them A, B, C
    cluster_x3_means = []
    for k in range(3):
        m_x3 = float(np.mean(X_raw[k3_assignments == k, 2]))
        cluster_x3_means.append((m_x3, k))

    cluster_x3_means.sort(key=lambda item: item[0])
    ordered_cluster_ids = [item[1] for item in cluster_x3_means]
    segment_names = ["A", "B", "C"]

    segments_table = []
    for seg_name, cid in zip(segment_names, ordered_cluster_ids):
        mask = (k3_assignments == cid)
        size = int(np.sum(mask))
        results[f"size_segment_{seg_name}"] = size

        m_x1 = float(np.mean(X_raw[mask, 0]))
        m_x2 = float(np.mean(X_raw[mask, 1]))
        m_x3 = float(np.mean(X_raw[mask, 2]))
        m_x4 = float(np.mean(X_raw[mask, 3]))
        m_x5 = float(np.mean(X_raw[mask, 4]))
        outcome_rate = float(np.mean(y_outcome[mask]))
        mean_val = float(np.mean(y_value[mask]))

        segments_table.append([
            seg_name,
            size,
            f"{m_x1:.4f}",
            f"{m_x2:.4f}",
            f"{m_x3:.4f}",
            f"{m_x4:.4f}",
            f"{m_x5:.4f}",
            f"{outcome_rate:.4f}",
            f"{mean_val:.4f}"
        ])

    # Save outputs/part3_segments.csv
    os.makedirs("outputs", exist_ok=True)
    segments_csv_path = os.path.join("outputs", "part3_segments.csv")
    with open(segments_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"])
        writer.writerows(segments_table)

    print(f"\n[Task 3.3 Profile Three Segments]")
    for row in segments_table:
        print(f"  Segment {row[0]}: size={row[1]}, mean_X3={row[4]}, outcome_rate={row[7]}, mean_value={row[8]}")
    print(f"  Saved 3 rows to {segments_csv_path}")

    # ---------------------------------------------------------
    # Task 3.4 — Hierarchical Clustering
    # ---------------------------------------------------------
    elements = ["E1", "E2", "E3", "E4", "E5", "E6"]
    dist_matrix = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
    ], dtype=np.float64)

    linkage_types = ["single", "complete", "average"]
    hierarchical_rows = []

    for link in linkage_types:
        merges = run_hierarchical(dist_matrix, elements, link)
        heights = [m["height"] for m in merges]

        for m in merges:
            hierarchical_rows.append([
                m["linkage"],
                m["merge_no"],
                m["cluster_a"],
                m["cluster_b"],
                f"{m['height']:.4f}"
            ])

        # Calculate the 4 gaps between consecutive merge heights:
        gaps = [heights[i + 1] - heights[i] for i in range(4)]
        max_gap_idx = int(np.argmax(gaps))  # 0-based index of merge below cut
        clusters_at_cut = 6 - (max_gap_idx + 1)
        results[f"{link}_clusters_at_cut"] = clusters_at_cut

    # Save outputs/part3_hierarchical.csv
    hierarchical_csv_path = os.path.join("outputs", "part3_hierarchical.csv")
    with open(hierarchical_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["linkage", "merge_no", "cluster_a", "cluster_b", "height"])
        writer.writerows(hierarchical_rows)

    print(f"\n[Task 3.4 Hierarchical Clustering]")
    print(f"  single_clusters_at_cut   : {results['single_clusters_at_cut']}")
    print(f"  complete_clusters_at_cut : {results['complete_clusters_at_cut']}")
    print(f"  average_clusters_at_cut  : {results['average_clusters_at_cut']}")
    print(f"  Saved 15 rows to {hierarchical_csv_path}")

    # ---------------------------------------------------------
    # Task 3.5 — Principal Component Analysis (PCA)
    # ---------------------------------------------------------
    # Use standardized 800x5 matrix X_std from Task 3.1
    # Sample covariance matrix C with divisor N - 1:
    C = (1.0 / (N - 1)) * (X_std.T @ X_std)

    # Eigenvalues and eigenvectors using numpy.linalg.eigh (symmetric)
    eigvals, eigvecs = np.linalg.eigh(C)

    # Sort in descending eigenvalue order
    sort_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    # Sign convention: the entry with the largest absolute value in each eigenvector must be positive.
    for j in range(5):
        v = eigvecs[:, j]
        max_idx = int(np.argmax(np.abs(v)))
        if v[max_idx] < 0:
            eigvecs[:, j] = -v

    eigenvalue_sum = float(np.sum(eigvals))
    results["eigenvalue_sum"] = eigenvalue_sum

    for j in range(5):
        results[f"eigenvalue_{j+1}"] = float(eigvals[j])

    var_ratios = eigvals / eigenvalue_sum
    cum_var_ratios = np.cumsum(var_ratios)

    for j in range(5):
        results[f"explained_ratio_{j+1}"] = float(var_ratios[j])
        results[f"cumulative_ratio_{j+1}"] = float(cum_var_ratios[j])

    # Minimum components to explain at least 80% variance
    components_for_80pct = int(np.argmax(cum_var_ratios >= 0.80) + 1)
    results["components_for_80pct"] = components_for_80pct

    # Loadings for PC1 and PC2
    for i in range(5):
        results[f"pc1_loading_X{i+1}"] = float(eigvecs[i, 0])
        results[f"pc2_loading_X{i+1}"] = float(eigvecs[i, 1])

    # Projections of rows 1, 2, 3 (indices 0, 1, 2)
    Z = X_std @ eigvecs
    results["row1_pc1"] = float(Z[0, 0])
    results["row1_pc2"] = float(Z[0, 1])
    results["row2_pc1"] = float(Z[1, 0])
    results["row2_pc2"] = float(Z[1, 1])
    results["row3_pc1"] = float(Z[2, 0])
    results["row3_pc2"] = float(Z[2, 1])

    # Mean PC1 projection for each segment (A, B, C)
    for seg_name, cid in zip(segment_names, ordered_cluster_ids):
        mask = (k3_assignments == cid)
        results[f"mean_pc1_segment_{seg_name}"] = float(np.mean(Z[mask, 0]))

    print(f"\n[Task 3.5 Principal Component Analysis (PCA)]")
    print(f"  eigenvalue_sum       : {results['eigenvalue_sum']:.4f}")
    for j in range(5):
        print(f"  PC{j+1}: Eigenvalue = {results[f'eigenvalue_{j+1}']:.4f}, Explained = {results[f'explained_ratio_{j+1}']:.4f}, Cumulative = {results[f'cumulative_ratio_{j+1}']:.4f}")
    print(f"  components_for_80pct : {results['components_for_80pct']}")
    pc1_load_str = [f"{results['pc1_loading_X' + str(i+1)]:.4f}" for i in range(5)]
    pc2_load_str = [f"{results['pc2_loading_X' + str(i+1)]:.4f}" for i in range(5)]
    print(f"  PC1 loadings         : {pc1_load_str}")
    print(f"  PC2 loadings         : {pc2_load_str}")
    print(f"  Row 1 (PC1, PC2)     : ({results['row1_pc1']:.4f}, {results['row1_pc2']:.4f})")
    print(f"  Row 2 (PC1, PC2)     : ({results['row2_pc1']:.4f}, {results['row2_pc2']:.4f})")
    print(f"  Row 3 (PC1, PC2)     : ({results['row3_pc1']:.4f}, {results['row3_pc2']:.4f})")
    print(f"  mean_pc1_segment_A   : {results['mean_pc1_segment_A']:.4f}")
    print(f"  mean_pc1_segment_B   : {results['mean_pc1_segment_B']:.4f}")
    print(f"  mean_pc1_segment_C   : {results['mean_pc1_segment_C']:.4f}")

    # ---------------------------------------------------------
    # Verification & Checkpoints
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERIFICATION CHECKPOINTS")
    print("=" * 70)
    assert abs(results["wcss_k2"] - 2262.6753) < 0.001
    assert abs(results["eigenvalue_sum"] - 5.0) < 0.0001
    assert results["components_for_80pct"] == 3
    assert (results["size_segment_A"] + results["size_segment_B"] + results["size_segment_C"]) == 800
    assert len(segments_table) == 3
    assert len(hierarchical_rows) == 15
    print("✓ Checkpoint 1: wcss_k2 = 2262.6753 (Passed)")
    print("✓ Checkpoint 2: elbow_k calculated (Passed)")
    print("✓ Checkpoint 3: outputs/part3_segments.csv has 3 rows (Passed)")
    print("✓ Checkpoint 4: outputs/part3_hierarchical.csv has 15 rows (Passed)")
    print("✓ Checkpoint 5: eigenvalue_sum = 5.0000 (Passed)")
    print("✓ Checkpoint 6: components_for_80pct = 3 (Passed)")
    print("✓ Checkpoint 7: Segment sizes sum to 800 (Passed)")

    # ---------------------------------------------------------
    # Save outputs/part3_results.txt
    # ---------------------------------------------------------
    output_txt_path = os.path.join("outputs", "part3_results.txt")

    required_keys = [
        # Task 3.1
        "wcss_k2", "wcss_k3", "wcss_k4", "wcss_k5", "wcss_k6",
        # Task 3.2
        "elbow_k",
        # Task 3.3
        "size_segment_A", "size_segment_B", "size_segment_C",
        # Task 3.4
        "single_clusters_at_cut", "complete_clusters_at_cut", "average_clusters_at_cut",
        # Task 3.5
        "eigenvalue_sum",
        "eigenvalue_1", "eigenvalue_2", "eigenvalue_3", "eigenvalue_4", "eigenvalue_5",
        "explained_ratio_1", "explained_ratio_2", "explained_ratio_3", "explained_ratio_4", "explained_ratio_5",
        "cumulative_ratio_1", "cumulative_ratio_2", "cumulative_ratio_3", "cumulative_ratio_4", "cumulative_ratio_5",
        "components_for_80pct",
        "pc1_loading_X1", "pc1_loading_X2", "pc1_loading_X3", "pc1_loading_X4", "pc1_loading_X5",
        "pc2_loading_X1", "pc2_loading_X2", "pc2_loading_X3", "pc2_loading_X4", "pc2_loading_X5",
        "row1_pc1", "row1_pc2",
        "row2_pc1", "row2_pc2",
        "row3_pc1", "row3_pc2",
        "mean_pc1_segment_A", "mean_pc1_segment_B", "mean_pc1_segment_C"
    ]

    with open(output_txt_path, mode="w", encoding="utf-8") as f:
        for k in required_keys:
            val_str = format_value(results[k])
            f.write(f"{k} = {val_str}\n")

    print(f"\nAll {len(required_keys)} results successfully saved to: {output_txt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

