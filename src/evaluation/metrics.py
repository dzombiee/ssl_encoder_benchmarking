"""
Evaluation metrics for recommendation.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def recall_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
    """
    Recall@K: Fraction of relevant items in top-K predictions.

    Args:
        predictions: Ranked list of item IDs
        ground_truth: List of relevant item IDs
        k: Cutoff position

    Returns:
        recall: Recall@K score
    """
    if len(ground_truth) == 0:
        return 0.0

    top_k = predictions[:k]
    relevant_in_top_k = len(set(top_k) & set(ground_truth))

    recall = relevant_in_top_k / len(ground_truth)

    return recall


def ndcg_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain at K.

    Args:
        predictions: Ranked list of item IDs
        ground_truth: List of relevant item IDs
        k: Cutoff position

    Returns:
        ndcg: NDCG@K score
    """
    if len(ground_truth) == 0:
        return 0.0

    # DCG: sum of (relevance / log2(position + 1))
    dcg = 0.0
    for i, item_id in enumerate(predictions[:k]):
        if item_id in ground_truth:
            # Relevance is 1 for relevant items
            dcg += 1.0 / np.log2(i + 2)  # +2 because position is 0-indexed

    # IDCG: DCG of perfect ranking
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))

    if idcg == 0:
        return 0.0

    ndcg = dcg / idcg

    return ndcg


def precision_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
    """
    Precision@K: Fraction of top-K predictions that are relevant.

    Args:
        predictions: Ranked list of item IDs
        ground_truth: List of relevant item IDs
        k: Cutoff position

    Returns:
        precision: Precision@K score
    """
    if k == 0:
        return 0.0

    top_k = predictions[:k]
    relevant_in_top_k = len(set(top_k) & set(ground_truth))

    precision = relevant_in_top_k / k

    return precision


def hit_rate_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
    """
    Hit@K: Whether any relevant item appears in top-K.

    Args:
        predictions: Ranked list of item IDs
        ground_truth: List of relevant item IDs
        k: Cutoff position

    Returns:
        hit: 1.0 if hit, 0.0 otherwise
    """
    top_k = predictions[:k]
    hit = 1.0 if len(set(top_k) & set(ground_truth)) > 0 else 0.0

    return hit


def mean_reciprocal_rank(predictions: List[str], ground_truth: List[str]) -> float:
    """
    MRR: Mean Reciprocal Rank.

    Args:
        predictions: Ranked list of item IDs
        ground_truth: List of relevant item IDs

    Returns:
        mrr: MRR score
    """
    for i, item_id in enumerate(predictions):
        if item_id in ground_truth:
            return 1.0 / (i + 1)

    return 0.0


def compute_all_metrics(
    predictions: List[str], ground_truth: List[str], k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Compute all metrics for a single user.

    Args:
        predictions: Ranked list of item IDs
        ground_truth: List of relevant item IDs
        k_values: List of K values for metrics

    Returns:
        metrics: Dictionary of metric values
    """
    metrics = {}

    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(predictions, ground_truth, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(predictions, ground_truth, k)
        metrics[f"precision@{k}"] = precision_at_k(predictions, ground_truth, k)
        metrics[f"hit@{k}"] = hit_rate_at_k(predictions, ground_truth, k)

    metrics["mrr"] = mean_reciprocal_rank(predictions, ground_truth)

    return metrics


def aggregate_metrics(
    all_metrics: List[Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Aggregate metrics across multiple users.

    Args:
        all_metrics: List of metric dictionaries (one per user)

    Returns:
        mean_metrics: Mean of each metric
        std_metrics: Standard deviation of each metric
    """
    if not all_metrics:
        return {}, {}

    # Collect values for each metric
    metric_values = defaultdict(list)
    for user_metrics in all_metrics:
        for metric_name, value in user_metrics.items():
            metric_values[metric_name].append(value)

    # Compute mean and std
    mean_metrics = {}
    std_metrics = {}

    for metric_name, values in metric_values.items():
        mean_metrics[metric_name] = np.mean(values)
        std_metrics[metric_name] = np.std(values)

    return mean_metrics, std_metrics


def print_metrics(
    mean_metrics: Dict[str, float],
    std_metrics: Dict[str, float],
    title: str = "Metrics",
):
    """
    Pretty print metrics.

    Args:
        mean_metrics: Mean values
        std_metrics: Standard deviation values
        title: Title for the metrics
    """
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)

    # Group by metric type
    metric_groups = {"Recall": [], "NDCG": [], "Precision": [], "Hit": [], "Other": []}

    for metric_name in sorted(mean_metrics.keys()):
        if "recall" in metric_name:
            metric_groups["Recall"].append(metric_name)
        elif "ndcg" in metric_name:
            metric_groups["NDCG"].append(metric_name)
        elif "precision" in metric_name:
            metric_groups["Precision"].append(metric_name)
        elif "hit" in metric_name:
            metric_groups["Hit"].append(metric_name)
        else:
            metric_groups["Other"].append(metric_name)

    for group_name, metrics in metric_groups.items():
        if metrics:
            print(f"\n{group_name}:")
            for metric_name in metrics:
                mean = mean_metrics[metric_name]
                std = std_metrics[metric_name]
                print(f"  {metric_name:15s}: {mean:.4f} ± {std:.4f}")

    print("=" * 60)
