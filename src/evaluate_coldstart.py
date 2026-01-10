"""
Evaluation script for cold-start recommendation.
"""

import json
import argparse
from pathlib import Path
from typing import Dict
from tqdm import tqdm  # type: ignore

from evaluation.metrics import compute_all_metrics, aggregate_metrics, print_metrics
from evaluation.zero_shot_recommender import (
    ZeroShotDotProductRecommender,
    load_user_histories,
    compute_item_popularity,
    sample_negatives,
)
from utils.helpers import load_config, load_embeddings, save_json


def evaluate_cold_start(
    embeddings_path: str,
    train_interactions_path: str,
    test_interactions_path: str,
    cold_items_path: str,
    warm_items_path: str,
    output_dir: str,
    config: Dict,
    model_name: str = "model",
):
    """
    Evaluate zero-shot recommendation on cold-start items.

    Args:
        embeddings_path: Path to precomputed item embeddings (.npz file)
        train_interactions_path: Path to training interactions
        test_interactions_path: Path to test interactions
        cold_items_path: Path to list of cold items
        warm_items_path: Path to list of warm items
        output_dir: Output directory for results
        config: Configuration dictionary
        model_name: Name of the model being evaluated
    """
    print(f"\n{'=' * 80}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 80}\n")

    # Load embeddings
    print("Loading embeddings...")
    item_embeddings = load_embeddings(embeddings_path)
    print(f"  Loaded embeddings for {len(item_embeddings)} items")

    # Load cold/warm item lists
    with open(cold_items_path, "r") as f:
        cold_items = set(line.strip() for line in f)

    with open(warm_items_path, "r") as f:
        warm_items = set(line.strip() for line in f)

    print(f"  Cold items: {len(cold_items)}")
    print(f"  Warm items: {len(warm_items)}")

    # Load user histories from training
    print("\nLoading user histories...")
    train_histories = load_user_histories(train_interactions_path)
    print(f"  {len(train_histories)} users in training")

    # Load test interactions
    print("\nLoading test interactions...")
    test_interactions = []
    with open(test_interactions_path, "r") as f:
        for line in f:
            test_interactions.append(json.loads(line.strip()))

    # Group test interactions by user
    user_test_items: Dict[str, Dict[str, list]] = {}
    for interaction in test_interactions:
        user_id = interaction["user_id"]
        item_id = interaction["parent_asin"]

        if user_id not in user_test_items:
            user_test_items[user_id] = {"warm": [], "cold": []}

        if item_id in cold_items:
            user_test_items[user_id]["cold"].append(item_id)
        elif item_id in warm_items:
            user_test_items[user_id]["warm"].append(item_id)

    print(f"  {len(user_test_items)} users in test set")

    # Compute item popularity for negative sampling
    print("\nComputing item popularity...")
    item_popularity = compute_item_popularity(train_interactions_path)

    # Initialize recommender
    print("\nInitializing recommender...")
    recommender = ZeroShotDotProductRecommender(
        item_embeddings=item_embeddings,
        user_aggregation=config["evaluation"].get("user_aggregation", "mean"),
        normalize=True,
    )

    # Evaluation settings
    k_values = config["evaluation"]["k_values"]
    num_negatives = config["evaluation"]["num_negatives"]
    negative_sampling = config["evaluation"]["negative_sampling"]

    # Evaluate on cold items
    print(f"\n{'=' * 60}")
    print("Evaluating on COLD items...")
    print(f"{'=' * 60}")

    cold_metrics_list = []

    for user_id, test_items in tqdm(user_test_items.items(), desc="Users"):
        if len(test_items["cold"]) == 0:
            continue  # No cold items for this user

        # Get user history (warm items only)
        user_history = train_histories.get(user_id, [])
        if len(user_history) == 0:
            continue  # No history

        # Sample negatives from cold items
        all_cold_items = list(cold_items & set(item_embeddings.keys()))
        negatives = sample_negatives(
            all_items=all_cold_items,
            positive_items=test_items["cold"],
            num_negatives=num_negatives,
            strategy=negative_sampling,
            item_popularity=item_popularity,
        )

        # Candidate items: positives + negatives
        candidates = test_items["cold"] + negatives

        if len(candidates) == 0:
            continue

        # Get recommendations
        recommendations = recommender.recommend(
            user_history=user_history,
            candidate_items=candidates,
            exclude_items=None,  # Don't exclude anything
            k=max(k_values),
        )

        # Compute metrics
        metrics = compute_all_metrics(
            predictions=recommendations,
            ground_truth=test_items["cold"],
            k_values=k_values,
        )

        cold_metrics_list.append(metrics)

    # Aggregate cold item metrics
    cold_mean, cold_std = aggregate_metrics(cold_metrics_list)
    print_metrics(cold_mean, cold_std, title=f"{model_name} - Cold Items")

    # Evaluate on warm items (for comparison)
    print(f"\n{'=' * 60}")
    print("Evaluating on WARM items...")
    print(f"{'=' * 60}")

    warm_metrics_list = []

    for user_id, test_items in tqdm(user_test_items.items(), desc="Users"):
        if len(test_items["warm"]) == 0:
            continue

        user_history = train_histories.get(user_id, [])
        if len(user_history) == 0:
            continue

        # Sample negatives from warm items
        all_warm_items = list(warm_items & set(item_embeddings.keys()))
        negatives = sample_negatives(
            all_items=all_warm_items,
            positive_items=test_items["warm"],
            num_negatives=num_negatives,
            strategy=negative_sampling,
            item_popularity=item_popularity,
        )

        candidates = test_items["warm"] + negatives

        if len(candidates) == 0:
            continue

        recommendations = recommender.recommend(
            user_history=user_history,
            candidate_items=candidates,
            exclude_items=user_history,  # Exclude training items
            k=max(k_values),
        )

        metrics = compute_all_metrics(
            predictions=recommendations,
            ground_truth=test_items["warm"],
            k_values=k_values,
        )

        warm_metrics_list.append(metrics)

    # Aggregate warm item metrics
    warm_mean, warm_std = aggregate_metrics(warm_metrics_list)
    print_metrics(warm_mean, warm_std, title=f"{model_name} - Warm Items")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "model_name": model_name,
        "cold_items": {
            "mean": cold_mean,
            "std": cold_std,
            "num_users": len(cold_metrics_list),
        },
        "warm_items": {
            "mean": warm_mean,
            "std": warm_std,
            "num_users": len(warm_metrics_list),
        },
    }

    save_json(results, str(output_path / f"{model_name}_results.json"))

    print(f"\n✓ Results saved to {str(output_path / f'{model_name}_results.json')}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cold-start recommendation")
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to item embeddings (.npz file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model_name", type=str, default="model", help="Model name for results"
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiments/results", help="Output directory"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Paths
    data_dir = Path(config["data"]["output_dir"])

    # Evaluate
    evaluate_cold_start(
        embeddings_path=args.embeddings,
        train_interactions_path=str(data_dir / "train_interactions.jsonl"),
        test_interactions_path=str(data_dir / "test_interactions.jsonl"),
        cold_items_path=str(data_dir / "cold_items.txt"),
        warm_items_path=str(data_dir / "warm_items.txt"),
        output_dir=args.output_dir,
        config=config,
        model_name=args.model_name,
    )
