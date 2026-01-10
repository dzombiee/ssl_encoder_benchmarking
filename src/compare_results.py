"""
Compare results across all models and perform statistical tests.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from utils.helpers import load_json


def load_all_results(results_dir: str):
    """Load results from all models."""
    results_path = Path(results_dir)

    all_results = {}

    for result_file in results_path.glob("*_results.json"):
        model_name = result_file.stem.replace("_results", "")
        results = load_json(str(result_file))
        all_results[model_name] = results

    return all_results


def create_comparison_table(
    all_results: dict, metric: str = "ndcg@10", item_type: str = "cold_items"
):
    """Create comparison table for a specific metric."""
    data = []

    for model_name, results in all_results.items():
        mean_val = results[item_type]["mean"].get(metric, 0.0)
        std_val = results[item_type]["std"].get(metric, 0.0)

        data.append(
            {
                "Model": model_name,
                "Mean": mean_val,
                "Std": std_val,
                "Display": f"{mean_val:.4f} ± {std_val:.4f}",
            }
        )

    df = pd.DataFrame(data)
    df = df.sort_values("Mean", ascending=False)

    return df


def create_comparison_plots(all_results: dict, output_dir: str):
    """Create visualization plots comparing models."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Metrics to compare
    metrics = ["recall@5", "recall@10", "ndcg@5", "ndcg@10"]

    # Prepare data
    cold_data: dict[str, list[float]] = {metric: [] for metric in metrics}
    warm_data: dict[str, list[float]] = {metric: [] for metric in metrics}
    model_names: list[str] = []

    for model_name, results in all_results.items():
        model_names.append(model_name)

        for metric in metrics:
            cold_data[metric].append(results["cold_items"]["mean"].get(metric, 0.0))
            warm_data[metric].append(results["warm_items"]["mean"].get(metric, 0.0))

    # Plot 1: Cold items performance
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Performance on Cold Items", fontsize=16, fontweight="bold")

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, cold_data[metric], color="steelblue", alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path / "cold_items_comparison.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path / 'cold_items_comparison.png'}")

    # Plot 2: Warm items performance
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Performance on Warm Items", fontsize=16, fontweight="bold")

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, warm_data[metric], color="coral", alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        ax.grid(axis="y", alpha=0.3)

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path / "warm_items_comparison.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path / 'warm_items_comparison.png'}")

    # Plot 3: Cold vs Warm comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    metric = "ndcg@10"
    x = np.arange(len(model_names))

    ax.set_xlabel("Models", fontsize=12)
    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.set_title(
        "Cold vs Warm Items Performance (NDCG@10)", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_path / "cold_vs_warm_comparison.png", dpi=300, bbox_inches="tight"
    )
    print(f"✓ Saved: {output_path / 'cold_vs_warm_comparison.png'}")


def compare_results(results_dir: str, output_dir: str):
    """Main comparison function."""
    print(f"\n{'=' * 80}")
    print("Comparing Model Results")
    print(f"{'=' * 80}\n")

    # Load all results
    print("Loading results...")
    all_results = load_all_results(results_dir)
    print(f"  Loaded results for {len(all_results)} models: {list(all_results.keys())}")

    # Create comparison tables
    print("\n" + "=" * 80)
    print("COLD ITEMS - NDCG@10")
    print("=" * 80)
    cold_table = create_comparison_table(
        all_results, metric="ndcg@10", item_type="cold_items"
    )
    print(cold_table.to_string(index=False))

    print("\n" + "=" * 80)
    print("WARM ITEMS - NDCG@10")
    print("=" * 80)
    warm_table = create_comparison_table(
        all_results, metric="ndcg@10", item_type="warm_items"
    )
    print(warm_table.to_string(index=False))

    print("\n" + "=" * 80)
    print("COLD ITEMS - Recall@10")
    print("=" * 80)
    recall_table = create_comparison_table(
        all_results, metric="recall@10", item_type="cold_items"
    )
    print(recall_table.to_string(index=False))

    # Create plots
    print("\nCreating comparison plots...")
    create_comparison_plots(all_results, output_dir)

    # Save tables
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cold_table.to_csv(output_path / "cold_items_ndcg10.csv", index=False)
    warm_table.to_csv(output_path / "warm_items_ndcg10.csv", index=False)
    recall_table.to_csv(output_path / "cold_items_recall10.csv", index=False)

    print(f"\n✓ Comparison tables saved to {output_path}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    best_cold_model = cold_table.iloc[0]["Model"]
    best_cold_ndcg = cold_table.iloc[0]["Mean"]

    best_warm_model = warm_table.iloc[0]["Model"]
    best_warm_ndcg = warm_table.iloc[0]["Mean"]

    print(
        f"\nBest model for COLD items: {best_cold_model} (NDCG@10: {best_cold_ndcg:.4f})"
    )
    print(
        f"Best model for WARM items: {best_warm_model} (NDCG@10: {best_warm_ndcg:.4f})"
    )

    # Calculate performance gap
    random_cold_ndcg = (
        cold_table[cold_table["Model"] == "random"].iloc[0]["Mean"]
        if "random" in cold_table["Model"].values
        else 0
    )
    if random_cold_ndcg > 0:
        improvement = ((best_cold_ndcg - random_cold_ndcg) / random_cold_ndcg) * 100
        print(f"\nImprovement over random: {improvement:.1f}%")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model results")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for comparison results",
    )

    args = parser.parse_args()

    compare_results(args.results_dir, args.output_dir)
