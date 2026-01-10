import pandas as pd
import json
from pathlib import Path
import argparse


def load_jsonl(filepath):
    """Load JSONL file into list of dicts"""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def clean_text(text):
    """Clean text fields (descriptions, features, etc.)"""
    if isinstance(text, list):
        text = " ".join(text)
    if not isinstance(text, str):
        return ""
    return text.strip().lower()


def prepare_coldstart_splits(
    reviews_path,
    meta_path,
    output_dir,
    train_ratio=0.85,
    min_interactions=5,
):
    """
    Create temporal split with cold-start items for test set

    Simplified Strategy for Thesis:
    1. Temporal split: 85% train, 15% test by timestamp
    2. Identify items that appear ONLY in test (cold-start items)
    3. Keep users who have enough history in training
    4. Use RecBole's internal validation split during fine-tuning

    This avoids RecBole compatibility issues and follows standard practice:
    - Train: used for pretraining + fine-tuning (with internal val split)
    - Test: evaluated ONCE for final thesis results
    """

    print("Loading data...")
    reviews = load_jsonl(reviews_path)
    metadata = load_jsonl(meta_path)

    # Convert to DataFrames
    reviews_df = pd.DataFrame(reviews)
    meta_df = pd.DataFrame(metadata)

    print(f"Reviews: {len(reviews_df)}, Items: {len(meta_df)}")

    # Merge reviews with metadata
    print("Merging with metadata...")
    print(f"Meta columns: {meta_df.columns.tolist()}")

    # Process metadata fields
    meta_df["title"] = (
        meta_df["title"].fillna("").apply(clean_text)
        + " "
        + meta_df["description"].fillna("").apply(clean_text)
    )
    meta_df["store"] = meta_df["store"].fillna("unknown").apply(clean_text)
    meta_df["avg_rating"] = meta_df["average_rating"].fillna(0.0).astype(float)

    # Extract main category (class) - use main_category if available, else categories
    def get_main_category(row):
        if pd.notna(row.get("main_category", None)) and row["main_category"]:
            return clean_text(row["main_category"])
        if isinstance(row.get("categories", None), list) and len(row["categories"]) > 0:
            return (
                row["categories"][0]
                if isinstance(row["categories"][0], str)
                else "default"
            )
        return "default"

    meta_df["class"] = meta_df.apply(get_main_category, axis=1)

    merged = reviews_df.merge(
        meta_df[["parent_asin", "title", "store", "avg_rating", "class"]],
        on="parent_asin",
        how="inner",
        suffixes=("_review", "_item"),
    )

    print(f"Merged columns: {merged.columns.tolist()}")

    # Use the item title (not review title)
    if "title_item" in merged.columns:
        merged["title"] = merged["title_item"]
    elif "title_y" in merged.columns:
        merged["title"] = merged["title_y"]

    # Remove items without title
    merged = merged[merged["title"].str.len() > 0]

    print(f"After merge: {len(merged)} interactions")

    # Filter users and items with minimum interactions
    print(f"Filtering users/items with <{min_interactions} interactions...")
    user_counts = merged["user_id"].value_counts()
    item_counts = merged["parent_asin"].value_counts()

    valid_users = user_counts[user_counts >= min_interactions].index
    valid_items = item_counts[item_counts >= min_interactions].index

    merged = merged[merged["user_id"].isin(valid_users)]
    merged = merged[merged["parent_asin"].isin(valid_items)]

    print(
        f"After filtering: {len(merged)} interactions, "
        f"{merged['user_id'].nunique()} users, "
        f"{merged['parent_asin'].nunique()} items"
    )

    # STEP 1: Temporal split
    print("\nCreating temporal splits...")
    merged = merged.sort_values("timestamp")

    n = len(merged)
    train_cutoff = int(n * train_ratio)

    train_df = merged.iloc[:train_cutoff].copy()
    test_df = merged.iloc[train_cutoff:].copy()

    print(
        f"Temporal split - Train: {len(train_df)} ({train_ratio * 100:.0f}%), "
        f"Test: {len(test_df)} ({(1 - train_ratio) * 100:.0f}%)"
    )

    # STEP 2: Identify cold-start items
    train_items = set(train_df["parent_asin"].unique())
    test_items = set(test_df["parent_asin"].unique())

    # Cold items = appear in test but NOT in train
    cold_test_items = test_items - train_items

    print(
        f"\nCold-start items in Test: {len(cold_test_items)} "
        f"({len(cold_test_items) / len(test_items) * 100:.1f}% of test items)"
    )

    # STEP 3: Filter to cold-start interactions
    # Keep only interactions with cold items
    test_cold = test_df[test_df["parent_asin"].isin(cold_test_items)].copy()

    # Keep only users who appeared in training (need user profile for predictions)
    train_users = set(train_df["user_id"].unique())
    test_cold = test_cold[test_cold["user_id"].isin(train_users)]

    print(
        f"Cold-start Test interactions: {len(test_cold)} "
        f"({len(test_cold) / len(test_df) * 100:.1f}% of test data)"
    )
    print(f"Cold-start Test users: {test_cold['user_id'].nunique()}")

    # STEP 4: Save splits
    print("\nSaving splits...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # First, create the combined item file with ALL items (warm + cold)
    # This shared vocabulary enables cold-start evaluation
    all_items = pd.concat(
        [
            train_df[["parent_asin", "title", "store", "avg_rating", "class"]],
            test_cold[["parent_asin", "title", "store", "avg_rating", "class"]],
        ]
    ).drop_duplicates("parent_asin")
    all_items.columns = [
        "item_id:token",
        "title:token_seq",
        "store:token_seq",
        "avg_rating:float",
        "class:token",
    ]

    print(f"\n  Creating shared item vocabulary: {len(all_items)} total items")
    print(f"    - Warm items (in train): {train_df['parent_asin'].nunique()}")
    print(f"    - Cold items (test only): {len(cold_test_items)}")

    # Save in RecBole format (.inter files + shared .item file)
    def save_recbole_format(df, name, output_path, all_items_df):
        # Create subdirectory for this dataset
        dataset_dir = output_path / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # .inter file (user-item interactions)
        inter_df = df[["user_id", "parent_asin", "rating", "timestamp"]].copy()
        inter_df.columns = [
            "user_id:token",
            "item_id:token",
            "rating:float",
            "timestamp:float",
        ]
        inter_path = dataset_dir / f"{name}.inter"
        inter_df.to_csv(inter_path, sep=",", index=False)

        # .item file - USE SHARED ALL ITEMS to ensure same vocabulary
        item_path = dataset_dir / f"{name}.item"
        all_items_df.to_csv(item_path, sep=",", index=False)

        print(
            f"  Saved {name}: {len(inter_df)} interactions, {len(all_items_df)} items (shared vocab)"
        )

    save_recbole_format(train_df, "All_Beauty_Train", output_path, all_items)
    save_recbole_format(test_cold, "All_Beauty_Test_ColdStart", output_path, all_items)

    # Save statistics
    stats = {
        "total_interactions": len(merged),
        "total_users": merged["user_id"].nunique(),
        "total_items": merged["parent_asin"].nunique(),
        "train_ratio": train_ratio,
        "train_interactions": len(train_df),
        "train_users": train_df["user_id"].nunique(),
        "train_items": train_df["parent_asin"].nunique(),
        "test_cold_interactions": len(test_cold),
        "test_cold_users": test_cold["user_id"].nunique(),
        "test_cold_items": len(cold_test_items),
        "test_cold_coverage": len(test_cold) / len(test_df) if len(test_df) > 0 else 0,
    }

    stats_path = output_path / "split_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 70)
    print("COLD-START SPLIT COMPLETE (Simplified for Thesis)")
    print("=" * 70)
    print(
        f"Train: {stats['train_interactions']} interactions, "
        f"{stats['train_users']} users, {stats['train_items']} items"
    )
    print("       → Use for: Pretraining + Fine-tuning")
    print("       → RecBole will create internal validation split for early stopping")
    print()
    print(
        f"Test (Cold-Start): {stats['test_cold_interactions']} interactions, "
        f"{stats['test_cold_users']} users, {stats['test_cold_items']} NEW items"
    )
    print("       → Use for: FINAL evaluation ONLY (report in thesis)")
    print("       → Evaluate ONCE after all hyperparameter tuning is complete")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare cold-start recommendation dataset"
    )
    parser.add_argument(
        "--reviews",
        type=str,
        default="./src/data/All_Beauty.jsonl",
        help="Path to reviews JSONL file",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="./src/data/meta_All_Beauty.jsonl",
        help="Path to metadata JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./src/data/coldstart_splits",
        help="Output directory for splits",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.85,
        help="Ratio of data for training (rest is test)",
    )
    parser.add_argument(
        "--min_interactions",
        type=int,
        default=5,
        help="Minimum interactions per user/item",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    prepare_coldstart_splits(
        reviews_path=args.reviews,
        meta_path=args.meta,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        min_interactions=args.min_interactions,
    )
