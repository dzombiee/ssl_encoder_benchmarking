"""
Data preprocessing and cold-start split generation for SSL recommendation.

This module implements the item-based cold-start split protocol:
1. Select 15-20% of items as cold items
2. Remove ALL interactions for cold items from training
3. Keep metadata for all items (warm and cold)
4. Ensure test users have at least 1 warm item in their history
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np
import pandas as pd  # type: ignore


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file into list of dicts."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict], filepath: str):
    """Save list of dicts to JSONL file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def clean_text(text) -> str:
    """Clean and normalize text fields."""
    if isinstance(text, list):
        text = " ".join(str(t) for t in text if t)
    if not isinstance(text, str):
        return ""

    # Basic cleaning
    text = str(text).strip()
    # Remove excessive whitespace
    text = " ".join(text.split())
    return text


def process_metadata(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean metadata fields."""
    print("Processing metadata...")

    # Clean title
    meta_df["title"] = meta_df["title"].fillna("").apply(clean_text)

    # Clean description
    if "description" in meta_df.columns:
        meta_df["description"] = meta_df["description"].apply(
            lambda x: clean_text(x) if x else ""
        )
    else:
        meta_df["description"] = ""

    # Extract features/attributes
    if "features" in meta_df.columns:
        meta_df["attributes"] = meta_df["features"].apply(
            lambda x: clean_text(x) if x else ""
        )
    else:
        meta_df["attributes"] = ""

    # Extract main category
    def get_main_category(row):
        if pd.notna(row.get("main_category")) and row["main_category"]:
            return clean_text(row["main_category"])
        if isinstance(row.get("categories"), list) and len(row["categories"]) > 0:
            cat = row["categories"][0]
            return clean_text(cat) if isinstance(cat, str) else "unknown"
        return "unknown"

    meta_df["category"] = meta_df.apply(get_main_category, axis=1)

    # Extract store/brand information
    if "store" in meta_df.columns:
        meta_df["store"] = meta_df["store"].fillna("").apply(clean_text)
    else:
        meta_df["store"] = ""

    # Extract product details (dimensions, weight, etc.)
    def extract_details(details):
        if not details or not isinstance(details, dict):
            return ""
        return " ".join(f"{k} {v}" for k, v in details.items() if v)

    if "details" in meta_df.columns:
        meta_df["product_details"] = meta_df["details"].apply(extract_details)
    else:
        meta_df["product_details"] = ""

    # Concatenate text for full item representation (ENRICHED VERSION)
    # Use periods as separators for better sentence boundaries
    meta_df["full_text"] = (
        meta_df["title"].astype(str)
        + " . "
        + meta_df["description"].astype(str)
        + " . "
        + meta_df["attributes"].astype(str)
        + " . Product category: "
        + meta_df["category"].astype(str)
        + " . Brand: "
        + meta_df["store"].astype(str)
        + " . "
        + meta_df["product_details"].astype(str)
    ).apply(lambda x: " ".join(x.split()))  # Remove extra spaces

    # Filter out items with insufficient text
    min_length = 3
    meta_df = meta_df[meta_df["full_text"].str.len() >= min_length]

    print(f"Processed {len(meta_df)} items with valid metadata")

    return meta_df


def create_cold_start_split(
    reviews_path: str,
    metadata_path: str,
    output_dir: str,
    cold_item_ratio: float = 0.15,
    min_user_interactions: int = 5,
    min_warm_items_per_user: int = 1,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Create item-based cold-start splits with NO LEAKAGE.

    CRITICAL: SSL pretraining uses ONLY metadata (title, description, attributes).
    Cold items have ALL their interactions removed from training to prevent leakage.

    Protocol:
    1. Load interactions and metadata
    2. Filter users with sufficient interactions
    3. Select cold items (items that will have NO training interactions)
    4. Split warm items into train/val/test
    5. For each user, ensure they have ≥min_warm_items_per_user warm items in history
       (guarantees valid user vectors for zero-shot evaluation)
    6. Save splits and metadata (including cold_items.txt for reproducibility)

    Returns:
        Dictionary with split statistics
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    print("=" * 80)
    print("Creating Cold-Start Splits")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    reviews = load_jsonl(reviews_path)
    metadata = load_jsonl(metadata_path)

    reviews_df = pd.DataFrame(reviews)
    meta_df = pd.DataFrame(metadata)

    print(f"   Loaded {len(reviews_df)} interactions")
    print(f"   Loaded {len(meta_df)} items")

    # Process metadata
    meta_df = process_metadata(meta_df)

    # Get items that have metadata
    items_with_metadata = set(meta_df["parent_asin"].unique())

    # Filter reviews to only include items with metadata
    reviews_df = reviews_df[reviews_df["parent_asin"].isin(items_with_metadata)]
    print(f"   {len(reviews_df)} interactions with valid metadata")

    # Filter users with minimum interactions
    print(f"\n2. Filtering users (min {min_user_interactions} interactions)...")
    user_counts = reviews_df["user_id"].value_counts()
    valid_users = set(user_counts[user_counts >= min_user_interactions].index)
    reviews_df = reviews_df[reviews_df["user_id"].isin(valid_users)]

    print(f"   {len(valid_users)} valid users")
    print(f"   {len(reviews_df)} interactions after filtering")

    # Get all items that appear in interactions
    all_items = set(reviews_df["parent_asin"].unique())
    print(f"\n3. Total items in interactions: {len(all_items)}")

    # Select cold items randomly
    num_cold_items = int(len(all_items) * cold_item_ratio)
    cold_items = set(random.sample(list(all_items), num_cold_items))
    warm_items = all_items - cold_items

    print(f"   Cold items: {len(cold_items)} ({cold_item_ratio * 100:.1f}%)")
    print(f"   Warm items: {len(warm_items)}")

    # Split interactions into those with warm vs cold items
    cold_interactions = reviews_df[reviews_df["parent_asin"].isin(cold_items)]
    warm_interactions = reviews_df[reviews_df["parent_asin"].isin(warm_items)]

    print("\n4. Splitting interactions...")
    print(f"   Warm item interactions: {len(warm_interactions)}")
    print(f"   Cold item interactions: {len(cold_interactions)}")

    # For warm items, split into train/val/test by timestamp
    warm_interactions = warm_interactions.sort_values("timestamp")

    # Split warm interactions
    num_warm = len(warm_interactions)
    num_val = int(num_warm * val_ratio)
    num_test = int(num_warm * test_ratio)
    num_train = num_warm - num_val - num_test

    train_df = warm_interactions.iloc[:num_train].copy()
    val_df = warm_interactions.iloc[num_train : num_train + num_val].copy()
    test_warm_df = warm_interactions.iloc[num_train + num_val :].copy()

    # All cold item interactions go to test
    test_cold_df = cold_interactions.copy()

    # Combine warm and cold test interactions
    test_df = pd.concat([test_warm_df, test_cold_df], ignore_index=True)

    print("\n5. Split sizes:")
    print(f"   Train: {len(train_df)} interactions (warm items only)")
    print(f"   Val: {len(val_df)} interactions (warm items only)")
    print(
        f"   Test: {len(test_df)} interactions ({len(test_warm_df)} warm + {len(test_cold_df)} cold)"
    )

    # Ensure test users have at least one warm item in training
    print("\n6. Filtering test users with warm item history...")
    train_user_items = defaultdict(set)
    for _, row in train_df.iterrows():
        train_user_items[row["user_id"]].add(row["parent_asin"])

    # Filter test users
    valid_test_users = set()
    for user_id, items in train_user_items.items():
        if len(items) >= min_warm_items_per_user:
            valid_test_users.add(user_id)

    test_df = test_df[test_df["user_id"].isin(valid_test_users)]
    val_df = val_df[val_df["user_id"].isin(valid_test_users)]

    print(f"   Valid test users: {len(valid_test_users)}")
    print(f"   Final test interactions: {len(test_df)}")
    print(f"   Final val interactions: {len(val_df)}")

    # Save splits
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n7. Saving splits to {output_dir}...")

    # Save interaction splits
    save_jsonl(
        train_df.to_dict("records"), str(output_path / "train_interactions.jsonl")
    )
    save_jsonl(val_df.to_dict("records"), str(output_path / "val_interactions.jsonl"))
    save_jsonl(test_df.to_dict("records"), str(output_path / "test_interactions.jsonl"))

    # Save item lists
    with open(output_path / "cold_items.txt", "w") as f:
        for item in sorted(cold_items):
            f.write(f"{item}\n")

    with open(output_path / "warm_items.txt", "w") as f:
        for item in sorted(warm_items):
            f.write(f"{item}\n")

    # Save metadata for all items
    meta_df.to_json(output_path / "item_metadata.jsonl", orient="records", lines=True)

    # Separate metadata for cold and warm items
    cold_meta_df = meta_df[meta_df["parent_asin"].isin(cold_items)]
    warm_meta_df = meta_df[meta_df["parent_asin"].isin(warm_items)]

    cold_meta_df.to_json(
        output_path / "cold_item_metadata.jsonl", orient="records", lines=True
    )
    warm_meta_df.to_json(
        output_path / "warm_item_metadata.jsonl", orient="records", lines=True
    )

    # Compute and save statistics
    stats = {
        "num_users": len(valid_test_users),
        "num_items_total": len(all_items),
        "num_warm_items": len(warm_items),
        "num_cold_items": len(cold_items),
        "num_train_interactions": len(train_df),
        "num_val_interactions": len(val_df),
        "num_test_interactions": len(test_df),
        "num_test_warm_interactions": len(test_warm_df),
        "num_test_cold_interactions": len(test_cold_df),
        "cold_item_ratio": cold_item_ratio,
        "sparsity": 1 - (len(reviews_df) / (len(valid_test_users) * len(all_items))),
    }

    # User-level statistics
    train_user_counts = train_df["user_id"].value_counts()
    test_user_counts = test_df["user_id"].value_counts()

    stats["avg_user_train_interactions"] = train_user_counts.mean()
    stats["avg_user_test_interactions"] = test_user_counts.mean()

    # Item-level statistics
    train_item_counts = train_df["parent_asin"].value_counts()

    stats["avg_item_train_interactions"] = train_item_counts.mean()

    # Save statistics
    with open(output_path / "split_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 80)
    print("Split Statistics:")
    print("=" * 80)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n✓ Cold-start splits created successfully!")

    return stats


if __name__ == "__main__":
    import yaml  # type: ignore

    # Load configuration
    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create splits
    stats = create_cold_start_split(
        reviews_path=config["data"]["reviews_path"],
        metadata_path=config["data"]["metadata_path"],
        output_dir=config["data"]["output_dir"],
        cold_item_ratio=config["split"]["cold_item_ratio"],
        min_user_interactions=config["split"]["min_user_interactions"],
        min_warm_items_per_user=config["split"]["min_warm_items_per_user"],
        val_ratio=config["split"]["val_ratio"],
        test_ratio=config["split"]["test_ratio"],
        random_seed=config["split"]["random_seed"],
    )
