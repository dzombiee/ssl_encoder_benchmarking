"""
Train baseline models (Random, TF-IDF).
"""

import argparse
import json
from pathlib import Path

from models.baselines import (
    RandomEmbeddingBaseline,
    TFIDFBaseline,
    SentenceBERTBaseline,
)
from utils.helpers import load_config, save_embeddings


def train_random_baseline(
    metadata_path: str, output_dir: str, embedding_dim: int = 256, seed: int = 42
):
    """Train random embedding baseline."""
    print(f"\n{'=' * 60}")
    print("Training Random Embedding Baseline")
    print(f"{'=' * 60}\n")

    # Load item IDs
    print("Loading items...")
    items = []
    with open(metadata_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            items.append(item["parent_asin"])

    print(f"  {len(items)} items")

    # Create random embeddings
    print("\nCreating random embeddings...")
    baseline = RandomEmbeddingBaseline(embedding_dim=embedding_dim, seed=seed)
    baseline.fit(items)

    # Save embeddings
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_path / "item_embeddings.npz"
    save_embeddings(baseline.get_embeddings(), str(embeddings_path))

    print(f"\n✓ Random embeddings saved: {embeddings_path}")


def train_tfidf_baseline(
    warm_metadata_path: str,
    all_metadata_path: str,
    output_dir: str,
    max_features: int = 5000,
):
    """Train TF-IDF baseline with proper cold-start protocol.

    Args:
        warm_metadata_path: Path to WARM item metadata (for fitting vocabulary)
        all_metadata_path: Path to ALL item metadata (warm + cold, for transformation)
        output_dir: Output directory for embeddings
        max_features: Maximum TF-IDF features
    """
    print(f"\n{'=' * 60}")
    print("Training TF-IDF Baseline (Cold-Start Protocol)")
    print(f"{'=' * 60}\n")

    # Load warm items (for vocabulary)
    print("Loading WARM items (for vocabulary learning)...")
    warm_items = []
    with open(warm_metadata_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            warm_items.append(item)
    print(f"  {len(warm_items)} warm items")

    # Load all items (for transformation)
    print("\nLoading ALL items (for transformation)...")
    all_items = []
    with open(all_metadata_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            all_items.append(item)
    print(f"  {len(all_items)} total items (warm + cold)")

    # Fit TF-IDF on warm, transform all
    print("\nFitting TF-IDF on WARM items only...")
    baseline = TFIDFBaseline(max_features=max_features, ngram_range=(1, 2))
    baseline.fit(train_items=warm_items, all_items=all_items)

    # Save embeddings
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_path / "item_embeddings.npz"
    save_embeddings(baseline.get_embeddings(), str(embeddings_path))

    print(f"\n✓ TF-IDF embeddings saved: {embeddings_path}")


def train_sbert_baseline(
    metadata_path: str, output_dir: str, model_name: str = "all-MiniLM-L6-v2"
):
    """Train Sentence-BERT baseline."""
    print(f"\n{'=' * 60}")
    print("Training Sentence-BERT Baseline")
    print(f"{'=' * 60}\n")

    # Load items
    print("Loading items...")
    items = []
    with open(metadata_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            items.append(item)

    print(f"  {len(items)} items")

    # Encode with Sentence-BERT
    print(f"\nEncoding with Sentence-BERT ({model_name})...")
    baseline = SentenceBERTBaseline(model_name=model_name)
    baseline.fit(items)

    # Save embeddings
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_path / "item_embeddings.npz"
    save_embeddings(baseline.get_embeddings(), str(embeddings_path))

    print(f"\n✓ Sentence-BERT embeddings saved: {embeddings_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument(
        "--baseline_type",
        type=str,
        required=True,
        choices=["random", "tfidf", "sbert"],
        help="Type of baseline",
    )
    parser.add_argument(
        "--metadata_path", type=str, required=True, help="Path to item metadata"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    if args.baseline_type == "random":
        train_random_baseline(
            metadata_path=args.metadata_path,
            output_dir=args.output_dir,
            embedding_dim=config["model"]["embedding_dim"],
            seed=config["split"]["random_seed"],
        )
    elif args.baseline_type == "tfidf":
        # For TF-IDF, we need both warm and all metadata to prevent leakage
        warm_metadata = args.metadata_path.replace(
            "item_metadata.jsonl", "warm_item_metadata.jsonl"
        )
        train_tfidf_baseline(
            warm_metadata_path=warm_metadata,
            all_metadata_path=args.metadata_path,
            output_dir=args.output_dir,
            max_features=5000,
        )
    elif args.baseline_type == "sbert":
        train_sbert_baseline(
            metadata_path=args.metadata_path,
            output_dir=args.output_dir,
            model_name="all-MiniLM-L6-v2",
        )
