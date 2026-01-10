#!/usr/bin/env python3
"""
Quick test script to verify training works with small data.
Tests the complete pipeline with reduced settings to catch issues early.
"""

import sys
from pathlib import Path
import torch  # type: ignore

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformers import AutoTokenizer  # type: ignore
from data.datasets import (
    ItemMetadataDataset,
    ContrastiveDataset,
    collate_fn_contrastive,
)
from data.augmentations import simcse_dropout_augmentation
from models.simcse import SimCSEModel
from utils.helpers import get_device, set_seed
from torch.utils.data import DataLoader  # type: ignore


def quick_train_test():
    """Test training with minimal data."""
    print("\n" + "=" * 60)
    print("Quick Training Test (SimCSE)")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # Check if processed data exists
    data_path = Path("data/processed/item_metadata.jsonl")
    if not data_path.exists():
        print(f"\n✗ Data not found: {data_path}")
        print("  Please run: python src/data/preprocessing.py")
        return False

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Load metadata (limit to 1000 items for testing)
    print("\nLoading metadata (limited to 1000 items for testing)...")
    metadata_dataset = ItemMetadataDataset(
        metadata_path=str(data_path), tokenizer=tokenizer, max_length=128
    )

    # Limit dataset size for quick test
    metadata_dataset.items = metadata_dataset.items[:1000]
    print(f"  Using {len(metadata_dataset.items)} items")

    # Create contrastive dataset
    print("\nCreating contrastive dataset...")
    dataset = ContrastiveDataset(
        metadata_dataset=metadata_dataset,
        augmentation_fn=simcse_dropout_augmentation,
        num_views=2,
    )

    # Create dataloader
    print("Creating dataloader...")
    batch_size = 16  # Small batch for testing
    collate_fn = lambda batch: collate_fn_contrastive(batch, tokenizer, 128)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"  Batches: {len(dataloader)}")

    # Create model
    print("\nCreating SimCSE model...")
    model = SimCSEModel(
        model_name="distilbert-base-uncased",
        embedding_dim=128,
        pooling_strategy="mean",
        dropout=0.1,
        temperature=0.05,
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {num_params:,}")

    # Optimizer
    print("\nCreating optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Test training loop
    print("\nTesting training loop (2 batches)...")
    model.train()

    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 2:  # Only test 2 batches
                break

            print(f"  Batch {batch_idx + 1}:")
            print(f"    Input IDs shape: {batch['input_ids'].shape}")
            print(f"    Num views: {batch['num_views']}")

            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            num_views = batch["num_views"]

            # Forward pass
            loss, embeddings = model(input_ids, attention_mask, num_views)

            print(f"    Loss: {loss.item():.4f}")
            print(f"    Embeddings shape: {embeddings.shape}")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("    ✓ Backward pass successful")

            # Clear cache
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print("\n✓ Training test successful!")

        # Test inference
        print("\nTesting inference...")
        model.eval()
        with torch.no_grad():
            test_batch = next(iter(dataloader))
            input_ids = test_batch["input_ids"][:4].to(device)
            attention_mask = test_batch["attention_mask"][:4].to(device)

            embeddings = model.encode(input_ids, attention_mask)
            print(f"  Embeddings shape: {embeddings.shape}")
            print("  ✓ Inference successful")

        return True

    except Exception as e:
        print(f"\n✗ Training test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("   Quick Training Test - Verify Setup Before Full Training")
    print("=" * 70)

    success = quick_train_test()

    if success:
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        print("\nYou can now run full training:")
        print("  python src/train_ssl.py --model_type simcse \\")
        print("    --config configs/simcse_config.yaml \\")
        print("    --output_dir experiments/simcse")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("✗ Tests failed. Please fix the errors above.")
        print("=" * 70)

    sys.exit(0 if success else 1)
