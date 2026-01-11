"""
Dataset classes for SSL pretraining and evaluation.
"""

import json
import random
from typing import Dict, List, Optional
import numpy as np
import torch  # type: ignore
from torch.utils.data import Dataset  # type: ignore
from transformers import PreTrainedTokenizer  # type: ignore


class ItemMetadataDataset(Dataset):
    """
    Dataset for item metadata (for SSL pretraining).
    Each item has: title, description, attributes, and full concatenated text.
    """

    def __init__(
        self,
        metadata_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        text_field: str = "full_text",
        sample_fraction: float = 1.0,  # NEW: fraction of data to use (0.0-1.0)
    ):
        """
        Args:
            metadata_path: Path to item metadata JSONL file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            text_field: Which text field to use ('full_text', 'title', 'description')
            sample_fraction: Fraction of data to sample (default 1.0 = use all)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field

        # Load metadata
        self.items = []
        with open(metadata_path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get(text_field, "").strip():  # Only keep items with text
                    self.items.append(item)

        # Sample if requested
        if sample_fraction < 1.0:
            original_size = len(self.items)
            random.seed(42)  # Reproducible sampling
            sample_size = max(1, int(len(self.items) * sample_fraction))
            self.items = random.sample(self.items, sample_size)
            print(
                f"Sampled {len(self.items)}/{original_size} items ({sample_fraction * 100:.1f}%)"
            )
        else:
            print(f"Loaded {len(self.items)} items from {metadata_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        text = item.get(self.text_field, "")
        if not text:
            text = item.get("title", "") + " " + item.get("description", "")

        return {
            "item_id": item["parent_asin"],
            "text": text,
            "title": item.get("title", ""),
            "description": item.get("description", ""),
            "attributes": item.get("attributes", ""),
            "category": item.get("category", "unknown"),
        }

    def get_item_by_id(self, item_id: str) -> Optional[Dict]:
        """Get item by its ID."""
        for item in self.items:
            if item["parent_asin"] == item_id:
                return item
        return None


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning (SimCSE, SimCLR).
    Returns pairs of augmented views for each item.
    """

    def __init__(
        self,
        metadata_dataset: ItemMetadataDataset,
        augmentation_fn=None,
        num_views: int = 2,
    ):
        """
        Args:
            metadata_dataset: Base ItemMetadataDataset
            augmentation_fn: Function to augment text
            num_views: Number of augmented views per item
        """
        self.metadata_dataset = metadata_dataset
        self.augmentation_fn = augmentation_fn
        self.num_views = num_views

    def __len__(self):
        return len(self.metadata_dataset)

    def __getitem__(self, idx):
        item = self.metadata_dataset[idx]
        text = item["text"]

        # Create multiple views
        views = []
        for _ in range(self.num_views):
            if self.augmentation_fn:
                augmented_text = self.augmentation_fn(text)
            else:
                augmented_text = text
            views.append(augmented_text)

        return {"item_id": item["item_id"], "views": views, "original_text": text}


class MultiViewDataset(Dataset):
    """
    Dataset for multi-view contrastive learning.
    Treats different metadata fields (title, description, attributes) as separate views.
    """

    def __init__(
        self,
        metadata_dataset: ItemMetadataDataset,
        views: List[str] = ["title", "description", "attributes"],
        fallback_strategy: str = "concatenate",
    ):
        """
        Args:
            metadata_dataset: Base ItemMetadataDataset
            views: List of metadata fields to use as views
            fallback_strategy: What to do when a view is missing ('concatenate', 'skip')
        """
        self.metadata_dataset = metadata_dataset
        self.view_fields = views
        self.fallback_strategy = fallback_strategy

        # Filter items that have at least one view
        self.valid_indices = []
        for idx in range(len(metadata_dataset)):
            item = metadata_dataset[idx]
            has_view = any(item.get(view, "").strip() for view in views)
            if has_view:
                self.valid_indices.append(idx)

        print(
            f"MultiViewDataset: {len(self.valid_indices)}/{len(metadata_dataset)} items with valid views"
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        item = self.metadata_dataset[actual_idx]

        # Get each view
        views = {}
        available_views = []

        for view_name in self.view_fields:
            view_text = item.get(view_name, "").strip()
            if view_text:
                views[view_name] = view_text
                available_views.append(view_name)

        # Fallback if some views are missing
        if (
            len(available_views) < len(self.view_fields)
            and self.fallback_strategy == "concatenate"
        ):
            # Concatenate available views
            fallback_text = " ".join(views.values())
            for view_name in self.view_fields:
                if view_name not in views:
                    views[view_name] = fallback_text

        return {
            "item_id": item["item_id"],
            "views": views,
            "available_views": available_views,
        }


class RecommendationDataset(Dataset):
    """
    Dataset for evaluation (user-item interactions).
    """

    def __init__(
        self,
        interactions_path: str,
        item_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Args:
            interactions_path: Path to interactions JSONL file
            item_embeddings: Optional precomputed item embeddings
        """
        self.item_embeddings = item_embeddings

        # Load interactions
        self.interactions = []
        with open(interactions_path, "r") as f:
            for line in f:
                interaction = json.loads(line.strip())
                self.interactions.append(interaction)

        print(f"Loaded {len(self.interactions)} interactions from {interactions_path}")

        # Build user histories
        self.user_histories: Dict[str, List[str]] = {}
        for interaction in self.interactions:
            user_id = interaction["user_id"]
            item_id = interaction["parent_asin"]

            if user_id not in self.user_histories:
                self.user_histories[user_id] = []
            self.user_histories[user_id].append(item_id)

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        interaction = self.interactions[idx]
        return {
            "user_id": interaction["user_id"],
            "item_id": interaction["parent_asin"],
            "rating": interaction.get("rating", 1.0),
            "timestamp": interaction.get("timestamp", 0),
        }

    def get_user_history(
        self, user_id: str, exclude_items: Optional[List[str]] = None
    ) -> List[str]:
        """Get user's interaction history."""
        history = self.user_histories.get(user_id, [])
        if exclude_items:
            history = [item for item in history if item not in exclude_items]
        return history


def collate_fn_contrastive(batch, tokenizer, max_length=256):
    """
    Collate function for contrastive learning.
    Returns tokenized views for each item.
    """
    item_ids = [item["item_id"] for item in batch]

    # Get all views
    all_views = []
    for item in batch:
        all_views.extend(item["views"])

    # Tokenize all views at once
    encoded = tokenizer(
        all_views,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "item_ids": item_ids,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "num_views": len(batch[0]["views"]),
    }


def collate_fn_multiview(batch, tokenizer, max_length=256):
    """
    Collate function for multi-view contrastive learning.
    """
    item_ids = [item["item_id"] for item in batch]
    view_names = list(batch[0]["views"].keys())

    # Organize texts by view
    views_data = {view_name: [] for view_name in view_names}

    for item in batch:
        for view_name in view_names:
            views_data[view_name].append(item["views"].get(view_name, ""))

    # Tokenize each view separately
    encoded_views = {}
    for view_name, texts in views_data.items():
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded_views[view_name] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    return {"item_ids": item_ids, "views": encoded_views, "view_names": view_names}


def collate_fn_tsdae(batch, tokenizer, max_length=256, deletion_prob=0.6):
    """
    Collate function for TSDAE (denoising autoencoder).
    Creates corrupted input and original target.
    """
    item_ids = [item["item_id"] for item in batch]
    texts = [item["text"] for item in batch]

    # Tokenize original texts (targets)
    targets = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )

    # Create corrupted inputs by deleting tokens
    corrupted_texts = []
    for text in texts:
        words = text.split()
        if len(words) > 5:
            # Randomly delete words
            kept_words = [w for w in words if random.random() > deletion_prob]
            if len(kept_words) < 3:  # Keep at least some words
                kept_words = random.sample(words, min(3, len(words)))
            corrupted_text = " ".join(kept_words)
        else:
            corrupted_text = text
        corrupted_texts.append(corrupted_text)

    # Tokenize corrupted inputs
    inputs = tokenizer(
        corrupted_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "item_ids": item_ids,
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "target_ids": targets["input_ids"],
        "target_attention_mask": targets["attention_mask"],
    }


def collate_fn_mlm(batch, tokenizer, max_length=256, mask_prob=0.15):
    """
    Collate function for Masked Language Modeling.
    """
    item_ids = [item["item_id"] for item in batch]
    texts = [item["text"] for item in batch]

    # Tokenize
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )

    # Create masked inputs
    input_ids = encoded["input_ids"].clone()
    labels = encoded["input_ids"].clone()

    # Mask tokens
    probability_matrix = torch.full(labels.shape, mask_prob)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
    )

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # 80% of time, replace with [MASK]
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% of time, replace with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # 10% of time, keep original

    return {
        "item_ids": item_ids,
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }
