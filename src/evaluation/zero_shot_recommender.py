"""
Zero-shot recommender using dot-product scoring.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import random


class ZeroShotDotProductRecommender:
    """
    Zero-shot recommender using dot-product between user and item embeddings.

    User embedding = mean (or attention-weighted mean) of item embeddings in user history.
    Score(u, i) = dot(user_embedding, item_embedding)
    """

    def __init__(
        self,
        item_embeddings: Dict[str, np.ndarray],
        user_aggregation: str = "mean",
        normalize: bool = True,
    ):
        """
        Args:
            item_embeddings: Dictionary mapping item_id -> embedding vector
            user_aggregation: How to aggregate user history ('mean' or 'attention')
            normalize: Whether to L2 normalize embeddings
        """
        self.item_embeddings = item_embeddings
        self.user_aggregation = user_aggregation
        self.normalize = normalize

        # Normalize item embeddings if requested
        if self.normalize:
            for item_id, emb in self.item_embeddings.items():
                norm = np.linalg.norm(emb)
                if norm > 0:
                    self.item_embeddings[item_id] = emb / norm

        self.all_item_ids = list(item_embeddings.keys())
        print(f"ZeroShotRecommender initialized with {len(self.all_item_ids)} items")

    def get_user_embedding(self, user_history: List[str]) -> Optional[np.ndarray]:
        """
        Compute user embedding from interaction history.

        Args:
            user_history: List of item IDs the user has interacted with

        Returns:
            user_embedding: Aggregated user embedding, or None if no valid items
        """
        # Get embeddings for items in history
        history_embeddings = []
        for item_id in user_history:
            if item_id in self.item_embeddings:
                history_embeddings.append(self.item_embeddings[item_id])

        if len(history_embeddings) == 0:
            return None

        history_embeddings = np.array(history_embeddings)

        # Aggregate
        if self.user_aggregation == "mean":
            user_embedding = np.mean(history_embeddings, axis=0)
        elif self.user_aggregation == "attention":
            # Simplified attention: use last item as query
            query = history_embeddings[-1]
            scores = np.dot(history_embeddings, query)
            weights = np.exp(scores) / np.sum(np.exp(scores))
            user_embedding = np.sum(history_embeddings * weights[:, np.newaxis], axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.user_aggregation}")

        # Normalize
        if self.normalize:
            norm = np.linalg.norm(user_embedding)
            if norm > 0:
                user_embedding = user_embedding / norm

        return user_embedding

    def score_items(
        self, user_embedding: np.ndarray, candidate_items: List[str]
    ) -> Dict[str, float]:
        """
        Score candidate items for a user.

        Args:
            user_embedding: User embedding vector
            candidate_items: List of candidate item IDs

        Returns:
            scores: Dictionary mapping item_id -> score
        """
        scores = {}

        for item_id in candidate_items:
            if item_id in self.item_embeddings:
                item_embedding = self.item_embeddings[item_id]
                score = np.dot(user_embedding, item_embedding)
                scores[item_id] = score
            else:
                scores[item_id] = 0.0

        return scores

    def recommend(
        self,
        user_history: List[str],
        candidate_items: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        k: int = 10,
    ) -> List[str]:
        """
        Recommend top-K items for a user.

        Args:
            user_history: List of item IDs the user has interacted with
            candidate_items: List of candidate items to rank (if None, use all items)
            exclude_items: Items to exclude from recommendations
            k: Number of items to recommend

        Returns:
            recommendations: List of top-K item IDs
        """
        # Get user embedding
        user_embedding = self.get_user_embedding(user_history)

        if user_embedding is None:
            # No valid items in history, return random
            candidates = candidate_items or self.all_item_ids
            return random.sample(candidates, min(k, len(candidates)))

        # Determine candidate items
        if candidate_items is None:
            candidate_items = self.all_item_ids

        # Exclude items
        if exclude_items:
            candidate_items = [
                item for item in candidate_items if item not in exclude_items
            ]

        # Score candidates
        scores = self.score_items(user_embedding, candidate_items)

        # Sort by score
        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return top-K
        recommendations = [item_id for item_id, score in ranked_items[:k]]

        return recommendations

    def recommend_with_scores(
        self,
        user_history: List[str],
        candidate_items: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Recommend top-K items with scores.

        Returns:
            recommendations: List of (item_id, score) tuples
        """
        user_embedding = self.get_user_embedding(user_history)

        if user_embedding is None:
            candidates = candidate_items or self.all_item_ids
            random_items = random.sample(candidates, min(k, len(candidates)))
            return [(item, 0.0) for item in random_items]

        if candidate_items is None:
            candidate_items = self.all_item_ids

        if exclude_items:
            candidate_items = [
                item for item in candidate_items if item not in exclude_items
            ]

        scores = self.score_items(user_embedding, candidate_items)
        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return ranked_items[:k]


def sample_negatives(
    all_items: List[str],
    positive_items: List[str],
    num_negatives: int,
    strategy: str = "random",
    item_popularity: Optional[Dict[str, int]] = None,
) -> List[str]:
    """
    Sample negative items for evaluation.

    Args:
        all_items: All available items
        positive_items: Positive items to exclude
        num_negatives: Number of negatives to sample
        strategy: 'random' or 'popularity'
        item_popularity: Popularity counts (required if strategy='popularity')

    Returns:
        negatives: List of negative item IDs
    """
    # Exclude positive items
    candidates = [item for item in all_items if item not in positive_items]

    if len(candidates) <= num_negatives:
        return candidates

    if strategy == "random":
        negatives = random.sample(candidates, num_negatives)
    elif strategy == "popularity":
        if item_popularity is None:
            raise ValueError("item_popularity required for popularity sampling")

        # Sample proportional to popularity
        candidate_pop = [item_popularity.get(item, 1) for item in candidates]
        total_pop = sum(candidate_pop)
        probs = [pop / total_pop for pop in candidate_pop]

        negatives = np.random.choice(
            candidates, size=num_negatives, replace=False, p=probs
        ).tolist()
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    return negatives


def load_user_histories(interactions_path: str) -> Dict[str, List[str]]:
    """
    Load user interaction histories from file.

    Args:
        interactions_path: Path to interactions JSONL file

    Returns:
        user_histories: Dictionary mapping user_id -> list of item_ids
    """
    user_histories = defaultdict(list)

    with open(interactions_path, "r") as f:
        for line in f:
            interaction = json.loads(line.strip())
            user_id = interaction["user_id"]
            item_id = interaction["parent_asin"]
            user_histories[user_id].append(item_id)

    return dict(user_histories)


def compute_item_popularity(interactions_path: str) -> Dict[str, int]:
    """
    Compute item popularity from interactions.

    Args:
        interactions_path: Path to interactions JSONL file

    Returns:
        popularity: Dictionary mapping item_id -> count
    """
    popularity = defaultdict(int)

    with open(interactions_path, "r") as f:
        for line in f:
            interaction = json.loads(line.strip())
            item_id = interaction["parent_asin"]
            popularity[item_id] += 1

    return dict(popularity)
