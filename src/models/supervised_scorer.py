"""
Light supervised scorer (MLP) for downstream fine-tuning.
"""

import torch  # type: ignore
import torch.nn as nn  # type: ignore


class SupervisedScorer(nn.Module):
    """
    Simple MLP scorer for supervised fine-tuning.

    Takes user and item embeddings and predicts interaction probability.
    """

    def __init__(
        self, embedding_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.3
    ):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()

        # Input: [user_emb, item_emb, user_emb * item_emb]
        input_dim = embedding_dim * 3

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_embeddings, item_embeddings):
        """
        Forward pass.

        Args:
            user_embeddings: [batch_size, embedding_dim]
            item_embeddings: [batch_size, embedding_dim]

        Returns:
            scores: [batch_size, 1] - probability of interaction
        """
        # Concatenate features
        elementwise_product = user_embeddings * item_embeddings
        features = torch.cat(
            [user_embeddings, item_embeddings, elementwise_product], dim=-1
        )

        # Predict
        scores = self.mlp(features)

        return scores

    def predict(self, user_embeddings, item_embeddings):
        """Predict scores (inference mode)."""
        self.eval()
        with torch.no_grad():
            scores = self.forward(user_embeddings, item_embeddings)
        return scores


class SupervisedRecommender:
    """
    Recommender using supervised scorer.
    """

    def __init__(
        self,
        item_embeddings: dict,
        scorer_model: SupervisedScorer,
        device: torch.device,
    ):
        """
        Args:
            item_embeddings: Dictionary of item_id -> embedding
            scorer_model: Trained scorer model
            device: Device to run on
        """
        self.item_embeddings = item_embeddings
        self.scorer = scorer_model
        self.device = device

        # Convert embeddings to tensors
        self.item_ids = list(item_embeddings.keys())
        self.item_embedding_matrix = torch.FloatTensor(
            [item_embeddings[item_id] for item_id in self.item_ids]
        ).to(device)

    def get_user_embedding(self, user_history, aggregation="mean"):
        """Compute user embedding from history."""
        history_embeddings = []
        for item_id in user_history:
            if item_id in self.item_embeddings:
                history_embeddings.append(self.item_embeddings[item_id])

        if len(history_embeddings) == 0:
            return None

        history_embeddings = torch.FloatTensor(history_embeddings).to(self.device)

        if aggregation == "mean":
            user_embedding = history_embeddings.mean(dim=0)
        else:
            user_embedding = history_embeddings.mean(dim=0)

        return user_embedding

    def recommend(self, user_history, candidate_items=None, k=10):
        """
        Recommend top-K items.

        Args:
            user_history: List of item IDs
            candidate_items: List of candidate item IDs (if None, use all)
            k: Number of recommendations

        Returns:
            recommendations: List of top-K item IDs
        """
        # Get user embedding
        user_embedding = self.get_user_embedding(user_history)

        if user_embedding is None:
            return []

        # Get candidate item embeddings
        if candidate_items is None:
            candidate_ids = self.item_ids
            candidate_embeddings = self.item_embedding_matrix
        else:
            candidate_ids = [
                item for item in candidate_items if item in self.item_embeddings
            ]
            candidate_embeddings = torch.FloatTensor(
                [self.item_embeddings[item_id] for item_id in candidate_ids]
            ).to(self.device)

        # Expand user embedding to match candidates
        user_embedding_expanded = user_embedding.unsqueeze(0).expand(
            len(candidate_ids), -1
        )

        # Score candidates
        scores = self.scorer.predict(user_embedding_expanded, candidate_embeddings)
        scores = scores.squeeze().cpu().numpy()

        # Get top-K
        top_indices = scores.argsort()[::-1][:k]
        recommendations = [candidate_ids[i] for i in top_indices]

        return recommendations
