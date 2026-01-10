"""
SimCSE model for contrastive learning with dropout augmentation.
"""

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from .base_encoder import BaseEncoder  # type: ignore


class SimCSEModel(nn.Module):
    """
    SimCSE: Simple Contrastive Learning of Sentence Embeddings.
    Uses dropout as augmentation - same input passed through model twice with different dropout masks.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 256,
        pooling_strategy: str = "mean",
        dropout: float = 0.1,
        temperature: float = 0.05,
    ):
        super().__init__()

        self.encoder = BaseEncoder(
            model_name=model_name,
            embedding_dim=embedding_dim,
            pooling_strategy=pooling_strategy,
            dropout=dropout,
        )

        self.temperature = temperature

    def forward(self, input_ids, attention_mask, num_views=2):
        """
        Forward pass for contrastive learning.

        Args:
            input_ids: [batch_size * num_views, seq_len]
            attention_mask: [batch_size * num_views, seq_len]
            num_views: Number of views per item (default 2)

        Returns:
            loss: Contrastive loss
            embeddings: Item embeddings
        """
        # Encode all views
        embeddings = self.encoder(input_ids, attention_mask)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Reshape: [batch_size * num_views, dim] -> [batch_size, num_views, dim]
        batch_size = embeddings.size(0) // num_views
        embeddings_reshaped = embeddings.view(batch_size, num_views, -1)

        # Compute contrastive loss
        loss = self.contrastive_loss(embeddings_reshaped)

        return loss, embeddings

    def contrastive_loss(self, embeddings):
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

        Args:
            embeddings: [batch_size, num_views, embedding_dim]

        Returns:
            loss: Scalar contrastive loss
        """
        batch_size, num_views, dim = embeddings.shape

        # Flatten to [batch_size * num_views, dim]
        embeddings_flat = embeddings.view(-1, dim)

        # Compute similarity matrix: [batch_size * num_views, batch_size * num_views]
        similarity_matrix = (
            torch.matmul(embeddings_flat, embeddings_flat.T) / self.temperature
        )

        # Create labels: positive pairs are (i, i+batch_size) for i in [0, batch_size)
        # This assumes num_views=2 and views are arranged as [view1_batch, view2_batch]
        labels = torch.arange(batch_size).to(embeddings.device)
        labels = torch.cat(
            [labels + batch_size, labels]
        )  # [0->bs, 1->bs+1, ..., bs->0, bs+1->1, ...]

        # Mask to remove self-similarity
        mask = torch.eye(
            batch_size * num_views, dtype=torch.bool, device=embeddings.device
        )
        similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))

        # Cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss

    def encode(self, input_ids, attention_mask):
        """Encode inputs to embeddings (inference)."""
        return self.encoder.encode(input_ids, attention_mask, normalize=True)


class SimCLRModel(nn.Module):
    """
    SimCLR-style contrastive learning with text augmentations.
    Similar to SimCSE but designed for heavier augmentations.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 256,
        pooling_strategy: str = "mean",
        dropout: float = 0.1,
        temperature: float = 0.1,
    ):
        super().__init__()

        self.encoder = BaseEncoder(
            model_name=model_name,
            embedding_dim=embedding_dim,
            pooling_strategy=pooling_strategy,
            dropout=dropout,
        )

        self.temperature = temperature

    def forward(self, input_ids, attention_mask, num_views=2):
        """Forward pass - same as SimCSE."""
        embeddings = self.encoder(input_ids, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        batch_size = embeddings.size(0) // num_views
        embeddings_reshaped = embeddings.view(batch_size, num_views, -1)

        loss = self.contrastive_loss(embeddings_reshaped)

        return loss, embeddings

    def contrastive_loss(self, embeddings):
        """NT-Xent loss (same as SimCSE)."""
        batch_size, num_views, dim = embeddings.shape
        embeddings_flat = embeddings.view(-1, dim)

        similarity_matrix = (
            torch.matmul(embeddings_flat, embeddings_flat.T) / self.temperature
        )

        labels = torch.arange(batch_size).to(embeddings.device)
        labels = torch.cat([labels + batch_size, labels])

        mask = torch.eye(
            batch_size * num_views, dtype=torch.bool, device=embeddings.device
        )
        similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))

        loss = F.cross_entropy(similarity_matrix, labels)

        return loss

    def encode(self, input_ids, attention_mask):
        """Encode inputs to embeddings (inference)."""
        return self.encoder.encode(input_ids, attention_mask, normalize=True)
