"""
Multi-view Contrastive Learning.
Treats different metadata fields (title, description, attributes) as separate views.
"""

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from .base_encoder import BaseEncoder  # type: ignore


class MultiViewContrastiveModel(nn.Module):
    """
    Multi-view contrastive learning for item metadata.

    Different metadata fields (title, description, attributes) are treated as
    separate views of the same item, similar to CLIP's image-text alignment.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 256,
        pooling_strategy: str = "mean",
        dropout: float = 0.1,
        temperature: float = 0.07,
        shared_encoder: bool = True,
        view_names: list | None = None,
    ):
        """
        Args:
            model_name: HuggingFace model name
            embedding_dim: Output embedding dimension
            pooling_strategy: Pooling strategy
            dropout: Dropout rate
            temperature: Temperature for contrastive loss
            shared_encoder: Whether to share encoder across views
            view_names: List of view names (e.g., ['title', 'description', 'attributes'])
        """
        super().__init__()

        self.temperature = temperature
        self.shared_encoder = shared_encoder
        self.view_names = view_names or ["title", "description", "attributes"]

        if shared_encoder:
            # Single encoder for all views
            self.encoder = BaseEncoder(
                model_name=model_name,
                embedding_dim=embedding_dim,
                pooling_strategy=pooling_strategy,
                dropout=dropout,
            )
        else:
            # Separate encoder for each view
            self.encoders = nn.ModuleDict(
                {
                    view_name: BaseEncoder(
                        model_name=model_name,
                        embedding_dim=embedding_dim,
                        pooling_strategy=pooling_strategy,
                        dropout=dropout,
                    )
                    for view_name in self.view_names
                }
            )

    def encode_view(self, view_name, input_ids, attention_mask):
        """Encode a single view."""
        if self.shared_encoder:
            embeddings = self.encoder(input_ids, attention_mask)
        else:
            embeddings = self.encoders[view_name](input_ids, attention_mask)

        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, views_dict):
        """
        Forward pass for multi-view contrastive learning.

        Args:
            views_dict: Dictionary with keys as view names and values as dicts with
                       'input_ids' and 'attention_mask'

        Returns:
            loss: Multi-view contrastive loss
            embeddings: Dictionary of view embeddings
        """
        # Encode each view
        view_embeddings = {}
        for view_name in self.view_names:
            if view_name in views_dict:
                input_ids = views_dict[view_name]["input_ids"]
                attention_mask = views_dict[view_name]["attention_mask"]
                embeddings = self.encode_view(view_name, input_ids, attention_mask)
                view_embeddings[view_name] = embeddings

        # Compute pairwise contrastive loss between all view pairs
        total_loss = 0
        num_pairs = 0

        view_list = list(view_embeddings.keys())
        for i in range(len(view_list)):
            for j in range(i + 1, len(view_list)):
                view1 = view_list[i]
                view2 = view_list[j]

                emb1 = view_embeddings[view1]
                emb2 = view_embeddings[view2]

                # Pairwise contrastive loss
                loss = self.pairwise_contrastive_loss(emb1, emb2)
                total_loss += loss
                num_pairs += 1

        # Average loss across all pairs
        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        return total_loss, view_embeddings

    def pairwise_contrastive_loss(self, embeddings1, embeddings2):
        """
        Contrastive loss between two views.

        Args:
            embeddings1: [batch_size, embedding_dim]
            embeddings2: [batch_size, embedding_dim]

        Returns:
            loss: Contrastive loss
        """
        batch_size = embeddings1.size(0)

        # Compute similarity matrix
        # [batch_size, batch_size]
        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature

        # Labels: diagonal elements are positives
        labels = torch.arange(batch_size).to(embeddings1.device)

        # Cross-entropy loss (both directions)
        loss_12 = F.cross_entropy(similarity_matrix, labels)
        loss_21 = F.cross_entropy(similarity_matrix.T, labels)

        loss = (loss_12 + loss_21) / 2

        return loss

    def encode(self, input_ids, attention_mask, view_name="title"):
        """
        Encode inputs to embeddings (inference).
        Uses a specific view encoder.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            view_name: Which view to encode with

        Returns:
            embeddings: Normalized embeddings
        """
        if self.shared_encoder:
            embeddings = self.encoder.encode(input_ids, attention_mask, normalize=True)
        else:
            if view_name not in self.encoders:
                view_name = self.view_names[0]  # Fallback to first view
            embeddings = self.encoders[view_name].encode(
                input_ids, attention_mask, normalize=True
            )

        return embeddings

    def encode_averaged(self, views_dict):
        """
        Encode multiple views and average them.
        Useful when combining information from all available views.

        Args:
            views_dict: Dictionary with view names and tokenized inputs

        Returns:
            embeddings: Averaged embeddings across views
        """
        all_embeddings = []

        for view_name in self.view_names:
            if view_name in views_dict:
                input_ids = views_dict[view_name]["input_ids"]
                attention_mask = views_dict[view_name]["attention_mask"]
                embeddings = self.encode_view(view_name, input_ids, attention_mask)
                all_embeddings.append(embeddings)

        if not all_embeddings:
            raise ValueError("No valid views found in views_dict")

        # Average embeddings
        averaged = torch.stack(all_embeddings).mean(dim=0)

        # Normalize
        averaged = F.normalize(averaged, p=2, dim=1)

        return averaged
