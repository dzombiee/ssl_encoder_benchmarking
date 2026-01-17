"""
Base encoder model with projection head.
"""

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from transformers import AutoModel, AutoConfig  # type: ignore


class BaseEncoder(nn.Module):
    """
    Base encoder with BERT backbone and optional projection head.

    For fair comparison with SBERT baseline:
    - use_projection_head=False: Output 384 dims (same as SBERT)
    - use_projection_head=True: Output custom dims with projection
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        pooling_strategy: str = "mean",
        dropout: float = 0.1,
        use_projection_head: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace model name
            embedding_dim: Output embedding dimension (ignored if use_projection_head=False)
            pooling_strategy: How to pool token embeddings ('mean' or 'cls')
            dropout: Dropout rate
            use_projection_head: If False, output = pooled BERT (for fair SBERT comparison)
        """
        super().__init__()

        self.pooling_strategy = pooling_strategy
        self.use_projection_head = use_projection_head

        # Load BERT model
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # Get BERT hidden size
        self.hidden_size = self.config.hidden_size

        # Set output dimension
        if use_projection_head:
            self.output_dim = embedding_dim
            # Projection head (BERT hidden -> embedding_dim)
            self.projection_head: nn.Sequential | None = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, embedding_dim),
            )
        else:
            # No projection: output = BERT hidden size (384 for all-MiniLM-L6-v2)
            self.output_dim = self.hidden_size
            self.projection_head: nn.Sequential | None = None

    def forward(self, input_ids, attention_mask, return_hidden=False):
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_hidden: If True, return both hidden states and embeddings

        Returns:
            embeddings: Projected embeddings [batch_size, embedding_dim]
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        # Pool token embeddings
        if self.pooling_strategy == "mean":
            # Mean pooling (excluding padding tokens)
            token_embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden]
            mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        elif self.pooling_strategy == "cls":
            # Use [CLS] token
            pooled = outputs.last_hidden_state[:, 0]  # [batch, hidden]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Project to embedding dimension (if projection head enabled)
        if self.projection_head is not None:
            embeddings = self.projection_head(pooled)
        else:
            embeddings = pooled  # No projection, use pooled BERT directly

        if return_hidden:
            return embeddings, pooled
        return embeddings

    def encode(self, input_ids, attention_mask, normalize=True):
        """
        Encode inputs to embeddings (inference mode).

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            normalize: Whether to L2 normalize embeddings

        Returns:
            embeddings: L2-normalized embeddings
        """
        embeddings = self.forward(input_ids, attention_mask)
        # If forward returns a tuple, extract embeddings
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]
        if normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for aggregating user history.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1, bias=False),
        )

    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch_size, num_items, embedding_dim]

        Returns:
            pooled: [batch_size, embedding_dim]
        """
        # Compute attention scores
        scores = self.attention(embeddings)  # [batch, num_items, 1]
        weights = torch.softmax(scores, dim=1)  # [batch, num_items, 1]

        # Weighted sum
        pooled = torch.sum(embeddings * weights, dim=1)  # [batch, embedding_dim]

        return pooled
