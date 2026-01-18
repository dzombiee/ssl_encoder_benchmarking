"""
TSDAE: Transformer-based Sequential Denoising AutoEncoder.
"""

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from transformers import AutoModel, AutoConfig  # type: ignore
from typing import Optional


class TSDAEModel(nn.Module):
    """
    TSDAE for learning sentence embeddings via denoising.

    Architecture:
    - Encoder: BERT that encodes corrupted text
    - Decoder: Transformer decoder that reconstructs original text
    """

    def __init__(
        self,
        model_name: str = "miniLM-L6-v2",
        embedding_dim: int = 256,
        pooling_strategy: str = "mean",
        dropout: float = 0.1,
        tie_encoder_decoder: bool = True,
        use_projection_head: bool = False,
    ):
        super().__init__()

        self.pooling_strategy = pooling_strategy
        self.embedding_dim = embedding_dim
        self.use_projection_head = use_projection_head

        # Encoder (BERT)
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size

        # Decoder (can be tied with encoder or separate)
        if tie_encoder_decoder:
            self.decoder = self.encoder
        else:
            self.decoder = AutoModel.from_pretrained(model_name)

        # Projection head for embeddings
        if use_projection_head:
            self.projection_head: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, embedding_dim),
            )
        else:
            self.projection_head: Optional[nn.Sequential] = None
            self.embedding_dim = self.hidden_size

        # Reconstruction head
        self.reconstruction_head = nn.Linear(self.hidden_size, self.config.vocab_size)

    def pool_embeddings(self, token_embeddings, attention_mask):
        """Pool token embeddings to sentence embedding."""
        if self.pooling_strategy == "mean":
            mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        elif self.pooling_strategy == "cls":
            pooled = token_embeddings[:, 0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        return pooled

    def forward(self, input_ids, attention_mask, target_ids, target_attention_mask):
        """
        Forward pass for TSDAE.

        Args:
            input_ids: Corrupted input [batch_size, seq_len]
            attention_mask: Attention mask for corrupted input
            target_ids: Original target text [batch_size, seq_len]
            target_attention_mask: Attention mask for target

        Returns:
            loss: Reconstruction loss
            embeddings: Sentence embeddings
        """
        # Encode corrupted input
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        # Pool to sentence embedding
        pooled = self.pool_embeddings(encoder_outputs.last_hidden_state, attention_mask)

        # Project to embedding dimension
        if self.use_projection_head and self.projection_head is not None:
            embeddings = self.projection_head(pooled)
        else:
            embeddings = pooled

        # Decode to reconstruct original text
        # Use encoder's last hidden states as decoder input
        decoder_outputs = self.decoder(
            input_ids=target_ids, attention_mask=target_attention_mask, return_dict=True
        )

        # Predict tokens
        logits = self.reconstruction_head(decoder_outputs.last_hidden_state)

        # Compute reconstruction loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, self.config.vocab_size), target_ids.view(-1))

        return loss, embeddings

    def encode(self, input_ids, attention_mask, normalize=True):
        """
        Encode inputs to embeddings (inference).

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            normalize: Whether to L2 normalize

        Returns:
            embeddings: Sentence embeddings
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        pooled = self.pool_embeddings(encoder_outputs.last_hidden_state, attention_mask)
        if self.use_projection_head and self.projection_head is not None:
            embeddings = self.projection_head(pooled)
        else:
            embeddings = pooled
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def get_embedding(self, input_ids, attention_mask):
        """Get sentence embedding for inference."""
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        pooled = self.pool_embeddings(outputs.last_hidden_state, attention_mask)
        if self.use_projection_head and self.projection_head is not None:
            embedding = self.projection_head(pooled)
        else:
            embedding = pooled
        return F.normalize(embedding, p=2, dim=1)
