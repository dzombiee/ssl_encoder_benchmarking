"""
MLM (Masked Language Modeling) for BERT fine-tuning.
"""

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from transformers import AutoModel, AutoConfig  # type: ignore


class MLMModel(nn.Module):
    """
    Masked Language Modeling for pretraining BERT on item metadata.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 256,
        pooling_strategy: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pooling_strategy = pooling_strategy

        # Load BERT
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size

        # MLM head (for masked language modeling objective)
        self.mlm_head = nn.Linear(self.hidden_size, self.config.vocab_size)

        # Projection head (for downstream embeddings)
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, embedding_dim),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass for MLM.

        Args:
            input_ids: Input with some tokens masked [batch_size, seq_len]
            attention_mask: Attention mask
            labels: Target token IDs (with -100 for non-masked tokens)

        Returns:
            loss: MLM loss (if labels provided)
            embeddings: Pooled embeddings
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        token_embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # MLM prediction
        logits = self.mlm_head(token_embeddings)  # [batch, seq_len, vocab_size]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index is ignored
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        # Pool for sentence embedding
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

        # Project to embedding dimension
        embeddings = self.projection_head(pooled)

        if loss is not None:
            return loss, embeddings
        return embeddings

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
        embeddings = self.forward(input_ids, attention_mask, labels=None)

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings
