# -*- coding: utf-8 -*-
"""Token embedding + position embedding."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """Convert token IDs to Transformer input vectors."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        if seq_len > self.context_length:
            raise ValueError("sequence length exceeds context_length")

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        return self.dropout(token_emb + pos_emb)
