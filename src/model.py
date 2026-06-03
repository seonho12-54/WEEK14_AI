# -*- coding: utf-8 -*-
"""Mini GPT model components."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """Layer normalization over the last tensor dimension."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class GELU(nn.Module):
    """GELU activation with the tanh approximation used by GPT-style models."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """GPT block with pre-norm attention and feed-forward layers."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, drop_rate=drop_rate, qkv_bias=qkv_bias)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=drop_rate)
        self.resid_dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), causal_mask=causal_mask)
        x = x + self.resid_dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        vocab_size = config["vocab_size"]
        context_length = config["context_length"]
        emb_dim = config["emb_dim"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        drop_rate = config.get("drop_rate", 0.1)
        qkv_bias = config.get("qkv_bias", False)

        self.embedding = InputEmbedding(vocab_size, emb_dim, context_length, drop_rate=drop_rate)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=emb_dim,
                    n_heads=n_heads,
                    drop_rate=drop_rate,
                    qkv_bias=qkv_bias,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward_features(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embedding(idx)
        for block in self.blocks:
            x = block(x, causal_mask=True)
        return self.final_norm(x)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.forward_features(idx)
        logits = self.lm_head(hidden)
        if targets is None:
            return logits

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss, logits


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """Greedily append max_new_tokens tokens to idx."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -context_size:]
                logits = model(idx_cond)
                if isinstance(logits, tuple):
                    logits = logits[1]
                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                idx = torch.cat((idx, next_id), dim=1)
    finally:
        model.train(was_training)
    return idx
