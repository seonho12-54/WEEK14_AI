# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구현할 구조:
    - token embedding: nn.Embedding(vocab_size, emb_dim)
    - position embedding: nn.Embedding(context_length, emb_dim)
    - token embedding + position embedding
    - dropout
    """

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
        # TODO: token_embedding, position_embedding, dropout을 정의하세요.
        #raise NotImplementedError("InputEmbedding.__init__을 구현하세요.")
        #nn.Embedding은 pytoch에 구현되어 있음, vocab 개수만큼의 토큰에 대해 emb_dim 차원 벡터 하나씩 준비
        #token embedding은 nn.Embedding이 가진 token id별 학습 가능한 벡터, token id를 넣으면 해당 벡터가 출력
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        #각 token 위치(0 ~ context_length-1)를 emb_dim 차원 위치 벡터로 변환하는 임베딩 층
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        #raise NotImplementedError("InputEmbedding.forward를 구현하세요.")
        token_emb = self.token_embedding(x)
        batch_size, seq_len = x.shape
        position_ids = torch.arange(seq_len)
        position_emb = self.position_embedding(position_ids)
        out = token_emb + position_emb
        out = self.dropout(out)

        return out