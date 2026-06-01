# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: 마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        #raise NotImplementedError("LayerNorm.forward를 구현하세요.")
        #x의 마지막 차원 평균과 분산을 구하기, keepdim=True는 shape를 유지한다는 뜻
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        #표준화 작업 
        x_hat = (x-mean) / torch.sqrt(var + self.eps)
        #표준화 된 값 재조정, gamma : scale, beta : shift
        out = self.gamma * x_hat + self.beta

        return out


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        #raise NotImplementedError("GELU.forward를 구현하세요.")

        return 0.5 * x * (
            1.0 + torch.tanh(
                (2.0 / torch.pi) ** 0.5 * (x + 0.044715 * torch.pow(x, 3))
            )
        )
        

class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # TODO: d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.
        #raise NotImplementedError("FeedForward.__init__을 구현하세요.")
        #차원을 mult만큼 늘리고, GELU를 통과시키고, 다시 축소 후 dropout적용
        #token vector를 4배 차원으로 확장해 더 많은 특징 조합을 만드려고
        self.net = nn.Sequential( #sequential : layer를 순서대로 묶어줌
            nn.Linear(d_model, mult * d_model),
            GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: FeedForward 네트워크를 통과시킵니다."""
        #raise NotImplementedError("FeedForward.forward를 구현하세요.")
        return self.net

class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        # TODO: attention, ffn, layernorm, dropout을 정의하세요.
        #raise NotImplementedError("TransformerBlock.__init__을 구현하세요.")
        #입력 x를 layernorm 한 뒤 attention에 넣고, 그 결과를 원래 x에 더함
        #다시 layernorm한 뒤 feedforward에 넣고, 그 결과를 원래 값에 더함
        self.attention = MultiHeadAttention(
            #attention객체를 만들기 위해 attention init에서 쓸 변수 넘겨주기
            d_model=d_model,
            n_heads=n_heads,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
        )

        self.feedforward = FeedForward(
            d_model=d_model,
            dropout=drop_rate,
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        #attention과 ff내부에 dropout이 이미 있어서, 꼭 필요한 것은 아님
        self.drop_resid = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """TODO: attention과 ffn을 residual connection으로 연결합니다."""
        #raise NotImplementedError("TransformerBlock.forward를 구현하세요.")
        # x = self.nor1(x)
        # x = self.attention(x)
        # x = x + self.drop_resid(x)    이렇게 하면 원래 x를 보존하지 못해서 residual 실패

        attention_out = self.attention(self.norm1(x))
        x = x + self.drop_resid(attention_out)

        feedforward_out = self.feedforward(self.norm2(x))
        x = x + self.drop_resid(feedforward_out)

        return x

class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config #모델을 만들 때 필요한 설정값들을 모아둔 딕셔너리
        # TODO: embedding, blocks, final layernorm, lm_head를 정의하세요.
        #raise NotImplementedError("GPTModel.__init__을 구현하세요.")
        #config는 테스트나 트레이닝 같은 곳에서 생성 후, model = GPTmodel(config) 이런 식으로 사용
        vocab_size = config["vocab_size"]
        context_length = config["context_length"]
        emb_dim = config["emb_dim"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        drop_rate = config["drop_rate"]
        qkv_bias = config.get("qkv_bias", False)
        #token id를 token_embedding + position_embedding으로 변환
        self.embedding = InputEmbedding(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            context_length=context_length,
            drop_rate=drop_rate,
        )
        #TransformerBlock 여러 개를 순서대로 통과시킴
        self.blocks = nn.Sequential(
            #n_layers개수 만큼 transformerblock을 만들고, 각각의 블럭에 아래 변수를 넣어줌
            *[
                TransformerBlock(
                    d_model=emb_dim,
                    n_heads=n_heads,
                    drop_rate=drop_rate,
                    qkv_bias=qkv_bias,
                )
                for _ in range(n_layers)
            ]
        )
        #마지막 출력 벡터를 한 번 더 안정화, transformer block 여러층을 지나면서 값의 분포가 흔들릴 수도 있음
        self.final_norm = LayerNorm(emb_dim)
        #각 token벡터를 다음번에 올 token점수 벡터로 변환, (vocab_size 차원 벡터임), logits를 만듦
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: logits를 만들고, targets가 있으면 cross entropy loss도 함께 반환합니다.

        Returns:
            targets가 None이면 logits
            targets가 있으면 (loss, logits)
        """
        #raise NotImplementedError("GPTModel.forward를 구현하세요.")
        #embedding, blocks, final_norm, lm_head 순서대로
        x = self.embedding(idx)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        #targets는 forward가 호출할 때마다 외부에서 들어옴, 객체가 계속 기억할 필요가 없음 self.가 아님
        if targets is None:
            return logits
        #tatgers이 있다면 loss계산
        loss = F.cross_entropy(
            #(B, T, vocab_size)를 (B*T, vocab_size)로 바꿈
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        return loss, logits
        
def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    raise NotImplementedError("generate_text_simple을 구현하세요.")
