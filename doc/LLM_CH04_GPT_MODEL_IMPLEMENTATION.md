# 4장 GPT 모델 구현 흐름 정리

## 0. 4장의 핵심 한 줄

**4장은 2장에서 만든 입력 임베딩과 3장에서 만든 attention을 조립해서, 진짜 GPT 모델의 기본 구조를 만드는 장이다.**

우리 과제에서는 거의 그대로 `src/model.py` 구현으로 이어진다.

```text
token ids
-> InputEmbedding
-> TransformerBlock x n_layers
-> final LayerNorm
-> lm_head
-> logits
-> 다음 token 예측
```

---

## 1. 4장에서 우리 구현에 필요한 부분만 고르기

교재 4장은 GPT 모델 전체 구조를 밑바닥부터 조립한다.

우리 과제에서 직접 구현해야 하는 부분은 아래와 같다.

| 교재 내용 | 과제 구현 위치 | 중요도 |
| --- | --- | --- |
| Layer Normalization | `src/model.py`의 `LayerNorm` | 매우 중요 |
| GELU | `src/model.py`의 `GELU` | 중요 |
| FeedForward Network | `src/model.py`의 `FeedForward` | 매우 중요 |
| Shortcut / Residual Connection | `TransformerBlock.forward` | 매우 중요 |
| TransformerBlock | `src/model.py`의 `TransformerBlock` | 매우 중요 |
| GPTModel | `src/model.py`의 `GPTModel` | 매우 중요 |
| logits | `GPTModel.forward` 출력 | 매우 중요 |
| greedy generation | `generate_text_simple` | 중요 |

반대로 4장에서 거대한 GPT-2 모델 크기 비교나 전체 파라미터 규모 설명은 흐름 이해용이다. 구현할 때는 위 표가 더 중요하다.

---

## 2. 4장은 2장, 3장과 어떻게 연결되나

2장에서는 문장을 모델이 읽을 수 있는 숫자와 벡터로 바꿨다.

```text
문장
-> BPETokenizer.encode()
-> token ids
-> token embedding + position embedding
-> input embedding
```

3장에서는 그 input embedding에 attention을 적용했다.

```text
input embedding
-> Q, K, V
-> causal multi-head self-attention
-> 문맥이 섞인 token vector
```

4장은 여기서 한 단계 더 나아가 attention 하나만 쓰지 않고, GPT에서 쓰는 블록 구조로 묶는다.

```text
input embedding
-> TransformerBlock
-> TransformerBlock
-> ...
-> vocab_size개의 점수
```

즉, 4장은 **GPT 모델의 몸통을 만드는 장**이다.

---

## 3. GPT 모델 전체 구조

GPT는 크게 보면 아래 구조다.

```text
token ids: (B, T)
    |
    v
InputEmbedding
    |
    v
x: (B, T, C)
    |
    v
TransformerBlock x n_layers
    |
    v
final LayerNorm
    |
    v
lm_head
    |
    v
logits: (B, T, vocab_size)
```

여기서 기호는 다음 뜻이다.

| 기호 | 뜻 |
| --- | --- |
| B | batch size, 한 번에 처리하는 문장 묶음 개수 |
| T | sequence length, token 개수 |
| C | embedding dimension, token vector의 차원 |
| vocab_size | vocabulary에 들어 있는 token 종류 수 |

예를 들어 설정이 아래와 같다면,

```text
B = 2
T = 16
C = 64
vocab_size = 1000
```

출력 logits shape는 아래처럼 된다.

```text
(2, 16, 1000)
```

뜻은 이렇다.

```text
2개 문장 각각에 대해
16개 token 위치마다
1000개 token 후보 점수를 낸다.
```

---

## 4. Logits란 무엇인가

`logits`는 softmax를 통과하기 전의 원점수다.

MNIST에서 숫자 0~9에 대한 점수를 만든 뒤 softmax를 적용했던 것과 같다.

LLM에서는 숫자 class 10개 대신 vocabulary 전체 token에 대한 점수를 만든다.

```text
MNIST:
logits -> 숫자 0~9 점수

GPT:
logits -> 다음 token 후보 전체 점수
```

예를 들어 vocabulary 크기가 1000이면, 각 token 위치마다 1000개의 점수가 나온다.

```text
logits[0, 3]
= 첫 번째 문장의 네 번째 위치에서 다음 token 후보 1000개에 대한 점수
```

### 꼭 외우기

**logits는 softmax 전 점수이고, GPT에서는 다음 token 후보 전체에 대한 점수다.**

---

## 5. LayerNorm

LayerNorm은 각 token vector의 값 분포를 정리하는 층이다.

입력 shape가 아래와 같다고 하자.

```text
x: (B, T, C)
```

LayerNorm은 마지막 차원 `C` 기준으로 평균과 분산을 구한다.

```text
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
x_hat = (x - mean) / sqrt(var + eps)
out = gamma * x_hat + beta
```

여기서 중요한 점은 batch 전체를 섞는 것이 아니라, **각 token vector 내부의 C개 값만 정규화한다**는 것이다.

예시:

```text
x[0, 0] = 첫 번째 문장의 첫 번째 token vector
이 vector 안의 64개 값 평균/분산을 계산해서 정규화한다.
```

### BatchNorm과 차이

MNIST에서 봤던 BatchNorm은 batch 방향을 기준으로 통계를 낸다.

LayerNorm은 한 token vector 내부의 feature 방향을 기준으로 통계를 낸다.

| 구분 | 기준 | LLM에서 쓰기 좋은 이유 |
| --- | --- | --- |
| BatchNorm | batch 방향 | 문장 길이, batch 구성에 영향을 많이 받음 |
| LayerNorm | 마지막 feature 차원 | batch 크기와 상관없이 안정적 |

GPT에서는 BatchNorm이 아니라 LayerNorm을 사용한다.

### 꼭 외우기

**LayerNorm은 `(B, T, C)`에서 마지막 차원 `C` 기준으로 각 token vector를 정규화한다.**

---

## 6. GELU

GELU는 GPT의 FeedForward Network에서 사용하는 활성화 함수다.

MNIST에서는 주로 ReLU를 사용했다.

```text
ReLU(x) = max(0, x)
```

ReLU는 음수면 바로 0으로 끊는다.

GELU는 값을 부드럽게 통과시킨다.

```text
GELU(x) ~= 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
```

외울 필요는 공식 전체보다 의미가 중요하다.

```text
ReLU: 음수는 딱 잘라냄
GELU: 작거나 음수인 값도 확률적으로 부드럽게 줄임
```

### ReLU와 GELU 비교

| 함수 | 특징 | 어디서 봤나 |
| --- | --- | --- |
| ReLU | 빠르고 단순함, 음수를 0으로 만듦 | MNIST |
| GELU | 더 부드럽고 Transformer 계열에서 자주 사용 | GPT |

### 꼭 외우기

**GPT의 FFN에서는 ReLU보다 부드러운 GELU를 사용한다.**

---

## 7. FeedForward Network

FeedForward Network는 attention 다음에 각 token vector를 한 번 더 가공하는 작은 MLP다.

구조는 보통 아래처럼 쓴다.

```text
d_model
-> 4 * d_model
-> GELU
-> d_model
-> Dropout
```

과제의 `FeedForward`는 이런 구조다.

```python
nn.Linear(d_model, mult * d_model)
GELU()
nn.Linear(mult * d_model, d_model)
nn.Dropout(dropout)
```

예를 들어 `d_model = 64`, `mult = 4`라면:

```text
64 -> 256 -> 64
```

shape 흐름은 아래와 같다.

```text
x:   (B, T, 64)
중간: (B, T, 256)
out: (B, T, 64)
```

즉, FeedForward는 shape를 최종적으로 유지한다.

### Attention과 FFN의 역할 차이

| 구성 요소 | 역할 |
| --- | --- |
| Attention | token끼리 정보를 섞는다 |
| FeedForward | 각 token vector 자체를 더 깊게 변환한다 |

조금 더 쉽게 말하면:

```text
Attention: 주변 token을 보고 문맥을 섞음
FFN: 섞인 정보를 바탕으로 각 token의 표현을 다듬음
```

### 꼭 외우기

**TransformerBlock 안에서 attention은 token 사이 관계를 처리하고, FFN은 각 token vector를 더 깊게 가공한다.**

---

## 8. Residual Connection

Residual connection은 층의 출력에 원래 입력을 더하는 구조다.

```text
out = x + layer(x)
```

TransformerBlock에서는 attention 뒤와 FFN 뒤에 각각 residual connection이 들어간다.

```text
x = x + attention(layernorm(x))
x = x + ffn(layernorm(x))
```

이걸 쓰는 이유는 깊은 모델에서 gradient가 잘 흐르게 하기 위해서다.

층이 깊어지면 역전파 중 gradient가 작아져 앞쪽 층까지 잘 전달되지 않을 수 있다. residual connection은 원래 입력이 그대로 지나갈 길을 만들어서 학습을 안정시킨다.

### MNIST와 연결

MNIST에서 층이 깊어지면 학습이 어려워질 수 있다는 이야기를 했다.

GPT는 TransformerBlock을 여러 개 쌓기 때문에 이런 문제가 더 커진다.

그래서 residual connection이 거의 필수다.

### 꼭 외우기

**Residual connection은 `x + layer(x)` 구조로, 깊은 모델에서 gradient가 잘 흐르도록 돕는다.**

---

## 9. TransformerBlock

TransformerBlock은 GPT의 핵심 반복 단위다.

하나의 TransformerBlock은 보통 아래 순서로 작동한다.

```text
x
-> LayerNorm
-> MultiHeadAttention
-> Dropout
-> Residual Add
-> LayerNorm
-> FeedForward
-> Dropout
-> Residual Add
```

과제 주석에는 이렇게 적혀 있다.

```text
GPT block:
LayerNorm -> Causal Self-Attention -> residual
LayerNorm -> FeedForward -> residual
```

구현은 보통 이런 모양이 된다.

```python
shortcut = x
x = self.norm1(x)
x = self.att(x, causal_mask=causal_mask)
x = self.drop_resid(x)
x = x + shortcut

shortcut = x
x = self.norm2(x)
x = self.ffn(x)
x = self.drop_resid(x)
x = x + shortcut
```

이 구조를 Pre-LayerNorm 구조라고 볼 수 있다.

LayerNorm을 attention/FFN 전에 적용하기 때문이다.

### shape 흐름

TransformerBlock은 입력과 출력 shape가 같다.

```text
입력: (B, T, C)
출력: (B, T, C)
```

shape가 같아야 residual add가 가능하다.

```text
x + attention(x)
```

에서 두 텐서의 shape가 같아야 하기 때문이다.

### 꼭 외우기

**TransformerBlock은 attention과 FFN을 residual connection으로 묶은 GPT의 반복 단위다.**

---

## 10. GPTModel

GPTModel은 지금까지 만든 부품을 전체 모델로 조립한다.

과제의 `GPTModel.__init__`에서 필요한 부품은 아래와 같다.

```text
InputEmbedding
TransformerBlock 여러 개
final LayerNorm
lm_head
```

config 예시는 아래와 같다.

```python
config = {
    "vocab_size": 1000,
    "context_length": 64,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
```

각 값의 의미:

| config key | 의미 |
| --- | --- |
| `vocab_size` | token 종류 수 |
| `context_length` | 모델이 한 번에 보는 token 길이 |
| `emb_dim` | token vector 차원 |
| `n_heads` | attention head 개수 |
| `n_layers` | TransformerBlock 개수 |
| `drop_rate` | dropout 비율 |
| `qkv_bias` | Q/K/V Linear에 bias를 쓸지 여부 |

`GPTModel.forward`의 큰 흐름은 아래와 같다.

```text
idx: (B, T)
-> embedding(idx)
-> blocks(x)
-> final_norm(x)
-> lm_head(x)
-> logits: (B, T, vocab_size)
```

### lm_head란 무엇인가

`lm_head`는 language modeling head의 줄임말이다.

마지막 hidden vector를 vocabulary 전체 점수로 바꾸는 Linear layer다.

```text
hidden vector: (B, T, emb_dim)
-> lm_head
logits:        (B, T, vocab_size)
```

MNIST로 비유하면 마지막 `Affine(10)`과 비슷하다.

다만 MNIST는 class가 10개였고, GPT는 class가 vocab_size개다.

### 꼭 외우기

**lm_head는 각 token 위치의 hidden vector를 다음 token 후보 점수로 바꾸는 마지막 Linear layer다.**

---

## 11. Loss 계산

`GPTModel.forward`는 targets가 없으면 logits만 반환한다.

```python
logits = model(idx, targets=None)
```

targets가 있으면 loss와 logits를 같이 반환한다.

```python
loss, logits = model(idx, targets=targets)
```

여기서 targets는 2장에서 만든 input보다 한 칸 밀린 정답 token ID다.

```text
input:  [10, 11, 12, 13]
target: [11, 12, 13, 14]
```

logits shape:

```text
(B, T, vocab_size)
```

targets shape:

```text
(B, T)
```

PyTorch의 `cross_entropy`는 보통 아래 형태를 기대한다.

```text
logits:  (N, C)
targets: (N,)
```

그래서 batch와 token 위치를 펼친다.

```python
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
)
```

shape 예시:

```text
logits:  (2, 16, 1000) -> (32, 1000)
targets: (2, 16)       -> (32,)
```

뜻은 32개의 token 위치 각각에서 정답 token ID를 맞히는 문제로 바꾸는 것이다.

### 꼭 외우기

**GPT 학습은 모든 위치에서 다음 token을 맞히는 분류 문제이고, loss는 cross entropy로 계산한다.**

---

## 12. generate_text_simple

`generate_text_simple`은 학습된 모델로 token을 하나씩 이어 붙이는 함수다.

가장 단순한 방식은 greedy decoding이다.

greedy decoding은 매번 점수가 가장 높은 token 하나를 고른다.

흐름:

```text
1. 현재 token ids를 모델에 넣는다.
2. logits를 얻는다.
3. 마지막 위치의 logits만 본다.
4. argmax로 가장 점수가 높은 다음 token을 고른다.
5. 기존 token ids 뒤에 붙인다.
6. max_new_tokens만큼 반복한다.
```

중요한 부분은 마지막 위치의 logits만 본다는 것이다.

```python
logits = model(idx_cond)
logits = logits[:, -1, :]
idx_next = torch.argmax(logits, dim=-1, keepdim=True)
idx = torch.cat((idx, idx_next), dim=1)
```

왜 마지막 위치만 볼까?

GPT는 지금까지의 token을 보고 다음 token을 예측하는 모델이다.

따라서 새 token을 생성할 때는 현재 sequence의 마지막 위치에서 나온 예측만 필요하다.

### context_size가 필요한 이유

GPT는 최대 `context_length`까지만 볼 수 있다.

sequence가 너무 길어지면 뒤쪽 일부만 잘라서 넣는다.

```python
idx_cond = idx[:, -context_size:]
```

예를 들어 context_size가 64면 마지막 64개 token만 모델에 넣는다.

### 꼭 외우기

**텍스트 생성은 logits의 마지막 token 위치에서 다음 token을 고르고, 그 token을 뒤에 붙이는 반복 과정이다.**

---

## 13. 4장 구현 순서

`src/model.py`는 아래 순서대로 구현하는 것이 좋다.

### 1단계. LayerNorm

먼저 shape가 유지되는지 확인한다.

```bash
pytest tests/test_model.py -v -k layernorm
```

핵심:

```text
마지막 차원 평균/분산
gamma, beta 적용
shape 유지
```

### 2단계. GELU

입력과 출력 shape가 같으면 된다.

```bash
pytest tests/test_model.py -v -k gelu
```

핵심:

```text
tanh 근사식
shape 유지
```

### 3단계. FeedForward

`d_model -> 4*d_model -> d_model` 구조를 만든다.

```bash
pytest tests/test_model.py -v -k feedforward
```

핵심:

```text
Linear
GELU
Linear
Dropout
```

### 4단계. TransformerBlock

3장에서 구현한 `MultiHeadAttention`을 가져와 묶는다.

```bash
pytest tests/test_model.py -v -k transformer
```

핵심:

```text
LayerNorm
Attention
Residual
LayerNorm
FFN
Residual
```

### 5단계. GPTModel

모델 전체를 조립한다.

```bash
pytest tests/test_model.py -v -k gpt
```

핵심:

```text
InputEmbedding
TransformerBlock 반복
final LayerNorm
lm_head
cross entropy loss
```

### 6단계. generate_text_simple

가장 단순한 greedy generation을 구현한다.

```bash
pytest tests/test_model.py -v -k generate
```

핵심:

```text
마지막 token logits
argmax
concat
반복
```

---

## 14. 구현할 때 자주 헷갈리는 부분

### 1. LayerNorm은 batch 기준이 아니다

`dim=-1` 기준이다.

```text
O: x.mean(dim=-1, keepdim=True)
X: x.mean(dim=0)
```

### 2. TransformerBlock은 shape를 바꾸면 안 된다

입력과 출력 모두 `(B, T, C)`여야 한다.

Residual connection을 하려면 shape가 같아야 한다.

### 3. lm_head 출력은 emb_dim이 아니라 vocab_size다

```text
입력:  (B, T, emb_dim)
출력:  (B, T, vocab_size)
```

### 4. loss 계산 전 logits를 펼쳐야 한다

```text
(B, T, vocab_size) -> (B*T, vocab_size)
(B, T)             -> (B*T)
```

### 5. generate에서는 마지막 위치 logits만 쓴다

```text
logits[:, -1, :]
```

이 값이 “현재 문맥 다음에 올 token 후보 점수”다.

---

## 15. 4장 핵심 흐름 다시 보기

아래 하나만 머리에 남겨도 4장의 큰 그림은 잡힌다.

```text
token ids
-> embedding
-> TransformerBlock 반복
    -> LayerNorm
    -> Causal Multi-Head Attention
    -> Residual
    -> LayerNorm
    -> FeedForward
    -> Residual
-> final LayerNorm
-> lm_head
-> logits
-> cross entropy loss 또는 next token 생성
```

---

## 16. 4장 체크리스트

아래 질문에 답할 수 있으면 4장 구현 준비가 된 것이다.

| 질문 | 답할 수 있나 |
| --- | --- |
| GPTModel의 입력 `idx` shape는 무엇인가? |  |
| InputEmbedding 출력 shape는 무엇인가? |  |
| LayerNorm은 어느 차원을 기준으로 평균/분산을 구하는가? |  |
| GELU는 ReLU와 무엇이 다른가? |  |
| FeedForward는 왜 `d_model -> 4*d_model -> d_model` 구조인가? |  |
| Residual connection은 왜 필요한가? |  |
| TransformerBlock 안에는 어떤 순서로 층이 들어가는가? |  |
| lm_head는 무엇을 출력하는가? |  |
| logits shape는 왜 `(B, T, vocab_size)`인가? |  |
| cross entropy를 계산할 때 왜 logits를 reshape하는가? |  |
| generate에서 왜 `logits[:, -1, :]`만 사용하는가? |  |

---

## 17. 최종 요약

4장은 GPT 모델을 실제 코드 구조로 조립하는 장이다.

2장의 token embedding과 position embedding이 입력을 만들고, 3장의 causal multi-head attention이 문맥을 섞는다.

4장에서는 attention을 `TransformerBlock` 안에 넣고, `LayerNorm`, `FeedForward`, `Residual connection`으로 안정적인 깊은 모델을 만든다.

최종적으로 `GPTModel`은 `(B, T)` token IDs를 받아 `(B, T, vocab_size)` logits를 출력한다.

학습할 때는 이 logits와 target token IDs로 cross entropy loss를 계산하고, 생성할 때는 마지막 token 위치의 logits에서 다음 token을 고른다.

