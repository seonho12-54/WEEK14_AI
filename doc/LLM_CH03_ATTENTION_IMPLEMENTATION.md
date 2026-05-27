# 3장 어텐션 메커니즘 구현하기

## 0. 이 장의 핵심 한 줄

**3장은 2장에서 만든 input embedding을 받아, 각 token이 앞 token들을 얼마나 참고할지 계산하는 attention을 구현하는 장이다.**

우리 과제에서는 이 장이 거의 그대로 `src/attention.py` 구현으로 이어진다.

```text
input_embeddings
-> Q, K, V projection
-> attention score
-> causal mask
-> softmax
-> dropout
-> weighted sum
-> multi-head
-> output projection
```

---

## 1. 3장에서 우리 구현에 필요한 부분만 고르기

교재 3장은 attention을 이해시키기 위해 단순한 attention부터 단계적으로 설명한다.

하지만 우리 과제 구현에 직접 필요한 것은 최종 형태인 **causal multi-head self-attention**이다.

| 교재 내용 | 구현 필요도 | 이유 |
| --- | --- | --- |
| 긴 시퀀스 문제 | 이해 필요 | 왜 attention이 필요한지 설명 |
| 간단한 self-attention | 이해용 | 원리 파악용, 최종 구현은 아님 |
| Q/K/V가 있는 self-attention | 매우 중요 | 실제 구현의 기본 |
| scaled dot-product attention | 매우 중요 | attention score 계산 방식 |
| causal mask | 매우 중요 | GPT가 미래 token을 못 보게 함 |
| attention dropout | 중요 | 과적합 방지 |
| multi-head attention | 매우 중요 | 과제의 최종 구현 형태 |

우리 과제 파일:

```text
src/attention.py
tests/test_attention.py
```

---

## 2. Attention이 필요한 이유

2장에서 만든 최종 입력은 이런 텐서였다.

```text
input_embeddings
shape = (batch_size, context_length, emb_dim)
```

각 token은 자기 embedding만 가지고 있으면 문맥을 알기 어렵다.

예:

```text
"나는 영화가 재미있어서 다시 봤다"
```

`봤다`라는 token을 이해하려면 앞의 `나는`, `영화`, `재미있어서`를 참고해야 한다.

Attention은 각 token이 다른 token들을 얼마나 참고할지 계산한다.

### 꼭 암기

**Attention은 각 token이 문장 안의 다른 token들을 얼마나 참고할지 계산해, 문맥이 반영된 새로운 vector를 만드는 구조다.**

---

## 3. Self-Attention이란 무엇인가

Self-attention은 같은 문장 안의 token들끼리 서로를 참고하는 attention이다.

입력과 참고 대상이 모두 같은 sequence 안에 있으므로 self-attention이라고 부른다.

```text
입력 token들:
[x1, x2, x3, x4]

x3을 새롭게 표현할 때:
x1, x2, x3, x4를 얼마나 참고할지 계산
```

GPT에서는 여기에 제한이 추가된다.

GPT는 미래 token을 보면 안 된다.

그래서 실제로는 다음처럼 본다.

```text
x1 -> x1만 봄
x2 -> x1, x2만 봄
x3 -> x1, x2, x3만 봄
x4 -> x1, x2, x3, x4만 봄
```

이것이 causal self-attention이다.

---

## 4. Context Vector란 무엇인가

Attention의 출력은 각 token마다 새롭게 만들어진 vector다.

이 vector를 문맥 벡터, 즉 context vector라고 생각하면 된다.

```text
원래 token vector
-> attention으로 주변 token 정보를 섞음
-> context vector
```

예:

```text
"좋았다" token embedding
-> 앞의 "영화", "정말"을 참고
-> 문맥이 반영된 "좋았다" vector
```

### 꼭 암기

**Context vector는 attention이 다른 token들의 정보를 섞어서 만든, 문맥이 반영된 token 표현이다.**

---

## 5. Q, K, V는 무엇인가

Attention은 입력 embedding을 바로 쓰지 않고, 세 가지 vector로 바꾼다.

```text
Q = Query
K = Key
V = Value
```

이 세 가지는 Linear layer로 만든다.

```text
Q = x @ Wq
K = x @ Wk
V = x @ Wv
```

PyTorch에서는 보통 이렇게 구현된다.

```python
self.q_proj = nn.Linear(d_model, d_model)
self.k_proj = nn.Linear(d_model, d_model)
self.v_proj = nn.Linear(d_model, d_model)
```

의미는 이렇게 잡으면 된다.

| 이름 | 의미 | 역할 |
| --- | --- | --- |
| Query | 내가 찾고 싶은 정보 | 현재 token이 무엇을 참고할지 묻는 벡터 |
| Key | 각 token의 이름표 | Query와 비교되어 관련도를 계산 |
| Value | 실제 가져올 내용 | attention weight로 섞이는 정보 |

### 꼭 암기

**Q와 K는 "얼마나 관련 있는지"를 계산하는 데 쓰이고, V는 실제로 섞어서 출력으로 만드는 정보다.**

---

## 6. Attention Score 계산

Q와 K를 만들었다면, 각 token이 다른 token과 얼마나 관련 있는지 점수를 계산한다.

공식:

```text
attention_scores = Q @ K^T
```

shape 흐름:

```text
Q: (B, T, C)
K: (B, T, C)
K^T: (B, C, T)

Q @ K^T -> (B, T, T)
```

여기서 `T x T`는 모든 token 쌍의 관계 점수다.

예:

```text
score[2, 0] = 3번째 token이 1번째 token을 얼마나 참고할지
score[2, 1] = 3번째 token이 2번째 token을 얼마나 참고할지
score[2, 2] = 3번째 token이 자기 자신을 얼마나 참고할지
```

---

## 7. 왜 sqrt(head_dim)으로 나누는가

실제 scaled dot-product attention은 그냥 `Q @ K^T`가 아니다.

```text
attention_scores = Q @ K^T / sqrt(head_dim)
```

이렇게 나누는 이유는 score 값이 너무 커지는 것을 막기 위해서다.

score가 너무 커지면 softmax 결과가 한쪽으로 너무 몰린다.

그러면 gradient가 작아져 학습이 느려지거나 불안정해질 수 있다.

### 꼭 암기

**`sqrt(head_dim)`으로 나누는 이유는 softmax가 너무 뾰족해지는 것을 막아 학습을 안정화하기 위해서다.**

---

## 8. Causal Mask가 필요한 이유

GPT는 다음 token을 예측하는 모델이다.

따라서 현재 token이 미래 token을 보면 안 된다.

예:

```text
input:  이 영화는 정말
target: 영화는 정말 좋았다
```

`정말` 위치에서 `좋았다`를 미리 볼 수 있으면 정답을 훔쳐보는 것이 된다.

그래서 attention score에서 미래 위치를 가린다.

```text
허용:
x3 -> x1, x2, x3

금지:
x3 -> x4, x5
```

이것이 causal mask다.

### 꼭 암기

**Causal mask는 GPT가 미래 token을 보지 못하게 막는 장치다.**

---

## 9. Causal Mask는 어떻게 구현하는가

attention score의 shape는 보통 다음과 같다.

```text
(B, n_heads, T, T)
```

여기서 마지막 `T x T` 행렬에서 주대각선 위쪽이 미래 token 위치다.

마스크 예:

```text
T = 4

허용 위치:
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

구현에서는 보통 미래 위치를 `-inf`로 채운 뒤 softmax를 적용한다.

```python
scores = scores.masked_fill(mask, float("-inf"))
attn_weights = torch.softmax(scores, dim=-1)
```

`-inf`는 softmax를 지나면 0이 된다.

즉 미래 token에는 attention weight가 0이 된다.

---

## 10. Softmax의 역할

attention score는 아직 그냥 점수다.

이 점수를 참고 비율로 바꾸기 위해 softmax를 사용한다.

```text
scores:        [2.0, 1.0, -inf]
softmax 후:    [0.73, 0.27, 0.00]
```

각 행의 합은 1이 된다.

즉, 각 token이 앞 token들을 어떤 비율로 참고할지 정해진다.

### 꼭 암기

**Softmax는 attention score를 합이 1인 attention weight로 바꾼다.**

---

## 11. Weighted Sum으로 출력 만들기

attention weight를 구했다면, 이제 V를 섞는다.

공식:

```text
context = attention_weights @ V
```

의미:

```text
중요한 token의 V는 많이 가져오고,
덜 중요한 token의 V는 적게 가져온다.
```

shape:

```text
attention_weights: (B, T, T)
V:                 (B, T, C)
context:           (B, T, C)
```

multi-head에서는 head 차원이 추가된다.

```text
attention_weights: (B, H, T, T)
V:                 (B, H, T, head_dim)
context:           (B, H, T, head_dim)
```

---

## 12. Attention Dropout

Dropout은 과적합을 줄이기 위한 기법이다.

Attention에서는 보통 softmax로 만든 attention weight에 dropout을 적용한다.

```text
attention score
-> causal mask
-> softmax
-> dropout
-> V와 곱하기
```

의미:

```text
학습 중 일부 attention 연결을 랜덤하게 끊는다.
```

추론할 때는 dropout을 끈다.

### 꼭 암기

**Attention dropout은 학습 중 일부 attention weight를 랜덤하게 0으로 만들어 특정 연결에 과하게 의존하는 것을 막는다.**

---

## 13. Multi-Head Attention이란 무엇인가

하나의 attention만 쓰면 한 가지 관점으로만 문맥을 본다.

Multi-head attention은 여러 개의 attention head를 사용해 여러 관점으로 문맥을 본다.

예:

```text
head 1: 문법 관계에 집중
head 2: 감정 표현에 집중
head 3: 앞뒤 반복 표현에 집중
```

실제 구현에서는 embedding 차원을 head 수만큼 나눈다.

```text
d_model = 128
n_heads = 4
head_dim = 32
```

각 head는 `32`차원씩 맡는다.

### 꼭 암기

**Multi-head attention은 하나의 embedding 공간을 여러 head로 나누어, 서로 다른 관점의 attention을 동시에 계산하는 구조다.**

---

## 14. Multi-Head Attention Shape 흐름

가장 중요한 구현 흐름이다.

입력:

```text
x: (B, T, C)
```

Q, K, V projection:

```text
q = q_proj(x) -> (B, T, C)
k = k_proj(x) -> (B, T, C)
v = v_proj(x) -> (B, T, C)
```

head로 나누기:

```text
C = n_heads * head_dim

(B, T, C)
-> (B, T, n_heads, head_dim)
-> (B, n_heads, T, head_dim)
```

attention score:

```text
q @ k.transpose(-2, -1)
-> (B, n_heads, T, T)
```

causal mask:

```text
(B, n_heads, T, T)에서 미래 위치를 -inf로 가림
```

softmax:

```text
attention weights
-> (B, n_heads, T, T)
```

weighted sum:

```text
attn_weights @ v
-> (B, n_heads, T, head_dim)
```

head 합치기:

```text
(B, n_heads, T, head_dim)
-> (B, T, n_heads, head_dim)
-> (B, T, C)
```

output projection:

```text
out_proj(context)
-> (B, T, C)
```

---

## 15. 구현 순서로 다시 보기

`MultiHeadAttention.forward(x)`는 보통 아래 순서로 구현한다.

```text
1. B, T, C = x.shape
2. q, k, v = Linear(x)
3. q, k, v를 head 개수로 나누기
4. attention score = q @ k.transpose(-2, -1)
5. score를 sqrt(head_dim)으로 나누기
6. causal mask 적용
7. softmax
8. attention dropout
9. attention weight @ v
10. head 다시 합치기
11. output projection
12. residual dropout 또는 output dropout
```

우리 과제 테스트에서는 특히 아래를 확인할 가능성이 높다.

```text
출력 shape가 입력과 같은가?
causal mask가 미래 token을 막는가?
attention weight shape가 맞는가?
d_model이 n_heads로 나누어지는가?
```

---

## 16. 구현할 때 헷갈리는 용어 정리

| 용어 | 뜻 | 구현 위치 |
| --- | --- | --- |
| projection | Linear layer로 벡터를 다른 표현으로 바꾸는 것 | q/k/v projection, output projection |
| Q | 현재 token이 찾는 정보 | q_proj |
| K | 각 token의 비교용 이름표 | k_proj |
| V | 실제로 가져올 내용 | v_proj |
| attention score | Q와 K의 유사도 점수 | `q @ k.T` |
| attention weight | softmax를 통과한 참고 비율 | `softmax(scores)` |
| context vector | attention weight로 V를 섞은 결과 | `attn @ v` |
| causal mask | 미래 token을 못 보게 하는 마스크 | `masked_fill(..., -inf)` |
| head | attention을 나누어 계산하는 관점 하나 | `n_heads` |
| head_dim | 각 head가 맡는 차원 | `d_model // n_heads` |

---

## 17. 2장과 3장의 연결

2장에서 만든 것은 attention의 입력이다.

```text
2장:
문장 -> token ID -> token embedding + position embedding

결과:
x = input_embeddings
shape = (B, T, C)
```

3장은 이 `x`를 받아 문맥 정보를 섞는다.

```text
3장:
x -> causal multi-head self-attention -> context-rich x
```

즉 3장의 attention은 token embedding에 문맥을 입히는 역할을 한다.

---

## 18. 4장과의 연결

3장에서 attention만 따로 구현한다.

4장에서는 이 attention을 TransformerBlock 안에 넣는다.

```text
TransformerBlock
-> LayerNorm
-> MultiHeadAttention
-> Residual connection
-> LayerNorm
-> FeedForward
-> Residual connection
```

그러므로 3장에서 만든 `MultiHeadAttention`은 4장 GPT 모델의 핵심 부품이 된다.

---

## 19. 꼭 암기할 문장

### 1

**Attention은 각 token이 다른 token을 얼마나 참고할지 계산하는 구조다.**

### 2

**Q와 K는 관련도 점수를 계산하는 데 쓰이고, V는 실제로 섞이는 정보다.**

### 3

**Attention score는 `Q @ K^T / sqrt(head_dim)`으로 계산한다.**

### 4

**Causal mask는 GPT가 미래 token을 보지 못하게 막는다.**

### 5

**Softmax는 attention score를 attention weight로 바꾼다.**

### 6

**Context vector는 attention weight로 V를 가중합한 결과다.**

### 7

**Multi-head attention은 여러 관점으로 attention을 동시에 계산하는 구조다.**

---

## 20. 3장 체크리스트

아래 질문에 답할 수 있으면 3장 구현 준비가 된 것이다.

| 질문 | 답할 수 있나 |
| --- | --- |
| attention이 왜 필요한가? |  |
| self-attention에서 self는 무슨 뜻인가? |  |
| Q, K, V는 각각 무엇인가? |  |
| attention score는 어떻게 계산하는가? |  |
| 왜 `sqrt(head_dim)`으로 나누는가? |  |
| causal mask는 왜 필요한가? |  |
| softmax는 attention에서 무슨 역할을 하는가? |  |
| attention weight와 weight parameter는 무엇이 다른가? |  |
| multi-head attention은 왜 head를 여러 개로 나누는가? |  |
| 최종 출력 shape가 왜 `(B, T, C)`로 돌아와야 하는가? |  |

---

## 21. 최종 요약

3장은 GPT의 핵심 부품인 attention을 구현하는 장이다.

우리 구현에서 중요한 흐름은 다음 하나다.

```text
x: (B, T, C)
-> Q, K, V
-> Q @ K^T / sqrt(head_dim)
-> causal mask
-> softmax
-> dropout
-> attention weights @ V
-> heads concat
-> output projection
-> out: (B, T, C)
```

이 구조가 완성되면 4장에서 TransformerBlock과 GPTModel을 만들 수 있다.

