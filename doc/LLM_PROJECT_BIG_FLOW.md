# mini GPT LLM 구현 큰 흐름

## 0. 한 줄 요약

이번 LLM 과제는 문장을 숫자로 바꾸는 tokenizer부터 시작해서, GPT 구조를 직접 만들고, 다음 token 예측으로 사전 학습한 뒤, 영화 리뷰 감성 분류로 미세 조정하는 프로젝트다.

```text
텍스트
-> 토큰 ID
-> 임베딩
-> 어텐션
-> 트랜스포머 블록
-> 다음 토큰 예측
-> 사전 학습
-> 감성 분류 미세 조정
```

---

## 1. 왜 토크나이저부터 시작하나

신경망은 글자를 그대로 읽지 못한다. 모델이 계산할 수 있는 것은 숫자이므로, 문장을 먼저 token ID라는 숫자 목록으로 바꿔야 한다.

MNIST에서는 이미지가 이미 `784`개의 숫자였다.

LLM에서는 문장을 직접 숫자로 바꾸는 과정이 필요하다.

```text
MNIST: 이미지 -> 784개 숫자
LLM: 문장 -> token ID 목록
```

이번 과제에서는 외부 tokenizer를 쓰지 않고, byte-level BPE tokenizer를 직접 구현한다.

과제 파일:

```text
src/bpe.py
```

핵심 흐름:

```text
문장
-> UTF-8 byte
-> byte token
-> 자주 붙는 token pair merge
-> BPE token ID
```

---

## 2. Dataset은 다음 토큰 예측 문제를 만든다

GPT는 문장을 보고 다음 token을 맞히도록 학습한다.

예를 들어 token ID가 다음과 같다고 하자.

```text
[10, 11, 12, 13]
```

context length가 3이면 학습 샘플은 이렇게 만들어진다.

```text
input  = [10, 11, 12]
target = [11, 12, 13]
```

즉, 모델은 현재까지의 token들을 보고 바로 다음 token을 맞히는 훈련을 한다.

과제 파일:

```text
src/dataset.py
```

핵심 개념:

```text
input과 target은 길이가 같다.
target은 input보다 한 칸 뒤에 있다.
이 구조가 GPT 사전 학습의 기본 문제다.
```

---

## 3. Embedding은 token ID를 벡터로 바꾼다

token ID는 단순한 숫자일 뿐이다.

예를 들어 `153`이라는 숫자가 있다고 해서, 모델이 그 숫자만 보고 의미를 이해할 수는 없다.

그래서 각 token ID를 학습 가능한 벡터로 바꾼다. 이것이 token embedding이다.

또한 GPT는 token의 순서도 알아야 하므로 position embedding을 더한다.

과제 파일:

```text
src/embeddings.py
```

흐름:

```text
token IDs
-> token embedding
-> position embedding 더하기
-> transformer가 읽을 수 있는 벡터 시퀀스
```

출력 shape:

```text
(batch_size, seq_len, emb_dim)
```

---

## 4. Attention은 각 token이 앞 token들을 참고하게 만든다

GPT의 핵심은 self-attention이다.

각 token은 혼자 계산되는 것이 아니라, 문장 안의 다른 token들을 참고해서 자신의 의미를 다시 만든다.

GPT는 미래 token을 보면 안 되므로 causal attention을 사용한다.

과제 파일:

```text
src/attention.py
```

큰 흐름:

```text
입력 x
-> Q, K, V 생성
-> Q와 K로 attention score 계산
-> causal mask로 미래 token 가리기
-> softmax로 참고 비율 계산
-> 참고 비율로 V를 가중합
-> output projection
```

Q, K, V는 이렇게 보면 된다.

| 이름 | 의미 |
| --- | --- |
| Q | 내가 찾고 싶은 정보 |
| K | 각 token이 가진 이름표 |
| V | 실제로 가져올 내용 |

---

## 5. TransformerBlock은 GPT의 반복 단위다

GPT는 attention 하나만으로 끝나지 않는다.

attention으로 문맥을 섞고, feed forward network로 각 token의 표현을 한 번 더 가공한다.

이때 학습을 안정화하기 위해 LayerNorm과 residual connection을 함께 사용한다.

과제 파일:

```text
src/model.py
```

한 블록의 흐름:

```text
x
-> LayerNorm
-> MultiHeadAttention
-> residual connection
-> LayerNorm
-> FeedForward
-> residual connection
```

여기서 중요한 부품:

| 부품 | 역할 |
| --- | --- |
| LayerNorm | 각 token 벡터의 분포를 안정화 |
| MultiHeadAttention | 여러 관점으로 문맥 참고 |
| FeedForward | 각 token 표현을 더 깊게 변환 |
| GELU | ReLU보다 부드러운 활성화 함수 |
| Residual connection | 깊은 모델에서 gradient가 잘 흐르게 도움 |

---

## 6. GPTModel은 TransformerBlock을 쌓은 모델이다

GPT 모델은 다음 구조로 이루어진다.

```text
token IDs
-> InputEmbedding
-> TransformerBlock x n_layers
-> LayerNorm
-> Linear lm_head
-> vocab logits
```

마지막 `lm_head`는 각 위치에서 다음 token이 무엇일지 vocab size만큼의 점수를 출력한다.

예를 들어 vocab size가 3000이면, 각 token 위치마다 3000개의 점수가 나온다.

```text
logits shape: (batch_size, seq_len, vocab_size)
```

정답 target과 비교해서 Cross Entropy Loss를 계산한다.

---

## 7. 사전 학습은 다음 token 맞히기를 반복하는 과정이다

사전 학습의 목표는 단순하다.

```text
앞 token들을 보고 다음 token을 맞힌다.
```

이 단순한 문제를 아주 많이 반복하면, 모델은 문장 구조와 단어 관계를 조금씩 배운다.

과제 파일:

```text
src/train.py
```

학습 루프:

```text
for epoch:
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        loss = model(input_batch, target_batch)
        loss.backward()
        optimizer.step()
```

MNIST와 비교하면 다음과 같다.

| MNIST | LLM |
| --- | --- |
| 이미지 보고 숫자 맞히기 | 앞 token 보고 다음 token 맞히기 |
| 출력 클래스 10개 | 출력 클래스 vocab_size개 |
| Cross Entropy Loss | Cross Entropy Loss |
| Adam/SGD로 업데이트 | AdamW/Adam으로 업데이트 |

---

## 8. 텍스트 생성은 모델 예측을 다시 입력에 붙이는 과정이다

학습된 GPT는 다음 token을 예측할 수 있다.

텍스트 생성은 이 예측을 반복하는 과정이다.

```text
시작 문장
-> 다음 token 예측
-> 예측 token을 뒤에 붙임
-> 다시 다음 token 예측
-> 반복
```

예:

```text
"이 영화는"
-> "정말"
-> "좋았다"
-> "!"
```

생성 전략:

| 방법 | 의미 |
| --- | --- |
| greedy | 가장 점수가 높은 token 선택 |
| temperature | 확률 분포를 날카롭거나 부드럽게 조절 |
| top-k | 상위 k개 후보 중에서만 선택 |

---

## 9. 미세 조정은 GPT를 특정 작업에 맞게 바꾸는 과정이다

사전 학습 GPT는 다음 token을 맞히는 모델이다.

하지만 이번 과제에서는 NSMC 영화 리뷰 감성 분류도 수행한다.

감성 분류는 리뷰가 긍정인지 부정인지 맞히는 문제다.

과제 파일:

```text
src/finetune.py
```

사전 학습과 감성 분류의 차이:

| 구분 | 사전 학습 | 감성 분류 미세 조정 |
| --- | --- | --- |
| 입력 | 문장 token |
| 목표 | 다음 token 예측 | 긍정/부정 예측 |
| 출력 크기 | vocab_size | 2 |
| head | lm_head | classification head |
| loss | token 단위 Cross Entropy | 문장 단위 Cross Entropy |

감성 분류 흐름:

```text
리뷰 문장
-> tokenizer
-> GPT backbone
-> 마지막 유효 token hidden state
-> Linear classifier
-> 긍정/부정 logits
```

---

## 10. 우리 과제 구현 순서

테스트 기준으로는 아래 순서대로 가면 된다.

| 순서 | 구현 대상 | 파일 | 테스트 |
| --- | --- | --- | --- |
| 1 | BPE tokenizer | `src/bpe.py` | `pytest tests/test_bpe.py -v` |
| 2 | Dataset / Embedding | `src/dataset.py`, `src/embeddings.py` | `pytest tests/test_dataset.py -v` |
| 3 | MultiHeadAttention | `src/attention.py` | `pytest tests/test_attention.py -v` |
| 4 | GPT 모델 구성 | `src/model.py` | `pytest tests/test_model.py -v` |
| 5 | 사전 학습 유틸리티 | `src/train.py` | `pytest tests/test_train.py -v` |
| 6 | 감성 분류 미세 조정 | `src/finetune.py` | `pytest tests/test_finetune.py -v` |
| 7 | 전체 확인 | 전체 | `pytest tests/ -v` |

---

## 11. 정말 큰 그림만 다시 보기

전체 흐름을 한 번 더 압축하면 다음과 같다.

```text
1. 텍스트를 token ID로 바꾼다.
2. token ID를 embedding vector로 바꾼다.
3. attention으로 앞 token들의 문맥을 참고한다.
4. transformer block을 여러 층 쌓아 표현을 깊게 만든다.
5. 다음 token을 맞히며 사전 학습한다.
6. 학습된 GPT backbone 위에 classifier를 붙여 감성 분류로 미세 조정한다.
```

---

## 12. MNIST에서 LLM으로 이어지는 감각

MNIST에서 배운 흐름은 그대로 LLM으로 이어진다.

| MNIST | LLM |
| --- | --- |
| 이미지 픽셀 입력 | token ID 입력 |
| Affine | Linear projection |
| ReLU | GELU |
| BatchNorm | LayerNorm |
| Softmax | Softmax |
| Cross Entropy | Cross Entropy |
| 숫자 클래스 예측 | 다음 token 예측 |
| train/evaluate | pretrain/generate/finetune |

결국 둘 다 핵심은 같다.

```text
입력 숫자화
-> forward
-> loss
-> backward
-> optimizer update
```

LLM은 여기에 tokenizer, embedding, attention, transformer block이 추가된 더 큰 신경망이다.

