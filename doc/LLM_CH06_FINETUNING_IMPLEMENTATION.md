# 6장 감성 분류 미세튜닝 구현 흐름 정리

## 0. 6장의 핵심 한 줄

**6장은 사전 학습한 GPT backbone 위에 분류용 head를 붙여, 영화 리뷰가 긍정인지 부정인지 맞히도록 추가 학습하는 장이다.**

사전 학습 단계의 목표:

```text
현재 token들을 보고 다음 token 맞히기
```

미세튜닝 단계의 목표:

```text
리뷰 문장 전체를 보고 긍정/부정 label 맞히기
```

즉, 6장은 GPT를 **언어 생성 모델**에서 **감성 분류 모델**로 사용하는 방법을 다룬다.

---

## 1. 우리 과제에서 6장과 연결되는 파일

구현 파일은 `src/finetune.py`이다.

| 함수/클래스 | 역할 | 중요도 |
| --- | --- | --- |
| `make_sentiment_dataset` | NSMC TSV를 train/val/test 데이터로 변환 | 중요 |
| `ReviewSentimentDataset` | 리뷰 문장을 token id로 바꾸고 padding | 매우 중요 |
| `GPTForSequenceClassification` | GPT 위에 감성 분류 head 추가 | 매우 중요 |
| `train_epoch_sentiment` | 감성 분류 모델 1 epoch 학습 | 중요 |
| `evaluate_sentiment` | 감성 분류 모델 평가 | 중요 |

---

## 2. 사전 학습과 미세튜닝의 차이

### 사전 학습

```text
입력: token id sequence
목표: 각 위치에서 다음 token 맞히기
출력: vocab_size개의 token 점수
loss: token-level cross entropy
```

예:

```text
input  = [이, 영화, 정말]
target = [영화, 정말, 좋다]
```

### 미세튜닝

```text
입력: 리뷰 문장 token id sequence
목표: 리뷰 전체의 label 맞히기
출력: 긍정/부정 2개 점수
loss: sentence-level cross entropy
```

예:

```text
input = "이 영화는 정말 좋았다"
label = 1
```

정리:

```text
사전 학습은 다음 token 예측
미세튜닝은 문장 전체 분류
```

---

## 3. 사용하는 데이터

`download_data.py`를 실행하면 아래 파일들이 생긴다.

```text
data/nsmc_sentiment_train.jsonl
data/nsmc_sentiment_val.jsonl
data/nsmc_sentiment_test.jsonl
```

각 줄은 이런 형태다.

```json
{"text": "이 영화는 정말 좋았다.", "label": 1}
{"text": "전개가 지루하고 별로였다.", "label": 0}
```

label 의미:

```text
0: 부정
1: 긍정
```

---

## 4. make_sentiment_dataset

이 함수는 NSMC 원본 TSV 파일을 읽어서 감성 분류용 데이터로 바꾼다.

입력 파일 형식:

```text
id    document    label
1     정말 좋았다  1
2     별로였다     0
```

해야 할 일:

```text
1. TSV 파일을 읽는다.
2. document가 비어 있는 row는 버린다.
3. label이 0 또는 1인 row만 사용한다.
4. {"text": 문장, "label": 정수} 형태로 만든다.
5. train 데이터를 train/validation으로 나눈다.
6. test 파일이 있으면 test_data도 만든다.
```

반환 형태:

```python
train_data, val_data, test_data
```

각 데이터는 list of dict이다.

```python
[
    {"text": "정말 좋았다", "label": 1},
    {"text": "별로였다", "label": 0},
]
```

---

## 5. ReviewSentimentDataset

이 Dataset은 리뷰 하나와 label 하나를 반환한다.

흐름:

```text
리뷰 문자열
-> tokenizer.encode()
-> token id list
-> max_length에 맞게 자르기 또는 padding
-> torch.LongTensor로 변환
-> label과 함께 반환
```

예:

```text
text = "정말 좋았다"
label = 1
max_length = 8
```

토크나이저 결과가:

```text
[15, 22, 91, 44, 3]
```

이면 padding 후:

```text
[15, 22, 91, 44, 3, 0, 0, 0]
```

반환:

```python
input_ids: torch.Tensor shape (max_length,)
label: int
```

중요:

```text
짧으면 pad_id로 채운다.
길면 max_length까지만 자른다.
input_ids dtype은 torch.long이어야 한다.
```

---

## 6. 왜 padding이 필요한가

DataLoader는 여러 리뷰를 batch로 묶는다.

하지만 리뷰마다 token 길이가 다르다.

```text
리뷰 A: 10 token
리뷰 B: 37 token
리뷰 C: 5 token
```

batch로 묶으려면 길이가 같아야 한다.

그래서 짧은 문장 뒤에 `<pad>` token을 붙인다.

```text
리뷰 A: [10개 token + pad...]
리뷰 B: [37개 token]
리뷰 C: [5개 token + pad...]
```

우리 과제에서는 고정 길이 `max_length`로 맞춘다.

---

## 7. GPTForSequenceClassification

이 클래스는 GPT backbone 위에 분류용 classifier를 붙인다.

구조:

```text
input_ids
-> GPT embedding
-> TransformerBlock들
-> final LayerNorm
-> 문장 대표 hidden state 선택
-> dropout
-> classifier
-> logits shape (B, num_labels)
```

중요한 차이:

```text
GPTModel의 lm_head:
각 위치마다 vocab_size개 점수 출력

분류 classifier:
문장 하나마다 num_labels개 점수 출력
```

감성 분류에서는 `vocab_size`가 필요 없다.

필요한 것은:

```text
부정 점수
긍정 점수
```

즉 `num_labels=2`이다.

---

## 8. LM head와 classification head 차이

### LM head

사전 학습용:

```text
hidden state
-> lm_head
-> vocab_size개 점수
-> 다음 token 예측
```

shape:

```text
(B, T, emb_dim)
-> (B, T, vocab_size)
```

### Classification head

감성 분류용:

```text
문장 대표 hidden state
-> classifier
-> 2개 점수
-> 긍정/부정 예측
```

shape:

```text
(B, emb_dim)
-> (B, 2)
```

정리:

```text
lm_head는 다음 token을 맞히기 위한 head
classifier는 문장 label을 맞히기 위한 head
```

---

## 9. 문장 대표 hidden state는 무엇을 쓰나

GPT backbone은 각 token 위치마다 hidden state를 만든다.

```text
hidden states shape = (B, T, emb_dim)
```

분류는 문장 전체에 대한 판단이 필요하다.

그래서 보통 하나의 token 위치를 대표 벡터로 사용한다.

간단한 구현에서는 마지막 token의 hidden state를 쓴다.

```python
last_hidden = hidden_states[:, -1, :]
```

shape:

```text
(B, T, emb_dim)
-> (B, emb_dim)
```

주의:

```text
padding이 있는 경우 진짜 마지막 token이 pad일 수 있다.
더 정확한 구현은 padding이 아닌 마지막 token 위치를 찾아야 한다.
```

하지만 테스트는 보통 shape 중심이라 단순 마지막 위치도 통과할 수 있다.

---

## 10. GPTModel을 그대로 쓰면 안 되는 이유

`GPTModel.forward()`는 보통 logits를 반환한다.

```text
(B, T, vocab_size)
```

하지만 감성 분류에 필요한 것은:

```text
(B, 2)
```

그래서 `GPTForSequenceClassification`에서는 GPTModel의 전체 forward를 그대로 쓰기보다, GPT 내부 부품을 이용해 hidden state를 얻고 classifier에 넣는 방식이 필요하다.

흐름:

```python
x = self.gpt.embedding(input_ids)
x = self.gpt.blocks(x)
x = self.gpt.final_norm(x)
sentence_vec = x[:, -1, :]
logits = self.classifier(sentence_vec)
```

---

## 11. classifier loss

감성 분류에서도 loss는 cross entropy를 쓴다.

```python
loss = F.cross_entropy(logits, labels)
```

shape:

```text
logits: (B, 2)
labels: (B,)
```

예:

```text
logits[0] = [부정 점수, 긍정 점수]
label[0] = 1
```

정답 label의 점수가 높아지도록 학습한다.

---

## 12. train_epoch_sentiment

이 함수는 감성 분류 모델을 1 epoch 학습한다.

흐름:

```text
model.train()

for input_ids, labels in train_loader:
    input_ids, labels를 device로 이동
    optimizer.zero_grad()
    loss, logits = model(input_ids, labels)
    loss.backward()
    optimizer.step()
    accuracy 계산

평균 loss, accuracy 반환
```

accuracy 계산:

```python
preds = torch.argmax(logits, dim=-1)
correct = (preds == labels).sum().item()
```

의미:

```text
logits에서 더 큰 점수를 가진 class를 예측 label로 본다.
```

---

## 13. evaluate_sentiment

이 함수는 감성 분류 모델을 평가한다.

흐름:

```text
model.eval()

with torch.no_grad():
    for input_ids, labels in data_loader:
        loss, logits = model(input_ids, labels)
        accuracy 계산

평균 loss, accuracy 반환
```

학습과 차이:

```text
optimizer.zero_grad() 없음
loss.backward() 없음
optimizer.step() 없음
```

평가에서는 파라미터를 바꾸지 않는다.

---

## 14. freeze란 무엇인가

미세튜닝에서는 선택적으로 GPT backbone 일부를 얼릴 수 있다.

freeze:

```text
기존 GPT 파라미터는 업데이트하지 않고,
새로 붙인 classifier만 학습하는 것
```

예:

```python
for param in gpt_model.parameters():
    param.requires_grad = False
```

장점:

```text
학습이 빠르다.
작은 데이터에서 과적합을 줄일 수 있다.
```

단점:

```text
backbone이 감성 분류 task에 충분히 적응하지 못할 수 있다.
```

우리 과제 필수 구현에서는 freeze가 필수는 아니다.

---

## 15. 과제 구현 순서 추천

`src/finetune.py`는 아래 순서로 구현하는 것이 좋다.

```text
1. make_sentiment_dataset
2. ReviewSentimentDataset.__getitem__
3. GPTForSequenceClassification.__init__
4. GPTForSequenceClassification.forward
5. train_epoch_sentiment
6. evaluate_sentiment
```

이유:

```text
데이터를 만들 수 있어야 Dataset을 만들 수 있다.
Dataset이 있어야 DataLoader로 batch를 만들 수 있다.
모델 forward가 되어야 train/eval 루프를 만들 수 있다.
```

---

## 16. 테스트 기준으로 꼭 맞춰야 할 부분

`tests/test_finetune.py` 기준으로 중요한 점:

```text
make_sentiment_dataset:
빈 리뷰를 제외하고 {"text", "label"} 형태를 반환해야 한다.

ReviewSentimentDataset:
input_ids shape가 (max_length,)여야 한다.
input_ids dtype은 torch.long이어야 한다.

GPTForSequenceClassification:
입력 shape (B, T)에 대해 logits shape (B, 2)를 반환해야 한다.

train_epoch_sentiment / evaluate_sentiment:
callable이어야 하며, 실제 구현에서는 평균 loss와 accuracy를 반환한다.
```

---

## 17. 사전 학습 모델과 미세튜닝 모델의 관계

미세튜닝 모델은 GPT를 새로 만드는 것이 아니다.

```text
GPTModel
-> GPTForSequenceClassification 안에 backbone으로 들어감
-> classifier head 추가
```

즉:

```python
backbone = GPTModel(config)
model = GPTForSequenceClassification(backbone, num_labels=2)
```

이 구조에서 `backbone`은 문장을 벡터로 이해하는 역할을 하고, `classifier`는 그 벡터를 긍정/부정으로 바꾸는 역할을 한다.

---

## 18. 최종 요약

6장은 GPT를 감성 분류 문제에 맞게 바꾸는 장이다.

핵심 흐름:

```text
리뷰 text
-> tokenizer.encode()
-> padding/truncation
-> GPT backbone
-> 마지막 hidden state
-> classifier
-> 긍정/부정 logits
-> cross entropy loss
```

가장 중요한 차이:

```text
사전 학습:
모든 token 위치에서 다음 token 예측

미세튜닝:
문장 하나에 대해 label 하나 예측
```

구현 관점에서 기억할 것:

```text
LM head는 vocab_size 출력
classification head는 num_labels 출력
```

즉, 미세튜닝은 GPT의 언어 이해 능력을 backbone으로 사용하고, 마지막에 task 전용 head를 붙여 새로운 문제를 풀게 만드는 과정이다.
