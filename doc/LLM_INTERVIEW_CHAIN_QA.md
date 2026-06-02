# LLM 면접 대비 꼬리질문 시뮬레이션

## 0. 이 문서의 목적

이 문서는 단순 개념 요약이 아니라, 면접이나 발표에서 시니어 개발자가 꼬리물기식으로 질문한다고 가정하고 정리한 답변 연습 문서다.

목표는 아래 질문에 막히지 않는 것이다.

```text
LLM이 뭔가요?
직접 구현했다면 어떤 흐름으로 만들었나요?
왜 tokenizer가 필요한가요?
왜 BPE를 썼나요?
Transformer에서 attention은 무엇을 하나요?
GPT는 어떻게 다음 token을 예측하나요?
학습과 미세튜닝은 무엇이 다른가요?
```

---

## 1. LLM이 뭔가요?

### Q. LLM이 뭔가요?

**LLM은 Large Language Model의 약자이며, 대량의 텍스트를 학습해 다음 token을 예측하고 문장을 생성하는 대규모 언어 모델입니다.**

조금 더 풀어서 말하면:

```text
문장을 token id로 바꾼 뒤,
Transformer 구조를 이용해 문맥을 학습하고,
현재까지의 token들을 보고 다음 token의 확률을 예측하는 모델
```

이다.

---

### Q. LLM은 사람처럼 문장을 이해하나요?

정확히 말하면 사람처럼 의미를 이해한다기보다, **대량의 텍스트 패턴을 학습해 주어진 문맥 다음에 올 가능성이 높은 token을 예측한다**고 보는 것이 맞다.

예를 들어:

```text
입력: 이 영화는 정말
예측 후보: 좋았다, 재미있다, 별로였다, ...
```

모델은 vocabulary 전체 token에 대해 점수를 만들고, 그중 다음 token으로 적절한 것을 선택한다.

---

### Q. 그럼 LLM의 가장 기본 목표는 뭔가요?

가장 기본 목표는 **다음 token 예측**이다.

```text
현재 token sequence -> 다음 token 맞히기
```

예:

```text
input  = [나는, 오늘, 영화를]
target = [오늘, 영화를, 봤다]
```

각 위치마다 다음 token을 맞히도록 학습한다.

---

## 2. mini GPT를 처음부터 구현한다면 전체 흐름은?

### Q. 직접 구현한 mini GPT의 전체 흐름을 설명해보세요.

전체 흐름은 다음과 같다.

```text
텍스트 데이터 준비
-> BPE tokenizer 학습
-> 문장을 token id로 변환
-> Dataset/DataLoader로 input/target batch 생성
-> token embedding + position embedding
-> causal multi-head self-attention
-> TransformerBlock 반복
-> lm_head로 vocab logits 생성
-> cross entropy loss 계산
-> backward와 optimizer update
-> generate로 token 생성
-> 감성 분류 미세튜닝
```

우리 과제 파일로 연결하면:

```text
bpe.py        -> tokenizer
dataset.py    -> input/target window
embeddings.py -> token/position embedding
attention.py  -> causal multi-head attention
model.py      -> GPT model
train.py      -> pretraining/generation/checkpoint
finetune.py   -> sentiment classification fine-tuning
```

---

### Q. LLM 구현에서 제일 먼저 필요한 것은 뭔가요?

가장 먼저 필요한 것은 tokenizer다.

이유:

```text
LLM은 문자열을 직접 계산하지 못한다.
문장을 숫자 token id로 바꿔야 한다.
```

즉:

```text
"이 영화는 좋았다"
-> tokenizer
-> [2, 301, 42, 91, 3]
```

이렇게 바꿔야 모델에 넣을 수 있다.

---

## 3. Tokenizer와 BPE

### Q. Tokenizer는 무엇인가요?

Tokenizer는 **문자열을 모델이 처리할 수 있는 token id 목록으로 바꾸고, 다시 token id를 문자열로 복원하는 도구**다.

역할:

```text
encode: text -> token ids
decode: token ids -> text
```

우리 구현에서는 `BPETokenizer`가 이 역할을 한다.

---

### Q. BPE는 무엇인가요?

BPE는 Byte Pair Encoding의 약자이며, **자주 붙어 나오는 token pair를 반복적으로 합쳐 vocabulary를 만드는 tokenization 방식**이다.

우리 과제에서는 UTF-8 byte-level BPE를 구현했다.

흐름:

```text
1. 특수 token 등록: <pad>, <unk>, <bos>, <eos>
2. byte 0~255 등록
3. corpus를 UTF-8 byte로 변환
4. 가장 자주 등장하는 이웃 pair를 찾음
5. 새 token id로 merge
6. vocab_size에 도달할 때까지 반복
```

---

### Q. 왜 단어 단위 tokenizer를 쓰지 않고 BPE를 쓰나요?

BPE를 쓰는 이유:

```text
1. 처음 보는 단어도 byte/subword로 쪼개 처리할 수 있다.
2. 단어 단위보다 vocabulary 크기를 적당히 줄일 수 있다.
3. 자주 나오는 글자 조합은 하나의 token으로 묶어 효율적으로 표현할 수 있다.
```

특히 한국어는 조사와 어미가 붙어 단어 형태가 다양하다.

예:

```text
재미있다
재미있었다
재미있네요
재미있지만
```

단어 단위로 보면 전부 다른 단어가 되지만, BPE는 공통 부분을 더 잘 활용할 수 있다.

---

### Q. byte-level BPE에서 한글은 어떻게 처리되나요?

한글 한 글자는 UTF-8에서 보통 3 byte로 표현된다.

예:

```python
"한".encode("utf-8")
# b'\xed\x95\x9c'
```

byte-level BPE는 이 byte 목록에서 시작한다.

```text
"한글"
-> UTF-8 bytes
-> byte token ids
-> 자주 붙는 pair를 merge
```

장점은 모든 한글, 영어, 숫자, 문장부호, 이모티콘 등을 `<unk>` 없이 최소 byte 단위로 표현할 수 있다는 점이다.

---

### Q. tokenizer 학습과 LLM 학습은 같은 건가요?

다르다.

Tokenizer 학습:

```text
corpus에서 vocabulary와 merge rule을 만든다.
모델 파라미터를 학습하는 것은 아니다.
```

LLM 학습:

```text
token id 데이터를 모델에 넣고,
loss를 줄이도록 모델 파라미터를 업데이트한다.
```

즉 tokenizer 학습은 텍스트를 숫자로 바꾸기 위한 사전 준비이고, LLM 학습은 실제 neural network 파라미터를 학습하는 과정이다.

---

### Q. tokenizer에서 save/load가 왜 필요한가요?

BPE vocab과 merge rule을 매번 다시 학습하면 시간이 오래 걸린다.

그래서 한 번 학습한 tokenizer를 저장한다.

```text
save:
id_to_token, token_to_id, merges 저장

load:
저장된 vocab과 merge rule 복원
```

이후 모델 학습이나 추론에서는 `load()`해서 같은 tokenizer를 재사용한다.

중요:

```text
학습 때 사용한 tokenizer와 추론 때 사용하는 tokenizer가 같아야 한다.
```

---

## 4. Dataset과 DataLoader

### Q. tokenizer 다음 단계는 무엇인가요?

tokenizer로 전체 corpus를 token id 목록으로 바꾼 뒤, GPT 학습용 input/target pair를 만들어야 한다.

GPT는 다음 token을 맞히므로 input과 target은 한 칸 차이난다.

예:

```text
token_ids = [10, 11, 12, 13]
context_length = 3

input  = [10, 11, 12]
target = [11, 12, 13]
```

---

### Q. context_length는 무엇인가요?

`context_length`는 모델이 한 번에 볼 수 있는 token 개수다.

예:

```text
context_length = 128
```

이면 모델은 한 번 forward에서 최대 128개 token을 본다.

모델의 position embedding도 이 길이를 기준으로 만들어진다.

---

### Q. stride는 무엇인가요?

`stride`는 다음 학습 샘플 window를 만들 때 몇 칸 이동할지 정하는 값이다.

예:

```text
token_ids = [0, 1, 2, 3, 4, 5, 6]
context_length = 3
stride = 2
```

이면:

```text
sample 1 input = [0, 1, 2]
sample 2 input = [2, 3, 4]
sample 3 input = [4, 5, 6]
```

처럼 이동한다.

---

### Q. DataLoader는 왜 필요한가요?

Dataset은 샘플 하나를 꺼내는 규칙이고, DataLoader는 여러 샘플을 batch로 묶어준다.

```text
Dataset.__getitem__(idx)
-> input_ids, target_ids 하나 반환

DataLoader
-> 여러 샘플을 모아 batch 반환
```

shape:

```text
input_batch  : (batch_size, context_length)
target_batch : (batch_size, context_length)
```

---

## 5. Embedding

### Q. token id를 그대로 attention에 넣으면 안 되나요?

안 된다.

token id는 단순한 정수 번호일 뿐이다.

예:

```text
100번 token이 50번 token보다 두 배 중요하다는 뜻이 아니다.
```

그래서 token id를 학습 가능한 vector로 바꾼다.

이것이 token embedding이다.

---

### Q. token embedding은 무엇인가요?

token embedding은 **각 token id를 d_model 차원의 벡터로 바꾸는 학습 가능한 lookup table**이다.

예:

```text
input_ids shape: (B, T)
embedding output shape: (B, T, d_model)
```

예를 들어:

```text
(8, 16)
-> (8, 16, 64)
```

이면 8개 window, 각 window 16개 token, 각 token은 64차원 벡터라는 뜻이다.

---

### Q. position embedding은 왜 필요한가요?

Transformer attention 자체는 token 순서를 자동으로 알지 못한다.

그래서 각 token이 몇 번째 위치에 있는지 알려주는 position embedding을 더한다.

```text
input embedding = token embedding + position embedding
```

즉 같은 token이라도 위치가 다르면 다른 입력 표현이 된다.

---

## 6. Attention

### Q. Attention은 무엇인가요?

Attention은 **각 token이 문맥을 이해하기 위해 다른 token들을 얼마나 참고할지 계산하는 구조**다.

GPT에서는 causal self-attention을 사용한다.

```text
self-attention:
같은 문장 안의 token들끼리 서로 참고

causal:
현재 token이 미래 token은 보지 못하게 제한
```

---

### Q. Q, K, V는 무엇인가요?

입력 embedding을 서로 다른 Linear layer에 통과시켜 Q, K, V를 만든다.

```text
Q: Query, 내가 찾고 싶은 정보
K: Key, 내가 어떤 정보를 가지고 있는지
V: Value, 실제로 전달할 정보
```

계산:

```python
q = q_proj(x)
k = k_proj(x)
v = v_proj(x)
```

각 projection은 학습 가능한 Linear layer다.

---

### Q. Q @ K^T는 왜 하나요?

Q와 K의 내적을 통해 token 간 관련도를 계산한다.

```text
Q @ K^T -> attention score
```

shape:

```text
Q: (B, n_heads, T, head_dim)
K^T: (B, n_heads, head_dim, T)
score: (B, n_heads, T, T)
```

`T x T` 표는 각 token이 다른 token을 얼마나 참고할지 나타내는 점수표다.

---

### Q. 왜 sqrt(head_dim)으로 나누나요?

head_dim이 커질수록 Q와 K의 내적값이 커질 수 있다.

score가 너무 커지면 softmax가 한쪽으로 과하게 쏠린다.

그래서:

```text
score = QK^T / sqrt(head_dim)
```

으로 스케일을 조절한다.

---

### Q. causal mask는 왜 필요한가요?

GPT는 다음 token을 예측하는 모델이다.

학습 중 현재 위치가 미래 token을 볼 수 있으면 정답을 미리 보는 꼴이 된다.

그래서 미래 위치 score를 `-inf`로 만든다.

```text
현재 token은 자기 자신과 이전 token만 볼 수 있다.
미래 token은 볼 수 없다.
```

---

### Q. attention score와 attention weight는 무엇이 다른가요?

attention score:

```text
softmax 전의 원시 관련도 점수
```

attention weight:

```text
softmax 후의 참고 비율
```

흐름:

```text
score
-> causal mask
-> softmax
-> weight
-> weight @ V
-> context vector
```

---

### Q. Multi-head attention은 왜 여러 head로 나누나요?

하나의 attention만 쓰면 하나의 관점으로만 문맥을 본다.

여러 head를 쓰면 서로 다른 관점으로 token 관계를 볼 수 있다.

예:

```text
head 1: 가까운 조사/어미 관계
head 2: 주어-서술어 관계
head 3: 감정 단어 관계
```

각 head에서 context를 만든 뒤 다시 합쳐 `out_proj`를 통과시킨다.

---

## 7. TransformerBlock

### Q. TransformerBlock은 무엇인가요?

TransformerBlock은 attention과 feed-forward network를 residual connection과 layer normalization으로 묶은 GPT의 기본 반복 단위다.

구조:

```text
x
-> LayerNorm
-> MultiHeadAttention
-> residual add
-> LayerNorm
-> FeedForward
-> residual add
-> output
```

---

### Q. LayerNorm은 왜 쓰나요?

LayerNorm은 각 token vector의 마지막 차원, 즉 feature 차원을 정규화한다.

shape:

```text
x: (B, T, C)
LayerNorm은 C 차원 기준으로 평균/분산 계산
```

목적:

```text
값의 스케일을 안정화해 gradient가 잘 흐르게 돕는다.
```

BatchNorm과 차이:

```text
BatchNorm:
batch 방향 통계를 사용

LayerNorm:
각 token vector 내부 feature 통계를 사용
```

Transformer에서는 sequence 길이와 batch 크기가 달라질 수 있으므로 LayerNorm이 잘 맞는다.

---

### Q. Residual connection은 왜 필요한가요?

Residual connection은 원래 입력 x를 변환 결과에 더하는 구조다.

```text
x + layer(x)
```

이유:

```text
깊은 모델에서 gradient가 잘 흐르게 한다.
원래 정보를 잃지 않고 유지할 수 있다.
```

TransformerBlock에서는:

```python
x = x + attention(norm(x))
x = x + feedforward(norm(x))
```

처럼 사용한다.

---

### Q. FeedForward Network는 무엇을 하나요?

Attention이 token 간 관계를 섞는다면, FeedForward는 각 token vector 내부 표현을 더 풍부하게 가공한다.

구조:

```text
d_model
-> 4 * d_model
-> GELU
-> d_model
```

즉 차원을 잠깐 크게 늘려 더 다양한 특징 조합을 만든 뒤 다시 원래 차원으로 줄인다.

---

### Q. GELU는 왜 쓰나요?

GELU는 ReLU보다 부드러운 activation이다.

Transformer/GPT 계열에서는 ReLU보다 GELU가 자주 쓰인다.

이유:

```text
음수 영역을 무조건 0으로 자르지 않고 부드럽게 조절한다.
큰 차원의 연속적인 표현을 다루는 Transformer에 잘 맞는다.
```

---

## 8. GPTModel

### Q. GPTModel의 전체 구조는?

우리 구현의 GPTModel 구조:

```text
token ids
-> InputEmbedding
-> TransformerBlock x n_layers
-> final LayerNorm
-> lm_head
-> logits
```

shape:

```text
idx: (B, T)
x: (B, T, emb_dim)
logits: (B, T, vocab_size)
```

---

### Q. lm_head는 무엇인가요?

`lm_head`는 각 token 위치의 hidden vector를 vocabulary 전체 token 후보 점수로 바꾸는 Linear layer다.

```text
(B, T, emb_dim)
-> lm_head
-> (B, T, vocab_size)
```

즉 각 위치마다 다음 token 후보 전체에 대한 logits를 만든다.

---

### Q. logits는 무엇인가요?

logits는 softmax를 적용하기 전의 원시 점수다.

GPT에서는 각 위치마다 다음 token 후보들이 얼마나 그럴듯한지 나타내는 점수표다.

```text
logits: (B, T, vocab_size)
```

---

### Q. targets가 있을 때와 없을 때 forward가 다른 이유는?

targets가 없으면 생성/추론용이다.

```text
input -> logits 반환
```

targets가 있으면 학습/평가용이다.

```text
input + targets -> loss와 logits 반환
```

즉:

```text
targets 없음: 다음 token 점수만 필요
targets 있음: 정답과 비교해 loss 계산 필요
```

---

### Q. Cross Entropy Loss는 어떻게 계산하나요?

GPT logits는:

```text
(B, T, vocab_size)
```

targets는:

```text
(B, T)
```

PyTorch cross entropy는:

```text
(N, class_num), (N,)
```

형태를 기대한다.

그래서 펼친다.

```python
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
)
```

의미:

```text
모든 batch의 모든 token 위치를 하나의 classification 문제 목록으로 펼쳐 loss 계산
```

---

## 9. Training

### Q. 학습이란 무엇인가요?

학습은 **모델의 예측과 정답의 차이인 loss를 줄이도록 파라미터를 반복적으로 조정하는 과정**이다.

기본 루프:

```text
forward
-> loss
-> backward
-> optimizer.step
```

---

### Q. GPT 사전 학습 루프를 설명해보세요.

```text
for epoch:
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        loss = calc_loss_batch(...)
        loss.backward()
        optimizer.step()
```

각 단계:

```text
zero_grad:
이전 batch gradient 제거

forward/loss:
현재 batch 예측과 정답 비교

backward:
파라미터별 gradient 계산

optimizer.step:
파라미터 업데이트
```

---

### Q. train loss와 validation loss는 왜 둘 다 보나요?

train loss:

```text
학습 데이터에서 얼마나 틀리는지
```

validation loss:

```text
학습에 직접 쓰지 않은 데이터에서 얼마나 틀리는지
```

해석:

```text
train loss 감소, val loss 감소:
정상 학습

train loss 감소, val loss 증가:
과적합 가능성

둘 다 안 줄어듦:
모델/학습률/데이터 문제 가능성
```

---

### Q. checkpoint는 왜 저장하나요?

학습은 오래 걸리고 중간에 끊길 수 있다.

checkpoint에는 다음을 저장한다.

```text
model state
optimizer state
epoch
global_step
```

optimizer state도 중요한 이유:

```text
AdamW 같은 optimizer는 내부 이동평균 상태를 가진다.
이걸 저장하지 않으면 이어서 학습할 때 흐름이 달라질 수 있다.
```

---

## 10. Generation

### Q. generate는 학습인가요?

아니다.

generate는 현재 학습된 모델로 다음 token을 선택해 이어 붙이는 추론 과정이다.

```text
loss.backward 없음
optimizer.step 없음
```

그래서 `torch.no_grad()`와 `model.eval()`을 사용한다.

---

### Q. generate의 기본 흐름은?

```text
현재 idx
-> 마지막 context_size개만 자름
-> model(idx_cond)
-> logits
-> 마지막 위치 logits
-> 다음 token 선택
-> idx 뒤에 붙임
-> 반복
```

코드 개념:

```python
last_logits = logits[:, -1, :]
next_id = torch.argmax(last_logits, dim=-1, keepdim=True)
idx = torch.cat((idx, next_id), dim=1)
```

---

### Q. 왜 마지막 위치 logits만 보나요?

생성에서는 현재까지의 전체 문맥 다음에 올 token 하나만 필요하다.

모델은 모든 위치의 logits를 만들지만, 새로 붙일 token은 마지막 위치의 다음 token 예측에서 나온다.

```text
logits: (B, T, vocab_size)
last_logits: (B, vocab_size)
```

---

### Q. Greedy decoding은 무엇인가요?

Greedy decoding은 마지막 logits에서 가장 점수가 높은 token을 항상 선택하는 방식이다.

```python
next_id = torch.argmax(last_logits, dim=-1)
```

장점:

```text
간단하고 안정적
```

단점:

```text
항상 비슷하고 단조로운 문장이 나올 수 있음
```

---

### Q. Temperature는 무엇인가요?

temperature는 생성 시 확률 분포의 날카로움을 조절하는 값이다.

```python
logits = logits / temperature
```

해석:

```text
temperature 낮음:
높은 점수 token에 더 쏠림

temperature 높음:
더 다양한 token 선택 가능
```

음수 temperature는 의미가 없다.

이유:

```text
logits 순서가 뒤집혀 낮게 평가된 token이 선택될 수 있기 때문
```

---

### Q. Top-k sampling은 무엇인가요?

Top-k sampling은 vocab 전체 후보 중 상위 k개 token만 남기고 그 안에서 확률적으로 선택하는 방법이다.

목적:

```text
너무 낮은 확률의 이상한 token은 제거
greedy보다 다양한 생성 가능
```

중요:

```text
top-k는 학습이 아니다.
파라미터를 바꾸지 않는다.
생성할 때 token을 고르는 전략이다.
```

---

## 11. Fine-tuning

### Q. 미세튜닝은 무엇인가요?

미세튜닝은 사전 학습한 GPT backbone을 특정 task에 맞게 추가 학습하는 과정이다.

우리 과제에서는 NSMC 감성 분류를 수행한다.

```text
입력: 영화 리뷰
출력: 부정/긍정
```

---

### Q. 사전 학습과 미세튜닝은 무엇이 다른가요?

사전 학습:

```text
다음 token 예측
출력 shape: (B, T, vocab_size)
```

미세튜닝:

```text
문장 전체 label 예측
출력 shape: (B, num_labels)
```

즉 사전 학습은 언어 패턴 학습이고, 미세튜닝은 특정 문제 해결을 위한 추가 학습이다.

---

### Q. GPTForSequenceClassification은 무엇인가요?

GPT backbone 위에 classification head를 붙인 모델이다.

구조:

```text
input_ids
-> GPT embedding
-> GPT blocks
-> final norm
-> 문장 대표 vector
-> classifier
-> logits: (B, 2)
```

---

### Q. lm_head와 classifier는 무엇이 다른가요?

lm_head:

```text
다음 token 예측용
emb_dim -> vocab_size
```

classifier:

```text
감성 label 예측용
emb_dim -> num_labels
```

감성 분류에서는 vocab 전체 점수가 아니라:

```text
부정 점수, 긍정 점수
```

만 필요하다.

---

### Q. 마지막 token hidden state를 문장 대표 벡터로 쓰는 이유는?

GPT의 causal attention에서 마지막 token은 앞의 모든 token을 참고할 수 있다.

따라서 마지막 token hidden state는 단순히 마지막 글자 정보만 가진 것이 아니라, 앞 문맥이 반영된 vector다.

```text
sentence_vec = x[:, -1, :]
```

다만 padding이 있는 경우 마지막 위치가 pad일 수 있으므로, 더 정확한 구현은 마지막 유효 token 위치를 찾아야 한다.

---

### Q. sentiment train loop는 pretraining loop와 무엇이 다른가요?

비슷하지만 target이 다르다.

사전 학습:

```text
input_ids, target_ids
다음 token 맞히기
```

감성 분류:

```text
input_ids, labels
긍정/부정 맞히기
```

공통점:

```text
loss 계산
backward
optimizer.step
평균 loss와 accuracy/loss 기록
```

---

## 12. 자주 나올 수 있는 깊은 꼬리질문

### Q. 왜 GPT는 미래 token을 보면 안 되나요?

다음 token을 예측하는 모델인데 미래 token을 보면 정답을 미리 보는 것이 된다.

그래서 causal mask로 미래 위치를 가린다.

---

### Q. 왜 softmax를 직접 하지 않고 cross_entropy를 쓰나요?

PyTorch의 `cross_entropy`는 내부적으로:

```text
log_softmax + negative log likelihood
```

를 안정적으로 계산한다.

그래서 loss 계산 전에 직접 softmax를 하지 않는다.

---

### Q. 왜 `reshape(-1, vocab_size)`를 하나요?

GPT는 모든 token 위치에서 다음 token을 예측한다.

logits:

```text
(B, T, vocab_size)
```

를:

```text
(B*T, vocab_size)
```

로 펼쳐 모든 위치를 하나의 classification 문제 목록으로 만든다.

---

### Q. 왜 `model.train()`과 `model.eval()`을 구분하나요?

Dropout 같은 layer는 학습/평가 동작이 다르다.

```text
train:
dropout 적용

eval:
dropout 비활성화
```

그래서 학습 때는 `train()`, 평가/생성 때는 `eval()`을 쓴다.

---

### Q. 왜 `torch.no_grad()`를 쓰나요?

평가나 생성에서는 파라미터를 업데이트하지 않는다.

따라서 gradient 계산 그래프가 필요 없다.

장점:

```text
메모리 절약
속도 향상
불필요한 gradient 계산 방지
```

---

### Q. LLM 구현에서 가장 중요한 shape는 무엇인가요?

반드시 기억해야 할 shape:

```text
input ids:
(B, T)

embedding:
(B, T, d_model)

Q/K/V:
(B, n_heads, T, head_dim)

attention score:
(B, n_heads, T, T)

GPT logits:
(B, T, vocab_size)

classification logits:
(B, num_labels)
```

---

### Q. 직접 구현하면서 가장 중요한 학습 포인트는 무엇이었나요?

핵심은 LLM이 마법이 아니라 아래 부품들의 조합이라는 점을 이해한 것이다.

```text
텍스트를 token id로 바꾸는 tokenizer
token id를 vector로 바꾸는 embedding
문맥을 섞는 attention
표현을 가공하는 FFN
깊은 학습을 돕는 residual/LayerNorm
다음 token 점수를 만드는 lm_head
loss를 줄이는 training loop
task에 맞게 바꾸는 fine-tuning
```

---

## 13. 면접용 1분 답변

### Q. mini GPT를 직접 구현했다고 했는데, 어떻게 만들었나요?

면접에서 짧게 답하면:

```text
LLM은 Large Language Model의 약자로, 현재 문맥을 보고 다음 token을 예측하도록 학습한 언어 모델입니다.
저희는 먼저 UTF-8 byte-level BPE tokenizer를 직접 구현해 문장을 token id로 바꿨고,
Dataset에서 input과 target을 한 칸 차이 나는 window로 구성했습니다.
그 다음 token embedding과 position embedding을 더해 Transformer 입력을 만들고,
causal multi-head self-attention으로 미래 token을 보지 못하게 하면서 문맥 정보를 반영했습니다.
TransformerBlock은 LayerNorm, attention, FeedForward, residual connection으로 구성했고,
GPTModel은 여러 block을 쌓은 뒤 lm_head로 vocab 전체 logits를 출력하게 했습니다.
학습은 logits와 target token id 사이의 cross entropy loss를 줄이는 방식으로 진행했고,
마지막에는 GPT backbone 위에 classifier head를 붙여 NSMC 감성 분류 미세튜닝까지 구현했습니다.
```

---

## 14. 최종 암기 목록

반드시 입으로 설명할 수 있어야 하는 것:

```text
LLM = Large Language Model
LLM의 기본 목표 = 다음 token 예측
Tokenizer의 역할 = text <-> token ids
BPE의 역할 = 자주 등장하는 byte/token pair를 merge해 vocab 구성
Dataset의 역할 = input/target window 생성
Embedding의 역할 = token id를 vector로 변환
Position embedding의 역할 = 순서 정보 추가
Attention의 역할 = token 간 참고 비율 계산
Causal mask의 역할 = 미래 token 차단
Multi-head의 역할 = 여러 관점으로 attention 수행
LayerNorm의 역할 = token vector feature 분포 안정화
Residual의 역할 = 원래 정보 보존과 gradient 흐름 개선
FFN의 역할 = 각 token vector 내부 표현 가공
lm_head의 역할 = hidden vector를 vocab logits로 변환
Cross entropy의 역할 = 정답 token 점수가 높아지도록 loss 계산
Generate의 역할 = 마지막 logits에서 다음 token 선택 후 이어 붙이기
Fine-tuning의 역할 = GPT backbone을 특정 task에 맞게 추가 학습
Classifier의 역할 = 문장 vector를 label logits로 변환
```

---

## 15. 마지막 정리

LLM 구현은 크게 세 단계로 볼 수 있다.

```text
1. 텍스트를 숫자로 바꾸는 단계
Tokenizer, BPE, Dataset, DataLoader

2. 숫자를 문맥 벡터로 처리하는 단계
Embedding, Attention, TransformerBlock, GPTModel

3. 학습하고 사용하는 단계
Cross entropy, optimizer, generation, checkpoint, fine-tuning
```

이 흐름을 설명할 수 있으면 “LLM을 구현했다”는 말을 단순 코드 작성이 아니라 구조와 학습 원리까지 이해한 것으로 말할 수 있다.
