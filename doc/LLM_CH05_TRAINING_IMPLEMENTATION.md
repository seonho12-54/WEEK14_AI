# 5장 GPT 사전 학습 구현 흐름 정리

## 0. 5장의 핵심 한 줄

**5장은 2장에서 만든 데이터, 3장에서 만든 attention, 4장에서 만든 GPTModel을 실제로 학습시키는 장이다.**

지금까지 만든 흐름은 아래와 같다.

```text
문장
-> BPETokenizer.encode()
-> token ids
-> GPTDataset / DataLoader
-> InputEmbedding
-> TransformerBlock
-> logits
```

5장에서는 여기에 아래 흐름이 붙는다.

```text
logits
-> target과 비교
-> loss 계산
-> loss.backward()
-> optimizer.step()
-> checkpoint 저장
-> 샘플 생성
```

즉, 5장은 **모델을 실제로 훈련시키는 방법**을 다룬다.

---

## 1. 우리 과제에서 5장과 직접 연결되는 파일

구현 파일은 `src/train.py`이다.

| 함수 | 역할 | 중요도 |
| --- | --- | --- |
| `calc_loss_batch` | 한 배치의 loss 계산 | 매우 중요 |
| `calc_loss_loader` | 여러 배치의 평균 loss 계산 | 매우 중요 |
| `save_checkpoint` | 모델/optimizer 상태 저장 | 중요 |
| `load_checkpoint` | 저장된 학습 상태 복원 | 중요 |
| `generate` | temperature, top-k 기반 token 생성 | 매우 중요 |
| `generate_and_print_sample` | 시작 문장으로 샘플 생성 후 출력 | 중요 |
| `train_model` | 전체 사전 학습 루프 | 매우 중요 |
| `plot_losses` | loss 그래프 출력 | 제공 함수 |

---

## 2. 학습이란 무엇인가

학습이란, 모델의 예측과 정답의 차이인 loss를 줄이도록 파라미터를 반복적으로 조정하는 과정.

GPT 학습의 목표는 단순하다.

```text
현재 token들을 보고 다음 token을 맞히기
```

예를 들어 token id가 아래와 같다면:

```text
[10, 11, 12, 13, 14]
```

Dataset은 input과 target을 이렇게 만든다.

```text
input  = [10, 11, 12, 13]
target = [11, 12, 13, 14]
```

모델은 각 위치에서 다음 token을 맞힌다.

```text
10을 보고 11 예측
11을 보고 12 예측
12를 보고 13 예측
13을 보고 14 예측
```

**GPT는 문장 전체의 정답 하나를 맞히는 모델이 아니라, 모든 위치에서 다음 token을 맞히는 모델이다.**

---

## 3. logits와 target의 관계

`GPTModel.forward(idx, targets)`를 호출하면 모델은 logits를 만든다.

logits는 각 위치에서 모델이 예측한 다음 token후보들의 점수,

target은 그 위치에서 실제로 와야 하는 정답 token id 입니다.

```text
idx shape     : (B, T)
logits shape  : (B, T, vocab_size)
targets shape : (B, T)
```

예를 들어:

```text
B = 8
T = 16
vocab_size = 3000
```

이면:

```text
idx     : (8, 16)
logits  : (8, 16, 3000)
targets : (8, 16)
```

뜻은 아래와 같다.

```text
8개의 window가 있고,
각 window에는 token 위치가 16개 있고,
각 위치마다 다음 token 후보 3000개의 점수가 있다.
```

여기서 `targets`는 각 위치의 정답 token id이다.

---

## 4. Cross Entropy Loss

GPT의 loss는 보통 cross entropy로 계산한다.

정의:

```text
모델이 정답 token에 얼마나 낮은 확률을 줬는지 측정하는 손실 함수
```

쉽게 말하면:

```text
정답 token 점수가 높으면 loss가 작아지고,
정답 token 점수가 낮으면 loss가 커진다.
```

PyTorch에서는 보통 아래처럼 쓴다.

```python
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
)
```

왜 reshape가 필요한가?

`F.cross_entropy`는 보통 아래 모양을 기대한다.

```text
예측값: (N, class_num)
정답값: (N,)
```

그런데 GPT logits는 아래 모양이다.

```text
logits  : (B, T, vocab_size)
targets : (B, T)
```

그래서 batch와 token 위치를 한 줄로 펼친다.

```text
logits  : (B, T, vocab_size) -> (B*T, vocab_size)
targets : (B, T)             -> (B*T,)
```

즉, 모든 token 위치의 다음 token 예측 문제를 한꺼번에 loss로 계산한다.

---

## 5. calc_loss_batch

`calc_loss_batch`는 한 배치의 loss를 계산하는 함수다.

해야 할 일:

```text
1. input_batch를 device로 옮긴다.
2. target_batch를 device로 옮긴다.
3. model(input_batch, target_batch)를 호출한다.
4. 반환된 loss를 돌려준다.
```

흐름:

```text
input_batch, target_batch
-> GPU/CPU device로 이동
-> model forward
-> loss 계산
-> loss 반환
```

주의할 점:

```text
model 내부에서 targets가 있으면 loss를 계산하도록 만들어두었다면,
calc_loss_batch에서는 model(input_batch, target_batch)를 호출하면 된다.
```

---

## 6. calc_loss_loader

`calc_loss_loader`는 DataLoader의 여러 배치를 돌면서 평균 loss를 계산한다.

훈련 중간에 아래를 확인할 때 쓴다.

```text
train loss가 줄고 있는가?
validation loss도 줄고 있는가?
과적합이 생기고 있는가?
```

중요한 점은 평가할 때 gradient를 계산하지 않는 것이다.

```python
with torch.no_grad():
    ...
```

왜 필요한가?

```text
평가에서는 파라미터를 업데이트하지 않는다.
따라서 gradient 계산이 필요 없다.
메모리와 시간을 아낄 수 있다.
```

또 평가할 때는 모델을 eval mode로 바꾼다.

```python
model.eval()
```

평가가 끝나면 다시 train mode로 돌린다.

```python
model.train()
```

이유:

```text
Dropout은 train/eval에서 동작이 다르다.
학습 중에는 dropout을 적용하고,
평가 중에는 dropout을 끈다.
```

---

## 7. train_model 전체 흐름

GPT 사전 학습 루프는 아래 순서로 진행된다.

```text
for epoch:
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        loss = calc_loss_batch(...)
        loss.backward()
        optimizer.step()
```

각 줄의 의미:

```text
optimizer.zero_grad()
이전 배치에서 계산된 gradient를 초기화한다.

loss = calc_loss_batch(...)
현재 배치에서 모델의 예측이 얼마나 틀렸는지 계산한다.

loss.backward()
loss를 줄이기 위해 각 파라미터를 어느 방향으로 바꿔야 하는지 gradient를 계산한다.

optimizer.step()
계산된 gradient를 이용해 파라미터를 실제로 업데이트한다.
```

MNIST와 비교하면 같은 흐름이다.

```text
Forward
-> Loss
-> Backward
-> Optimizer update
```

다만 MNIST에서는 NumPy로 직접 했고, GPT에서는 PyTorch autograd가 `backward`를 자동으로 처리한다.

---

## 8. global_step

`global_step`은 전체 학습 과정에서 optimizer update가 몇 번 일어났는지 세는 값이다.

```text
epoch는 전체 데이터 반복 횟수
global_step은 배치 업데이트 횟수
```

예를 들어:

```text
1 epoch에 배치가 100개 있다면,
1 epoch가 끝났을 때 global_step은 100 증가한다.
```

이 값은 주로 아래 용도로 쓴다.

```text
몇 step마다 평가할지 결정
몇 step마다 checkpoint 저장할지 결정
학습 로그를 기록
```

---

## 9. eval_freq와 eval_iter

`eval_freq`는 몇 step마다 평가할지 정하는 값이다.

```text
eval_freq = 100
-> 100 step마다 train/val loss 평가
```

`eval_iter`는 평가할 때 몇 배치만 볼지 정하는 값이다.

```text
eval_iter = 10
-> validation 전체를 다 보지 않고 10개 배치만 평균냄
```

왜 일부만 보나?

```text
전체 validation을 매번 평가하면 시간이 오래 걸린다.
학습 중간 확인용이면 일부 배치만 봐도 충분하다.
```

---

## 10. Checkpoint

checkpoint는 학습 상태 저장 파일이다.

저장해야 할 것:

```text
model state
optimizer state
epoch
global_step
```

모델 파라미터만 저장하면 부족하다.

이유:

```text
optimizer도 내부 상태를 가진다.
AdamW 같은 optimizer는 momentum/variance 정보를 들고 있다.
이 상태를 복원하지 않으면 이어서 학습할 때 흐름이 달라질 수 있다.
```

저장 흐름:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "global_step": global_step,
}, path)
```

로드 흐름:

```python
checkpoint = torch.load(path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

checkpoint는 특히 Colab에서 중요하다.

```text
런타임이 끊겨도 저장된 checkpoint부터 다시 시작할 수 있다.
```

---

## 11. generate와 greedy 생성

생성은 학습과 다르다.

학습:

```text
idx와 targets를 넣음
loss를 계산함
파라미터를 업데이트함
```

생성:

```text
idx만 넣음
logits를 얻음
다음 token을 고름
기존 idx 뒤에 붙임
반복함
```

가장 단순한 방식은 greedy decoding이다.

```text
마지막 위치 logits에서 점수가 가장 높은 token을 고른다.
```

흐름:

```text
idx
-> model(idx)
-> logits
-> logits[:, -1, :]
-> argmax
-> next_id
-> idx 뒤에 붙이기
```

`logits[:, -1, :]`의 뜻:

```text
모든 배치에서
마지막 token 위치의
vocab 전체 점수를 가져온다.
```

---

## 12. temperature

temperature는 생성할 때 확률 분포를 얼마나 날카롭거나 부드럽게 만들지 정한다.

```text
temperature가 낮다 -> 가장 높은 점수 쪽으로 강하게 쏠림
temperature가 높다 -> 다양한 token이 나올 가능성 증가
```

공식 느낌:

```python
logits = logits / temperature
```

예:

```text
temperature = 0
greedy처럼 가장 높은 token만 선택

temperature = 0.7
조금 보수적인 생성

temperature = 1.0
기본적인 확률 샘플링

temperature = 1.5
더 다양하지만 이상한 문장이 나올 가능성 증가
```

중요:

```text
temperature는 학습 파라미터가 아니다.
생성할 때 사용하는 디코딩 설정이다.
```

---

## 13. top-k sampling

top-k는 다음 token 후보를 상위 k개로 제한하는 방법이다.

예를 들어 vocab_size가 3000이고 `top_k=40`이면:

```text
3000개 token 중 점수가 높은 40개만 후보로 남긴다.
나머지는 선택되지 못하게 한다.
```

왜 쓰나?

```text
너무 낮은 확률의 이상한 token이 뽑히는 것을 막기 위해서
```

흐름:

```text
last_logits
-> top_k개만 남김
-> softmax
-> 확률 기반 sampling
```

greedy와 차이:

```text
greedy는 항상 1등 token만 선택
top-k sampling은 상위 k개 안에서 확률적으로 선택
```

---

## 14. eos_id

`eos_id`는 문장 끝 token id이다.

생성 중 `eos_id`가 나오면 멈출 수 있다.

```text
모델이 <eos>를 생성했다
-> 문장이 끝났다고 판단
-> 생성 종료
```

과제에서는 tokenizer의 `<eos>` id를 사용한다.

```python
eos_id = tokenizer.get_eos_id()
```

---

## 15. generate_and_print_sample

이 함수는 학습 중간에 모델이 어떤 문장을 생성하는지 확인하는 용도다.

흐름:

```text
start_context 문자열
-> tokenizer.encode()
-> torch.tensor로 변환
-> generate()
-> tokenizer.decode()
-> print
```

예:

```text
시작 문장: "이 영화는"
모델 생성: "이 영화는 정말 재미있다..."
```

처음 학습 전에는 말이 안 되는 결과가 나올 수 있다.

학습이 진행되면:

```text
처음에는 랜덤한 token 나열
조금 지나면 자주 나오는 조사/어미/단어 조합 등장
더 학습하면 짧은 문장 구조가 보이기 시작
```

---

## 16. train loss와 validation loss 해석

학습 중 반드시 loss를 기록해야 한다.

| 상황 | 해석 |
| --- | --- |
| train loss 감소, val loss 감소 | 정상적으로 학습 중 |
| train loss 감소, val loss 증가 | 과적합 가능성 |
| 둘 다 거의 감소하지 않음 | learning rate, 모델 크기, 데이터 문제 가능성 |
| loss가 갑자기 nan | learning rate 과도, gradient 폭주, 수치 안정성 문제 |
| train loss만 매우 낮음 | 데이터 암기 가능성 |

MNIST와 마찬가지로 GPT도 loss curve를 보고 학습 상태를 판단한다.

---

## 17. model.train(), model.eval(), torch.no_grad()

PyTorch 학습에서 반드시 구분해야 한다.

### model.train()

학습 모드로 바꾼다.

```text
Dropout이 켜진다.
파라미터 업데이트를 위한 forward를 수행한다.
```

### model.eval()

평가 모드로 바꾼다.

```text
Dropout이 꺼진다.
검증 loss나 샘플 생성을 안정적으로 확인한다.
```

### torch.no_grad()

gradient 계산을 끈다.

```text
평가/생성에서는 backward를 하지 않으므로 gradient가 필요 없다.
메모리를 절약하고 속도를 높인다.
```

---

## 18. 5장에서 꼭 암기해야 할 흐름

### 한 배치 학습

```text
input_batch, target_batch
-> device 이동
-> model(input_batch, target_batch)
-> loss
-> optimizer.zero_grad()
-> loss.backward()
-> optimizer.step()
```

### 평가

```text
model.eval()
with torch.no_grad():
    여러 batch loss 평균 계산
model.train()
```

### 생성

```text
idx
-> model(idx)
-> 마지막 위치 logits
-> temperature / top-k
-> next_id 선택
-> idx 뒤에 붙이기
-> 반복
```

### 저장과 복원

```text
save_checkpoint:
model state + optimizer state + epoch + global_step 저장

load_checkpoint:
저장된 상태를 불러와 이어서 학습
```

---

## 19. MNIST와 연결해서 이해하기

MNIST에서 했던 학습 흐름:

```text
Forward
-> Loss
-> Backward
-> Optimizer update
```

GPT에서도 같다.

```text
Forward:
idx -> logits

Loss:
logits와 targets 비교

Backward:
loss.backward()

Update:
optimizer.step()
```

차이점:

```text
MNIST는 이미지 하나당 숫자 class 하나를 예측
GPT는 모든 token 위치마다 다음 token을 예측
```

즉, GPT는 한 문장 조각 안에서 여러 개의 classification 문제를 동시에 푸는 것처럼 볼 수 있다.

---

## 20. 과제 구현 순서 추천

`src/train.py`는 아래 순서로 구현하는 것이 좋다.

```text
1. calc_loss_batch
2. calc_loss_loader
3. save_checkpoint
4. load_checkpoint
5. generate
6. generate_and_print_sample
7. train_model
```

이유:

```text
loss 계산이 되어야 학습 루프를 만들 수 있다.
평균 loss 계산이 되어야 학습 상태를 평가할 수 있다.
checkpoint는 오래 걸리는 학습을 보호한다.
generate는 학습 결과를 눈으로 확인하는 도구다.
train_model은 위 기능들을 모두 조합한다.
```

---

## 21. 구현할 때 자주 헷갈리는 부분

### 1. targets가 없으면 loss를 계산하지 않는다

```text
학습: model(input_batch, target_batch)
생성: model(idx)
```

### 2. logits에서 마지막 위치만 보는 것은 생성 때다

```text
학습 loss 계산:
모든 위치의 logits 사용

생성:
마지막 위치 logits만 사용
```

### 3. softmax는 loss 계산 때 직접 안 해도 된다

`F.cross_entropy`는 내부적으로 softmax와 negative log likelihood를 함께 처리한다.

따라서 loss 계산 전에 직접 softmax를 하지 않는다.

### 4. optimizer.zero_grad()를 빼먹으면 안 된다

PyTorch의 gradient는 기본적으로 누적된다.

그래서 매 배치마다 기존 gradient를 지워야 한다.

### 5. checkpoint에는 optimizer도 저장한다

model만 저장하면 AdamW의 내부 상태가 사라진다.

---

## 22. 최종 요약

5장은 GPT를 실제로 학습시키는 장이다.

핵심은 아래 네 가지다.

```text
1. logits와 targets로 loss를 계산한다.
2. loss.backward()로 gradient를 계산한다.
3. optimizer.step()으로 파라미터를 업데이트한다.
4. generate/checkpoint/evaluation으로 학습 상태를 확인하고 보존한다.
```

가장 중요한 구현 흐름:

```text
DataLoader
-> calc_loss_batch
-> backward
-> optimizer update
-> calc_loss_loader
-> generate sample
-> checkpoint
```

이 장을 이해하면 GPT가 단순히 구조만 있는 모델이 아니라, 실제 데이터로 학습되고 문장을 생성하는 모델이 된다.
