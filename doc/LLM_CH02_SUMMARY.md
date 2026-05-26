# 2장 텍스트 데이터 다루기

## 0. 이 장의 핵심 한 줄

**2장은 사람이 읽는 텍스트를 GPT가 계산할 수 있는 숫자 텐서로 바꾸는 전체 전처리 과정을 다룬다.**

```text
원문 텍스트
-> 토큰화
-> 토큰 ID
-> 입력/타깃 샘플
-> 토큰 임베딩
-> 위치 임베딩
-> GPT 입력 텐서
```

1장에서 GPT가 “다음 토큰을 예측한다”고 배웠다면, 2장은 그 “토큰”이 실제로 무엇이고, 어떻게 모델에 들어가는지 배우는 장이다.

---

## 1. 2장을 왜 배워야 하는가

신경망은 글자를 그대로 이해하지 못한다.

모델이 계산할 수 있는 것은 숫자다.

따라서 LLM의 첫 단계는 다음과 같다.

```text
텍스트를 숫자로 바꾼다.
```

MNIST에서는 이미지가 이미 `784`개의 숫자였다.

하지만 LLM에서는 문장이 원본 입력이므로, 문장을 숫자로 바꾸는 과정이 필요하다.

| MNIST | LLM |
| --- | --- |
| 이미지 픽셀 | 텍스트 |
| 이미 숫자 배열 | 아직 문자열 |
| 바로 신경망 입력 가능 | 토큰화와 임베딩 필요 |

### 꼭 암기

**LLM은 원시 텍스트를 직접 처리하지 못한다. 텍스트는 token ID와 embedding vector로 바뀐 뒤 모델에 들어간다.**

---

## 2. 전체 파이프라인 먼저 보기

2장의 전체 흐름은 아래 하나로 정리된다.

```text
"이 영화는 정말 좋았다"
-> ["이", "영화", "는", "정말", "좋았다"]
-> [31, 205, 7, 81, 499]
-> input  = [31, 205, 7, 81]
-> target = [205, 7, 81, 499]
-> token embedding
-> position embedding 더하기
-> GPT 입력
```

우리 과제 파일로 연결하면 다음과 같다.

| 단계 | 과제 파일 |
| --- | --- |
| BPE tokenizer | `src/bpe.py` |
| Dataset / input-target 생성 | `src/dataset.py` |
| Token embedding / position embedding | `src/embeddings.py` |
| 다음 장 attention 입력 | `src/attention.py` |

---

## 3. 갑자기 나오는 용어를 이해하는 법

LLM 책에서는 zero-shot, few-shot, token, embedding, vocabulary 같은 말이 갑자기 나온다.

이 용어들은 따로따로 외우면 어렵고, 아래 흐름 안에서 봐야 한다.

```text
사용자가 프롬프트를 쓴다.
-> 프롬프트가 token으로 쪼개진다.
-> token이 token ID로 바뀐다.
-> token ID가 embedding으로 바뀐다.
-> GPT가 다음 token을 예측한다.
```

### zero-shot과 few-shot은 어디서 나온 개념인가

1장에서 zero-shot과 few-shot은 GPT가 사전 학습 후 어떤 작업을 수행하는 방식으로 등장했다.

| 용어 | 의미 | 어디서 쓰이나 |
| --- | --- | --- |
| Zero-shot | 예시 없이 바로 지시만 주는 방식 | 프롬프트만 보고 답변 |
| Few-shot | 예시 몇 개를 같이 주는 방식 | 프롬프트 안에 예제를 넣어 답변 유도 |

예:

```text
Zero-shot:
"이 리뷰를 긍정/부정으로 분류해줘: 이 영화 정말 좋았다"

Few-shot:
"재밌다 -> 긍정"
"지루하다 -> 부정"
"이 영화 정말 좋았다 -> ?"
```

여기서 중요한 연결은 이것이다.

**zero-shot과 few-shot도 결국 모델 입장에서는 그냥 긴 텍스트 입력이다.**

즉, few-shot 예시도 모두 tokenizer를 거쳐 token ID가 되고, embedding이 되어 GPT에 들어간다.

그래서 2장의 tokenization은 zero-shot/few-shot 같은 프롬프트 방식의 바닥 개념이다.

---

## 4. 임베딩이란 무엇인가

임베딩은 단어, 토큰, 이미지 같은 비수치 대상을 신경망이 계산할 수 있는 벡터로 바꾸는 것이다.

정의:

```text
embedding = 이산적인 대상을 연속적인 벡터 공간의 점으로 바꾸는 것
```

예:

```text
"영화" -> [0.12, -0.48, 1.03, ...]
"재미" -> [0.22, -0.31, 0.88, ...]
```

왜 필요한가?

토큰 ID는 그냥 번호다.

```text
영화 = 205
재미 = 81
```

이 숫자 자체에는 의미가 없다.

`205`가 `81`보다 크다고 해서 더 중요한 단어라는 뜻이 아니다.

그래서 token ID를 의미를 담을 수 있는 벡터로 바꿔야 한다.

### 꼭 암기

**token ID는 단순한 정수 번호이고, embedding은 모델이 학습하는 의미 벡터다.**

---

## 5. Tokenization이란 무엇인가

Tokenization은 텍스트를 모델이 읽는 단위인 token으로 나누는 과정이다.

예:

```text
문장: "Hello, world!"
토큰: ["Hello", ",", "world", "!"]
```

토큰은 꼭 단어 하나일 필요가 없다.

토큰은 다음 중 하나일 수 있다.

| 형태 | 예 |
| --- | --- |
| 단어 | `movie` |
| 구두점 | `!`, `,`, `?` |
| 부분단어 | `un`, `believ`, `able` |
| 문자 또는 byte | UTF-8 byte |

한국어에서는 공백 기준 단어 분리가 특히 어렵다.

예:

```text
재미있다
재미있었다
재미있네요
재미있지만
```

사람은 비슷한 의미로 보지만, 공백 기준 tokenizer는 모두 다른 단어처럼 볼 수 있다.

그래서 LLM에서는 부분단어 또는 byte 기반 tokenizer가 중요하다.

### 꼭 암기

**Tokenization은 텍스트를 모델이 다룰 수 있는 작은 단위로 자르는 과정이다.**

---

## 6. Vocabulary란 무엇인가

Vocabulary는 tokenizer가 알고 있는 token 목록이다.

각 token은 고유한 정수 ID를 가진다.

예:

| token | token ID |
| --- | --- |
| `<pad>` | 0 |
| `<unk>` | 1 |
| `영화` | 205 |
| `좋았다` | 499 |

토큰화된 문자열을 token ID로 바꾸려면 vocabulary가 필요하다.

```text
["영화", "좋았다"]
-> [205, 499]
```

반대로 token ID를 다시 텍스트로 바꾸려면 역방향 vocabulary가 필요하다.

```text
[205, 499]
-> ["영화", "좋았다"]
```

이것이 tokenizer의 `encode()`와 `decode()`다.

| 함수 | 역할 |
| --- | --- |
| encode | 텍스트를 token ID 목록으로 변환 |
| decode | token ID 목록을 텍스트로 복원 |

### 꼭 암기

**Tokenizer의 핵심 기능은 encode와 decode다.**

---

## 7. 특수 토큰이란 무엇인가

특수 토큰은 일반 단어가 아니라 모델에게 문맥 정보를 알려주는 약속된 token이다.

예:

| 특수 토큰 | 의미 |
| --- | --- |
| `<pad>` | 길이를 맞추기 위한 padding |
| `<unk>` | 모르는 token |
| `<bos>` | 문장 시작 |
| `<eos>` | 문장 끝 |
| `<endoftext>` | 문서 경계 |

특수 토큰이 필요한 이유:

```text
문장이 어디서 시작하고 끝나는지 알려준다.
배치 안에서 길이를 맞춘다.
서로 관련 없는 문서를 구분한다.
```

우리 과제 공지에서는 기본 특수 토큰 ID가 정해져 있다.

| ID | token |
| --- | --- |
| 0 | `<pad>` |
| 1 | `<unk>` |
| 2 | `<bos>` |
| 3 | `<eos>` |

### 꼭 암기

**특수 토큰은 모델에게 문장의 시작, 끝, padding, 모르는 token 같은 메타 정보를 알려주는 token이다.**

---

## 8. BPE란 무엇인가

BPE는 Byte Pair Encoding의 약자다.

자주 붙어 나오는 token pair를 하나의 새 token으로 합치는 방식이다.

아주 단순하게 보면:

```text
처음: ["l", "o", "w"]
자주 등장하는 pair: ("l", "o")
merge 후: ["lo", "w"]
```

이 과정을 반복하면 자주 등장하는 문자 조합이나 부분단어가 하나의 token이 된다.

왜 필요한가?

단어 단위 tokenizer는 모르는 단어가 나오면 `<unk>`가 많아질 수 있다.

BPE는 처음 보는 단어도 더 작은 부분단어 또는 byte로 나눠 표현할 수 있다.

예:

```text
처음 보는 단어
-> 부분단어들
-> 그래도 안 되면 byte 단위
```

그래서 BPE는 `<unk>`를 줄이고, 다양한 언어와 새 단어를 처리하는 데 유리하다.

### 우리 과제와 교재의 차이

교재는 설명 중 `tiktoken`을 사용하지만, 우리 과제에서는 금지다.

우리 과제는 직접 구현해야 한다.

```text
교재: tiktoken으로 BPE 사용
과제: src/bpe.py에서 byte-level BPE 직접 구현
```

### 꼭 암기

**BPE는 자주 등장하는 token pair를 반복적으로 합쳐 vocabulary를 만드는 tokenizer 알고리즘이다.**

---

## 9. Byte-level BPE와 한글

우리 과제는 byte-level BPE를 구현한다.

한글 한 글자는 UTF-8에서 보통 byte 3개로 표현된다.

예:

```python
"한".encode("utf-8")
# b'\xed\x95\x9c'
```

사람에게는 `"한"` 한 글자지만, 컴퓨터에게는 byte 3개다.

byte-level BPE는 여기서 시작한다.

```text
문장
-> UTF-8 bytes
-> byte token
-> 자주 붙는 byte/token pair merge
-> BPE token
```

장점:

```text
모든 한글을 최소한 byte 단위로 표현할 수 있다.
처음 보는 단어도 처리할 수 있다.
영어, 숫자, 문장부호, 이모티콘도 처리할 수 있다.
```

decode할 때 주의할 점:

**byte를 하나씩 문자로 바꾸면 한글이 깨질 수 있다.**

따라서 merge token을 원래 byte까지 펼친 뒤, 마지막에 한 번 UTF-8 decode를 해야 한다.

---

## 10. Sliding Window란 무엇인가

Sliding window는 긴 token sequence에서 일정 길이의 학습 샘플을 잘라내는 방법이다.

GPT는 다음 token을 예측하도록 학습한다.

예:

```text
token_ids = [10, 11, 12, 13]
context_length = 3

input  = [10, 11, 12]
target = [11, 12, 13]
```

target은 input보다 한 칸 뒤로 밀려 있다.

이 구조가 바로 next-token prediction이다.

### stride란 무엇인가

stride는 window가 다음 샘플로 이동하는 간격이다.

예:

```text
tokens = [A, B, C, D, E, F]
context_length = 3
```

stride가 1이면:

```text
[A, B, C]
[B, C, D]
[C, D, E]
```

stride가 3이면:

```text
[A, B, C]
[D, E, F]
```

| 값 | 특징 |
| --- | --- |
| 작은 stride | 샘플이 많이 생김, 중복 많음 |
| 큰 stride | 샘플이 적음, 중복 적음 |

### 꼭 암기

**GPTDataset은 input과 target을 한 칸 차이로 만들어 다음 token 예측 문제를 구성한다.**

---

## 11. Dataset과 DataLoader

Dataset은 하나의 샘플을 어떻게 꺼낼지 정의한다.

DataLoader는 그 샘플들을 batch로 묶어준다.

| 개념 | 역할 |
| --- | --- |
| Dataset | `__len__`, `__getitem__`으로 샘플 제공 |
| DataLoader | 여러 샘플을 batch로 묶고 shuffle 처리 |

우리 과제에서는 `src/dataset.py`에서 다음을 구현한다.

```text
GPTDataset.__len__
GPTDataset.__getitem__
create_dataloader
```

출력 형태:

```text
input_batch:  (batch_size, context_length)
target_batch: (batch_size, context_length)
```

예:

```text
input_batch:
[[10, 11, 12],
 [20, 21, 22]]

target_batch:
[[11, 12, 13],
 [21, 22, 23]]
```

---

## 12. Token Embedding

token ID는 정수다.

하지만 신경망은 각 token의 의미와 관계를 벡터 공간에서 학습해야 한다.

그래서 `nn.Embedding(vocab_size, emb_dim)`을 사용한다.

개념적으로는 embedding table에서 token ID에 해당하는 행을 꺼내는 lookup이다.

예:

```text
vocab_size = 3000
emb_dim = 128
```

그러면 embedding table의 shape는 다음과 같다.

```text
(3000, 128)
```

token ID 하나는 128차원 벡터 하나로 바뀐다.

```text
153 -> [0.12, -0.33, ...]  # 길이 128
```

### 꼭 암기

**Embedding layer는 token ID를 의미를 담을 수 있는 학습 가능한 벡터로 바꾸는 lookup table이다.**

---

## 13. Position Embedding

token embedding에는 위치 정보가 없다.

같은 token ID는 문장 어디에 있어도 같은 vector가 된다.

예:

```text
"나"가 첫 번째 위치에 있어도 같은 vector
"나"가 다섯 번째 위치에 있어도 같은 vector
```

하지만 문장에서는 순서가 중요하다.

```text
강아지가 사람을 물었다
사람이 강아지를 물었다
```

단어는 비슷해도 순서가 바뀌면 의미가 달라진다.

그래서 position embedding을 더한다.

```text
input_embedding = token_embedding + position_embedding
```

종류:

| 종류 | 설명 |
| --- | --- |
| 절대 위치 임베딩 | 0번 위치, 1번 위치처럼 고정 위치마다 벡터 학습 |
| 상대 위치 임베딩 | token 사이의 거리 관계를 중심으로 표현 |

GPT 계열 모델은 보통 학습 가능한 절대 위치 임베딩을 사용한다.

### 꼭 암기

**Token embedding은 token의 의미를 담고, position embedding은 token의 순서를 알려준다.**

---

## 14. 2장 전체 shape 흐름

LLM 구현에서 shape를 놓치면 바로 막힌다.

아래 흐름은 반드시 익숙해져야 한다.

```text
텍스트:
"이 영화는 정말 좋았다"

token IDs:
[31, 205, 7, 81, 499]

input_batch:
(batch_size, context_length)

token_embeddings:
(batch_size, context_length, emb_dim)

position_embeddings:
(context_length, emb_dim)

input_embeddings:
(batch_size, context_length, emb_dim)
```

예:

```text
batch_size = 8
context_length = 64
emb_dim = 128

input_ids shape        = (8, 64)
token_embeddings shape = (8, 64, 128)
position_embeddings    = (64, 128)
final input shape      = (8, 64, 128)
```

이 최종 입력이 3장 attention으로 들어간다.

---

## 15. 2장에서 갑자기 나오는 개념 연결표

| 용어 | 왜 등장했나 | 어디에 쓰이나 |
| --- | --- | --- |
| token | 텍스트를 모델 입력 단위로 쪼개기 위해 | tokenizer, dataset, next-token prediction |
| token ID | token을 숫자로 바꾸기 위해 | embedding layer 입력 |
| vocabulary | token과 ID의 매핑표가 필요해서 | encode/decode |
| encode | 텍스트를 token ID로 바꾸기 위해 | 학습 데이터 준비 |
| decode | 모델 출력 ID를 텍스트로 복원하기 위해 | 텍스트 생성 |
| `<unk>` | 모르는 token 처리 | 단순 tokenizer |
| `<pad>` | batch 길이 맞추기 | 미세 조정, 분류 |
| `<bos>` | 시작 표시 | generation 시작 |
| `<eos>` | 끝 표시 | generation 종료 |
| BPE | 모르는 단어를 부분단어/byte로 처리하기 위해 | `src/bpe.py` |
| sliding window | 긴 텍스트에서 학습 샘플을 만들기 위해 | `src/dataset.py` |
| stride | window 이동 간격 조절 | 데이터 샘플 수와 중복 조절 |
| embedding | token ID를 벡터로 바꾸기 위해 | `src/embeddings.py` |
| position embedding | token 순서를 알려주기 위해 | `src/embeddings.py` |
| zero-shot | 예시 없이 프롬프트만 주는 사용 방식 | 프롬프트가 token ID로 변환됨 |
| few-shot | 예시를 프롬프트에 넣는 사용 방식 | 예시까지 모두 token ID로 변환됨 |

---

## 16. 우리 과제에서 2장이 연결되는 파일

| 교재 2장 내용 | 과제 구현 |
| --- | --- |
| 텍스트 토큰화 | `src/bpe.py` |
| 토큰을 token ID로 변환 | `BPETokenizer.encode` |
| token ID를 텍스트로 복원 | `BPETokenizer.decode` |
| BPE vocabulary 저장 | `BPETokenizer.save` |
| BPE vocabulary 로드 | `BPETokenizer.load` |
| sliding window dataset | `src/dataset.py` |
| DataLoader 생성 | `create_dataloader` |
| token embedding | `src/embeddings.py` |
| position embedding | `InputEmbedding.forward` |

---

## 17. 2장에서 반드시 암기할 문장

### 1

**LLM은 텍스트를 직접 읽지 못하므로, 텍스트를 token ID로 바꿔야 한다.**

### 2

**Tokenizer는 encode와 decode를 수행한다.**

### 3

**Vocabulary는 token과 token ID의 매핑표다.**

### 4

**BPE는 자주 등장하는 token pair를 합쳐 부분단어 vocabulary를 만드는 방식이다.**

### 5

**GPTDataset은 input과 target을 한 칸 차이로 만들어 다음 token 예측 문제를 만든다.**

### 6

**Embedding layer는 token ID를 학습 가능한 vector로 바꾸는 lookup table이다.**

### 7

**Position embedding은 token의 순서 정보를 모델에 알려준다.**

### 8

**최종 GPT 입력 shape는 `(batch_size, context_length, emb_dim)`이다.**

---

## 18. 초보자용 비유

LLM에게 문장을 먹이는 과정은 외국어 문장을 숫자 카드로 바꾸는 과정과 비슷하다.

```text
문장: "이 영화는 좋았다"
토큰 카드: ["이", "영화", "는", "좋았다"]
번호 카드: [31, 205, 7, 499]
의미 벡터: 각 번호 카드에 붙은 여러 숫자 특징
위치 벡터: 첫 번째, 두 번째, 세 번째라는 순서 표시
```

모델은 원문 글자를 보는 것이 아니라, 이 숫자 벡터들을 본다.

따라서 2장은 LLM에게 줄 “입력 재료”를 준비하는 장이다.

---

## 19. 2장 최종 요약

2장은 텍스트를 GPT 입력으로 바꾸는 전처리 장이다.

핵심 흐름은 다음과 같다.

```text
텍스트를 token으로 나눈다.
token을 token ID로 바꾼다.
BPE로 처음 보는 단어도 처리한다.
sliding window로 input/target 쌍을 만든다.
token ID를 embedding vector로 바꾼다.
position embedding을 더해 순서 정보를 넣는다.
token embedding과 position embedding을 더한 최종 input embedding을 attention에 넘긴다.
```

여기서 **최종 input embedding**은 원문 텍스트가 아니라, GPT가 실제로 계산할 수 있도록 만든 3차원 텐서다.

```text
input_embeddings = token_embeddings + position_embeddings
shape = (batch_size, context_length, emb_dim)
```

---

## 20. 2장 체크리스트

아래 질문에 답할 수 있으면 2장은 통과다.

| 질문 | 답할 수 있나 |
| --- | --- |
| token과 token ID의 차이는 무엇인가? |  |
| vocabulary는 왜 필요한가? |  |
| encode와 decode는 각각 무엇을 하는가? |  |
| BPE는 왜 `<unk>`를 줄일 수 있는가? |  |
| 한글에서 byte-level BPE가 중요한 이유는 무엇인가? |  |
| input과 target이 왜 한 칸 차이 나는가? |  |
| stride가 작거나 크면 어떤 차이가 생기는가? |  |
| token embedding과 position embedding은 각각 무슨 역할인가? |  |
| 최종 input embedding shape는 무엇인가? |  |
