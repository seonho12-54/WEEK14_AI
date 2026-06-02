# LLM 핵심 키워드 체크리스트

## 0. 사용 방법

아래 키워드는 LLM 과제를 설명하거나 면접에서 질문받았을 때 반드시 말로 풀 수 있어야 하는 핵심 개념만 추린 것이다.

추천 공부 방식:

```text
1. 키워드 정의를 1~2줄로 직접 쓴다.
2. 우리 코드의 어느 파일에 나왔는지 적는다.
3. shape가 있는 개념은 shape까지 외운다.
4. "왜 필요한가?"를 한 문장으로 답해본다.
```

---

## 1. LLM 기본

- LLM
- Large Language Model
- 언어 모델
- 다음 token 예측
- 사전 학습
- 미세튜닝
- 추론
- 생성
- vocabulary
- token
- token id
- logits
- loss
- parameter
- gradient
- optimizer
- checkpoint

---

## 2. Tokenizer / BPE

- tokenizer
- encode
- decode
- BPE
- Byte Pair Encoding
- corpus
- vocabulary
- token_to_id
- id_to_token
- special token
- `<pad>`
- `<unk>`
- `<bos>`
- `<eos>`
- UTF-8
- byte-level tokenizer
- byte offset
- merge rule
- token pair
- pair frequency
- vocab_size
- OOV
- subword
- save / load

---

## 3. Dataset / DataLoader

- GPTDataset
- Dataset
- DataLoader
- input ids
- target ids
- input/target shift
- sliding window
- context_length
- stride
- batch
- batch_size
- num_batches
- LongTensor
- `torch.long`
- window sample

---

## 4. Embedding

- embedding
- token embedding
- position embedding
- InputEmbedding
- emb_dim
- d_model
- `nn.Embedding`
- embedding table
- position id
- dropout
- input embedding
- token id to vector
- 순서 정보

---

## 5. Attention

- attention
- self-attention
- causal attention
- multi-head attention
- Query
- Key
- Value
- Q/K/V projection
- `nn.Linear`
- matrix multiplication
- dot product
- `Q @ K^T`
- attention score
- attention weight
- softmax
- causal mask
- future token masking
- `-inf`
- head
- n_heads
- head_dim
- `sqrt(head_dim)`
- context vector
- output projection

---

## 6. TransformerBlock

- TransformerBlock
- LayerNorm
- FeedForward
- FFN
- residual connection
- Pre-LayerNorm
- normalized_shape
- eps
- mean
- variance
- gamma
- beta
- GELU
- MLP
- `nn.Sequential`
- dropout
- `x + layer(x)`
- gradient flow
- `d_model -> 4*d_model -> d_model`

---

## 7. GPTModel

- GPTModel
- config
- vocab_size
- context_length
- emb_dim
- n_heads
- n_layers
- blocks
- final LayerNorm
- lm_head
- language modeling head
- forward
- targets
- logits
- `F.cross_entropy`
- reshape
- `reshape(-1)`
- `(B, T, vocab_size)`
- `(B*T, vocab_size)`

---

## 8. Training

- training loop
- pretraining
- calc_loss_batch
- calc_loss_loader
- train_model
- train_loader
- val_loader
- train loss
- validation loss
- epoch
- global_step
- eval_freq
- eval_iter
- ckpt_freq
- optimizer.zero_grad
- loss.backward
- optimizer.step
- model.train
- model.eval
- torch.no_grad
- device
- `.to(device)`
- loss.item
- overfitting
- generalization
- state_dict
- AdamW

---

## 9. Generation

- generate
- generate_text_simple
- greedy decoding
- sampling
- temperature
- top-k sampling
- eos_id
- max_new_tokens
- context_size
- idx
- idx_cond
- last logits
- `logits[:, -1, :]`
- `torch.argmax`
- `torch.multinomial`
- `torch.cat`
- `dim=-1`
- softmax sampling
- `<eos>` 종료

---

## 10. Fine-tuning

- fine-tuning
- sentiment classification
- NSMC
- TSV
- JSONL
- train data
- validation data
- test data
- val_ratio
- ReviewSentimentDataset
- max_length
- padding
- truncation
- GPTForSequenceClassification
- backbone
- classifier
- classification head
- LM head vs classifier head
- num_labels
- sentence vector
- last hidden state
- train_epoch_sentiment
- evaluate_sentiment
- accuracy

---

## 11. Shape 필수 암기

- token ids: `(B, T)`
- target ids: `(B, T)`
- embedding output: `(B, T, emb_dim)`
- Q/K/V before split: `(B, T, d_model)`
- Q/K/V after split: `(B, n_heads, T, head_dim)`
- attention scores: `(B, n_heads, T, T)`
- attention weights: `(B, n_heads, T, T)`
- merged context: `(B, T, d_model)`
- GPT logits: `(B, T, vocab_size)`
- flattened logits: `(B*T, vocab_size)`
- flattened targets: `(B*T,)`
- classification input ids: `(B, max_length)`
- sentence vector: `(B, emb_dim)`
- classification logits: `(B, num_labels)`
- labels: `(B,)`

---

## 12. 꼬리질문 

- LLM은 왜 다음 token 예측으로 학습하는가?
- tokenizer가 없으면 왜 모델 학습이 불가능한가?
- BPE와 단어 단위 tokenizer의 차이는?
- byte-level BPE가 한국어에 유리한 이유는?
- context_length가 너무 작으면 어떤 문제가 생기는가?
- position embedding이 없으면 어떤 문제가 생기는가?
- Q, K, V는 왜 따로 만드는가?
- attention score와 attention weight의 차이는?
- causal mask가 없으면 어떤 문제가 생기는가?
- multi-head attention이 필요한 이유는?
- LayerNorm과 BatchNorm의 차이는?
- residual connection이 깊은 모델에 중요한 이유는?
- lm_head와 classifier head는 무엇이 다른가?
- logits에 softmax를 직접 하지 않고 cross entropy를 쓰는 이유는?
- generation과 training은 무엇이 다른가?
- greedy decoding과 top-k sampling은 무엇이 다른가?
- validation loss가 필요한 이유는?
- checkpoint에 optimizer state까지 저장하는 이유는?
- fine-tuning과 pretraining은 무엇이 다른가?
- 마지막 token hidden state를 문장 대표 벡터로 쓰는 이유는?

---

## 13. 직접 써보기 필수 개념

```text
LLM:

Tokenizer:

BPE:

Context length:

Embedding:

Attention:

Causal mask:

Multi-head attention:

LayerNorm:

Residual connection:

FeedForward:

Logits:

Cross entropy:

Training:

Generation:

Fine-tuning:

Classifier:
```
