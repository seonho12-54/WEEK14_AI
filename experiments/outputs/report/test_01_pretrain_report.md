# Test 01 - Mini GPT 사전학습 실험 보고서

## 1. 실험 목적

이번 실험은 Test 01로, 구현한 mini GPT 모델이 NSMC 영화 리뷰 텍스트를 이용해 다음 토큰 예측을 실제로 학습하는지 확인하기 위한 사전학습 실험이다.
단순 테스트 통과를 넘어서, train loss와 validation loss가 함께 감소하는지 확인하고 생성 샘플의 변화도 관찰했다.

## 2. 모델 구조

| 항목 | 값 |
| --- | --- |
| 모델 | mini GPT |
| Tokenizer | UTF-8 byte-level BPE |
| vocab size | 3,000 |
| context length | 64 |
| embedding dim | 128 |
| attention heads | 4 |
| transformer layers | 2 |
| parameters | 1,172,224 |

## 3. 학습 설정

| 항목 | 값 |
| --- | --- |
| device | CUDA GPU |
| train text size | 1,000,000 characters |
| validation text size | 100,000 characters |
| train tokens | 583,778 |
| validation tokens | 58,388 |
| train batches per epoch | 570 |
| validation batches | 57 |
| planned epochs | 20 |
| max steps | 5,000 |
| 실제 진행 | epoch 9 중 global step 5,000까지 |
| 기록 주기 | train loss 10 step마다, validation loss 50 step마다 |

## 4. 실험 결과

| 항목 | 시작 | 마지막 | 변화 |
| --- | ---: | ---: | ---: |
| train loss | 8.1383 at step 10 | 5.6312 at step 5000 | -2.5071 |
| validation loss | 7.6062 at step 50 | 5.7082 at step 5000 | -1.8980 |

train loss와 validation loss가 모두 꾸준히 감소했다.
특히 1,000 step 이후부터 validation loss도 안정적으로 내려가며, 모델이 단순히 train batch만 외우는 것이 아니라 validation 데이터에 대해서도 다음 토큰 예측 성능을 개선하고 있음을 확인했다.

다만 step 5,000 기준으로도 생성 샘플은 아직 자연스럽지 않다.
이는 모델 크기와 학습량이 작은 편이고, byte-level BPE 특성상 충분히 학습되지 않은 상태에서는 UTF-8로 자연스럽게 해석되지 않는 조합이 생성될 수 있기 때문이다.

## 5. Loss Curve

아래 그래프는 붙여넣은 테스트 로그에서 train loss 500개, validation loss 100개를 다시 파싱해 생성했다.

![Test 01 Mini GPT pretraining loss curve](test_01_loss_curve.png)

## 6. 생성 샘플 관찰

초기 epoch에서는 문장 구조가 거의 형성되지 않았고, 깨진 문자 또는 의미 없는 조합이 많이 나타났다.
epoch가 진행될수록 `영화`, `정말`, `재미`, `연기`처럼 영화 리뷰 데이터에서 자주 등장하는 표현의 패턴이 조금씩 보이지만, 아직 문장으로 읽을 만큼 안정적인 수준은 아니다.

현재 결과는 "언어 생성이 완성되었다"기보다는 "loss가 감소하고 리뷰 말투의 일부 패턴을 학습하기 시작했다"로 해석하는 것이 적절하다.

## 7. 고찰

- loss 감소 추이로 보아 forward, loss 계산, backward, optimizer update 흐름은 정상적으로 동작한다.
- validation loss도 함께 감소하므로 현재 구간에서는 과적합보다는 학습 진행 중으로 보는 것이 타당하다.
- train loss가 validation loss보다 약간 낮아지는 구간이 생기지만, gap이 급격히 벌어지지는 않았다.
- 생성 샘플의 깨짐은 tokenizer decode 문제라기보다, 아직 모델이 유효한 byte/token 조합을 안정적으로 생성하지 못하는 문제로 보인다.
- 더 자연스러운 결과를 보려면 학습 step 증가, 모델 크기 확대, temperature 조정, top-k 조정 실험이 필요하다.

## 8. 다음 실험 제안

| 실험 | 목적 |
| --- | --- |
| max_steps 10,000 이상 | loss가 더 내려가는지 확인 |
| emb_dim 192 또는 256 | 모델 표현력 증가 효과 확인 |
| n_layers 4 | transformer block 깊이 증가 효과 확인 |
| temperature 0.7 / 0.5 | 생성 샘플의 깨짐과 무작위성 감소 확인 |
| top_k 20 / 10 | 후보 토큰 제한이 생성 품질에 미치는 영향 확인 |
