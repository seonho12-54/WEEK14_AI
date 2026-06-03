# Test 02 - Mini GPT 사전학습 실험 보고서

## 1. 실험 가설

Test 01에서는 같은 모델과 데이터 조건에서 `max_steps=5000` 제한에 먼저 도달해 epoch 20까지 학습하지 못했다.
따라서 Test 02에서는 step 제한을 해제하고 20 epoch 전체를 학습하면, train loss와 validation loss가 더 감소하고 생성 문장의 한글 깨짐과 어색함도 줄어들 것이라고 가정했다.

## 2. 실험 목적

이번 실험은 모델 구조나 데이터 크기를 바꾸지 않고, 학습량만 늘렸을 때 mini GPT의 사전학습 성능이 얼마나 개선되는지 확인하는 것이 목적이다.
즉, Test 01의 한계가 모델 구조 때문인지, 단순히 학습량 부족 때문인지 1차적으로 판단하기 위한 비교 실험이다.

## 3. 실험 조건

| 항목 | 값 |
| --- | --- |
| 실험 번호 | Test 02 |
| device | CUDA GPU |
| tokenizer | UTF-8 byte-level BPE |
| vocab size | 3,000 |
| train text size | 1,000,000 characters |
| validation text size | 100,000 characters |
| context length | 64 |
| embedding dim | 128 |
| attention heads | 4 |
| transformer layers | 2 |
| drop rate | 0.1 |
| qkv bias | False |
| planned epochs | 20 |
| max steps | 0, 제한 없음 |
| 실제 global step | 11,400 |
| checkpoint resume | 사용하지 않음 |

## 4. 결과 요약

| 항목 | 시작 | 마지막 | 변화 |
| --- | ---: | ---: | ---: |
| train loss | 8.1383 at step 10 | 5.3018 at step 11400 | -2.8366 |
| validation loss | 7.6062 at step 50 | 5.5441 at step 11400 | -2.0621 |

Test 02에서는 20 epoch 전체가 진행되며 global step 11,400까지 학습되었다.
train loss와 validation loss가 모두 감소했으므로, 모델은 추가 학습을 통해 다음 토큰 예측 성능을 계속 개선했다.

## 5. Loss Curve

![Test 02 Mini GPT pretraining loss curve](test_02_loss_curve.png)

그래프를 보면 초반에는 loss가 빠르게 감소하고, 이후에는 감소 속도가 완만해진다.
이는 모델이 초반에 자주 등장하는 byte/token 패턴과 리뷰 말투의 기본 분포를 빠르게 학습하고, 뒤로 갈수록 더 어려운 문맥 패턴을 천천히 학습하는 것으로 해석할 수 있다.

## 6. Test 01과 비교

| 항목 | Test 01 | Test 02 |
| --- | ---: | ---: |
| max steps | 5,000 | 제한 없음 |
| 실제 global step | 5,000 | 11,400 |
| train loss 마지막 | 5.6312 | 5.3018 |
| validation loss 마지막 | 5.7082 | 5.5441 |
| 진행 범위 | epoch 9 중단 | epoch 20 완료 |

Test 02는 Test 01보다 약 2.3배 더 많은 step을 학습했다.
그 결과 train loss는 약 0.33, validation loss는 약 0.16 더 낮아졌다.
따라서 학습량 증가가 성능 개선에 실제로 도움이 되었음을 확인했다.

다만 validation loss 감소폭은 train loss 감소폭보다 작다.
이는 모델이 train 데이터에 더 잘 맞춰지는 방향으로 학습되고 있으며, 앞으로 학습을 더 오래 하면 train/validation gap을 함께 관찰해야 함을 의미한다.

## 7. 가설 검증

가설은 대체로 맞았다.
step 제한을 해제하고 20 epoch까지 학습하자 train loss와 validation loss가 모두 Test 01보다 더 낮아졌다.
따라서 Test 01의 부족한 생성 품질은 어느 정도 학습량 부족의 영향이 있었다고 볼 수 있다.

하지만 validation loss가 5.5대에서 완만하게 감소하고 있으며, 그래프 후반부에서 train/validation gap이 조금 벌어진다.
따라서 단순히 epoch만 더 늘리는 방식으로는 개선 폭이 점점 줄어들 가능성이 있다.
다음 실험에서는 모델 크기, context length, decoding 설정을 함께 조정할 필요가 있다.

## 8. 고찰

- 20 epoch 전체 학습은 정상적으로 완료되었다.
- loss 곡선은 안정적으로 감소했고 급격한 발산이나 NaN은 나타나지 않았다.
- validation loss도 감소했으므로 아직 완전한 과적합이라고 보기는 어렵다.
- 후반부에는 validation loss 감소가 매우 느려져, 추가 학습만으로 큰 개선을 기대하기는 어렵다.
- 생성 품질이 여전히 부족하다면 학습량보다 모델 표현력이나 생성 전략의 영향도 고려해야 한다.

## 9. 다음 실험 제안

| 실험 | 가설 |
| --- | --- |
| Test 03: emb_dim 192 또는 256 | 모델 표현력을 키우면 validation loss와 생성 품질이 개선될 것이다. |
| Test 04: n_layers 4 | 더 깊은 Transformer block이 문맥 표현력을 높일 것이다. |
| Test 05: context_length 128 | 더 긴 문맥을 보면 생성 문장의 연결성이 나아질 것이다. |
| Test 06: temperature/top_k 조정 | 학습된 모델은 같아도 decoding 전략을 바꾸면 생성 문장의 깨짐과 무작위성이 줄어들 것이다. |

