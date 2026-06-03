# Test 05 Report - Fine-tuning Data Size 50,000

## 1. 실험 목적

Test 05는 Test 04에서 만든 사전학습 checkpoint를 고정한 상태에서, 미세튜닝 학습 데이터 수를 늘렸을 때 감성 분류 정확도가 개선되는지 확인하기 위한 실험이다.

Test 04에서는 fine-tuning train data 20,000개로 10 epoch를 수행했고, 최종 test accuracy는 76.50%였다.

Test 05에서는 사전학습 조건과 optimizer 설정은 최대한 유지하고, fine-tuning train data만 50,000개로 늘렸다.

## 2. 실험 가설

학습 데이터 수를 20,000개에서 50,000개로 늘리면 모델이 더 다양한 리뷰 표현을 학습할 수 있으므로 validation/test accuracy가 상승할 것이다.

다만 데이터가 많아지면서 한 epoch에 걸리는 시간이 길어지고, 학습이 진행될수록 train/validation gap이 커질 가능성도 있다.

## 3. 실험 설정

| 항목 | 값 |
| --- | --- |
| 실험 번호 | Test 05 |
| 사전학습 checkpoint | `test_04_pretrain_experiment_last.pt` |
| fine-tuning checkpoint | `test_05_finetune_best.pt` |
| fine-tuning train samples | 50,000 |
| validation samples | 5,000 |
| test samples | 5,000 |
| optimizer | AdamW |
| learning rate | 1e-4 |
| weight decay | 0.01 |
| batch size | 32 |
| 계획 epoch | 10 |
| 실제 진행 epoch | 5 |

## 4. 학습 결과

| epoch | train loss | train acc | val loss | val acc | acc gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6838 | 58.23% | 0.5924 | 68.46% | -10.23% |
| 2 | 0.5785 | 69.37% | 0.5738 | 71.84% | -2.47% |
| 3 | 0.5343 | 72.99% | 0.4921 | 76.18% | -3.19% |
| 4 | 0.5022 | 75.11% | 0.4808 | 77.36% | -2.25% |
| 5 | 0.4838 | 76.78% | 0.4756 | 78.30% | -1.52% |

5 epoch 기준 best checkpoint는 epoch 5에서 저장되었다.

```text
best validation accuracy: 78.30%
best checkpoint: test_05_finetune_best.pt
```

![Test 05 fine-tuning curve](../test_05_finetune_5epoch_accuracy_curve.png)

## 5. Test Accuracy

저장된 `test_05_finetune_best.pt`를 사용해 test dataset 5,000개에 대해 평가했다.

| 항목 | 값 |
| --- | ---: |
| checkpoint epoch | 5 |
| test samples | 5,000 |
| test loss | 0.4486 |
| test accuracy | 79.44% |

Test 05의 최종 test accuracy는 79.44%로 측정되었다.

## 6. Test 04와 비교

| 항목 | Test 04 | Test 05 |
| --- | ---: | ---: |
| fine-tuning train samples | 20,000 | 50,000 |
| 진행 epoch | 10 | 5 |
| validation accuracy | 76.16% | 78.30% |
| test accuracy | 76.50% | 79.44% |

Test 05는 5 epoch까지만 진행했지만, Test 04의 10 epoch 결과보다 높은 validation accuracy와 test accuracy를 기록했다.

```text
test accuracy improvement: 76.50% -> 79.44%
improvement: +2.94%p
```

따라서 fine-tuning 데이터 수를 늘리면 더 적은 epoch에서도 성능이 개선될 수 있다는 가설을 뒷받침한다.

## 7. 결과 해석

Test 05는 epoch 1부터 validation accuracy가 train accuracy보다 높게 나타났다.

이는 validation 데이터가 상대적으로 더 쉬웠거나, dropout이 train mode에서만 적용되어 train accuracy가 낮게 측정되었기 때문일 수 있다.

epoch가 진행될수록 train accuracy와 validation accuracy의 차이는 줄어들었다.

```text
epoch 1 gap: -10.23%
epoch 5 gap: -1.52%
```

또한 validation loss는 5 epoch까지 계속 감소했다.

```text
val loss: 0.5924 -> 0.4756
val acc: 68.46% -> 78.30%
```

따라서 5 epoch 시점에서는 과적합보다는 아직 정상적으로 학습이 진행 중인 상태로 해석할 수 있다.

Test accuracy 역시 79.44%로 Test 04보다 높게 나왔으므로, validation 성능 향상이 test 성능 향상으로도 이어졌다고 볼 수 있다.

## 8. 한계

이번 Test 05는 시간이 오래 걸려 10 epoch 전체를 수행하지 못하고 5 epoch에서 중단했다.

하지만 5 epoch까지 validation loss가 계속 감소했고 validation accuracy도 상승했기 때문에, 추가 epoch를 진행하면 더 개선될 가능성은 남아 있다.

다만 미세튜닝 시간이 길고 Colab 연결이 끊길 수 있으므로, 이후 실험에서는 epoch별 checkpoint를 Google Drive에 저장하는 방식이 필요하다.

## 9. 결론

Test 05는 Test 04 사전학습 checkpoint를 고정하고 fine-tuning 데이터 수만 50,000개로 늘린 실험이다.

5 epoch까지만 진행했음에도 validation accuracy 78.30%, test accuracy 79.44%를 기록했다.

이는 Test 04의 test accuracy 76.50%보다 2.94%p 높은 결과다.

따라서 현재 실험에서는 learning rate 조정보다 fine-tuning 데이터량 증가가 감성 분류 정답률 개선에 더 직접적인 효과를 보였다.

최종적으로 Test 05는 사전학습 checkpoint를 고정한 상태에서 fine-tuning 데이터량을 늘리는 전략이 유효하다는 근거를 제공한다.
