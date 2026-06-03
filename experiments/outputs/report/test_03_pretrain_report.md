# Test 03 - Dropout 0.3 사전학습 실험 보고서

## 1. 실험 가설

Test 02에서는 step 제한을 해제하고 20 epoch까지 학습하면서 train loss와 validation loss가 모두 감소했다.
하지만 학습이 길어질수록 train loss가 validation loss보다 더 낮아지는 구간이 나타나 과적합 가능성을 의심할 수 있었다.

따라서 Test 03에서는 Test 02와 거의 같은 조건을 유지하되 `drop_rate`만 `0.1 -> 0.3`으로 높였다.
가설은 다음과 같다.

> Dropout을 강하게 적용하면 train loss 감소는 느려질 수 있지만, train/validation gap이 줄어 과적합 가능성이 낮아질 것이다.

## 2. 실험 목적

이번 실험의 목적은 충분한 epoch를 유지하면서도 과적합을 완화할 수 있는지 확인하는 것이다.
하이퍼파라미터를 한 번에 여러 개 바꾸지 않고 dropout만 변경하여, 결과 차이의 원인을 비교적 명확하게 해석할 수 있도록 했다.

## 3. 실험 조건

| 항목 | 값 |
| --- | --- |
| 실험 번호 | Test 03 |
| 변경한 하이퍼파라미터 | `drop_rate: 0.1 -> 0.3` |
| device | CUDA GPU |
| tokenizer | UTF-8 byte-level BPE |
| vocab size | 3,000 |
| train text size | 1,000,000 characters |
| validation text size | 100,000 characters |
| context length | 64 |
| embedding dim | 128 |
| attention heads | 4 |
| transformer layers | 2 |
| qkv bias | False |
| planned epochs | 20 |
| max steps | 0, 제한 없음 |
| 실제 global step | 11,400 |
| checkpoint resume | 사용하지 않음 |

## 4. 결과 요약

| 항목 | 시작 | 마지막 | 변화 |
| --- | ---: | ---: | ---: |
| train loss | 8.1503 at step 10 | 5.6488 at step 11400 | -2.5015 |
| validation loss | 7.7092 at step 50 | 5.6377 at step 11400 | -2.0714 |

Test 03에서도 train loss와 validation loss는 모두 감소했다.
따라서 dropout을 0.3으로 높여도 학습 자체는 정상적으로 진행되었다.

다만 Test 02와 비교하면 train loss는 더 높게 남았다.
이는 dropout이 학습 중 일부 뉴런 출력을 무작위로 끄기 때문에 모델이 train 데이터에 빠르게 맞춰지는 것을 억제한 결과로 볼 수 있다.

## 5. Loss Curve

![Test 03 Mini GPT pretraining loss curve](../test_03_loss_curve.png)

그래프 후반부를 보면 train loss와 validation loss가 Test 02보다 더 가까이 붙어 있다.
즉, dropout 증가가 train/validation gap을 줄이는 방향으로 작용했다.
하지만 validation loss 자체는 Test 02보다 낮아지지 않았기 때문에, 단순 성능 개선보다는 과적합 완화 효과가 더 두드러진 실험이다.

## 6. Test 02와 비교

| 항목 | Test 02 | Test 03 |
| --- | ---: | ---: |
| drop_rate | 0.1 | 0.3 |
| 실제 global step | 11,400 | 11,400 |
| final train loss | 5.3018 | 5.6488 |
| final validation loss | 5.5441 | 5.6377 |
| final gap `(val - train)` | +0.2424 | -0.0110 |

Test 03은 Test 02보다 train loss가 높다.
이는 예상한 결과다.
dropout이 강해지면 모델이 train 데이터에 과하게 적응하기 어려워지므로 train loss가 빠르게 낮아지지 않는다.

반면 final gap은 Test 02보다 크게 줄었다.
Test 02에서는 validation loss가 train loss보다 약 0.24 높았지만, Test 03에서는 두 값이 거의 비슷하다.
따라서 과적합 가능성은 Test 03에서 더 낮아진 것으로 해석할 수 있다.

하지만 validation loss는 Test 02보다 약간 높아졌다.
즉, dropout 0.3은 과적합 완화에는 효과가 있었지만, 현재 조건에서는 validation 성능을 개선하지는 못했다.

## 7. 생성 샘플 관찰

초기 샘플은 문장 구조가 거의 형성되지 않았다.

```text
[prompt] 이 영화는
이 영화는가,한 영화나을가어하고고영화도은게나나이이은고 고,...

[prompt] 정말
정말도다도.에지는의.게에아지.한를...
```

20 epoch 이후 샘플은 한글 자체는 비교적 안정적으로 나오고, 영화 리뷰 말투도 더 많이 나타났다.

```text
[prompt] 이 영화는
이 영화는 재밌네요
영화관에서 본 기억이 난 영화...

[prompt] 정말
정말 재밌던데...ㅠㅠ
정말 재미없었으면 진짜 재밌게봤습니다.
```

다만 의미 연결은 여전히 어색하다.
`재밌던데`, `재미없었으면`, `재밌게봤습니다`처럼 서로 충돌하는 표현이 한 문장 안에 섞인다.
따라서 모델은 리뷰 말투와 자주 나오는 표현은 학습했지만, 긴 문장의 의미 일관성은 아직 부족하다.

## 8. 가설 검증

가설은 절반 정도 맞았다.

- 맞은 부분: dropout을 0.3으로 높이자 train/validation gap이 크게 줄었다.
- 맞은 부분: train loss 감소는 Test 02보다 느려졌다.
- 틀리거나 애매한 부분: validation loss 자체는 Test 02보다 낮아지지 않았다.

따라서 Test 03은 "과적합 완화"에는 도움이 되었지만, "최종 validation 성능 개선"까지는 이어지지 않았다.
현재 모델에서는 `drop_rate=0.3`이 다소 강한 규제일 가능성이 있다.

## 9. 결론

Dropout을 0.3으로 높인 결과, 모델이 train 데이터에 과하게 맞춰지는 현상은 줄어들었다.
하지만 validation loss가 Test 02보다 약간 높게 남았으므로, 현재 조건에서 dropout 0.3은 최적값이라고 보기 어렵다.

다음 실험에서는 `drop_rate=0.2`처럼 중간값을 테스트하는 것이 가장 자연스럽다.
또는 dropout은 0.1로 되돌리고 `weight_decay`만 조정하여 regularization 효과를 분리해서 비교할 수 있다.

## 10. 다음 실험 제안

| 실험 | 가설 |
| --- | --- |
| Test 04: drop_rate 0.2 | 0.3보다 학습 성능 손실은 줄이고, 0.1보다 과적합은 완화할 수 있다. |
| Test 05: weight_decay 0.03 | dropout 대신 가중치 크기 규제를 강화하면 validation loss가 안정될 수 있다. |
| Test 06: learning rate 1e-4 | 후반부 train/val gap이 천천히 벌어지는 현상을 완화할 수 있다. |
| Test 07: context_length 128 | 더 긴 문맥을 보게 하면 생성 문장의 연결성이 좋아질 수 있다. |

