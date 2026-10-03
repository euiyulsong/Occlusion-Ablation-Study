# Brand Occlusion Robustness 13-Way 실험 결과

## 1. 실험 개요

브랜드/logo 영역이 가려졌을 때 image retrieval 성능이 얼마나 저하되는지 확인하고, 다양한 occlusion augmentation 및 retrieval 전략의 robustness를 비교하였다.

총 13가지 방법을 비교하였다.

1. `baseline`
2. `random_erasing`
3. `cutout`
4. `gridmask`
5. `hide_and_seek`
6. `cutmix`
7. `logo_mask`
8. `mixed`
9. `part_based`
10. `combo_lc`
11. `combo_lcr`
12. `combo_logo_heavy`
13. `combo_clean`

신규 combination 설정은 다음과 같다.

| Regime | 구성 |
|---|---|
| `combo_lc` | Clean 50% + Logo Mask 30% + Cutout 20% |
| `combo_lcr` | Clean 35% + Logo Mask 30% + Cutout 20% + Random Erasing 15% + 일부 CutMix |
| `combo_logo_heavy` | Clean 30% + Logo Mask 40% + Cutout 20% + Random Erasing 10% |
| `combo_clean` | Clean 55% + Logo Mask 25% + Cutout 15% + Random Erasing 5% |

평가 조건은 다음과 같다.

- Clean
- Logo 25% occlusion
- Logo 50% occlusion
- Logo 100% occlusion
- Heavy occlusion

평가 지표:

- Recall@1
- Recall@5
- MRR@10
- Robust Drop = Clean R@1 - Logo 100% R@1

---

## 2. 전체 결과

| Regime | Clean R@1 | 25% R@1 | 50% R@1 | 100% R@1 | Heavy R@1 | Clean R@5 | 100% R@5 | Clean MRR@10 | 100% MRR@10 | Drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline** | **95.18%** | 43.41% | 27.65% | 25.40% | 23.79% | **98.07%** | 35.05% | **0.9633** | 0.2965 | 69.77%p |
| Random Erasing | 92.60% | 56.27% | 27.65% | 25.40% | 22.83% | 96.46% | 36.98% | 0.9393 | 0.2982 | 67.20%p |
| Cutout | 93.25% | 57.56% | 30.87% | 27.97% | 25.40% | 96.46% | 36.01% | 0.9469 | 0.3145 | 65.27%p |
| GridMask | 92.60% | 55.31% | 27.97% | 25.72% | 23.47% | 95.50% | 33.76% | 0.9365 | 0.2935 | 66.88%p |
| Hide-and-Seek | 88.42% | 56.59% | 26.37% | 24.44% | 23.47% | 92.28% | 32.48% | 0.9042 | 0.2808 | 63.99%p |
| CutMix | 91.96% | 49.20% | 27.97% | 26.37% | 23.15% | 94.53% | 35.37% | 0.9334 | 0.3037 | 65.59%p |
| **Logo Mask** | 90.03% | 45.98% | 40.19% | 37.62% | 33.44% | 95.82% | 46.62% | 0.9200 | 0.4136 | **52.41%p** |
| Mixed | 90.03% | 53.70% | 33.44% | 31.51% | 27.65% | 94.53% | 40.84% | 0.9188 | 0.3560 | 58.52%p |
| Part-based | 90.68% | 55.31% | 33.76% | 30.23% | 27.33% | 94.53% | 43.09% | 0.9238 | 0.3567 | 60.45%p |
| **Combo LC** | 94.21% | **58.84%** | **43.09%** | **39.55%** | **33.76%** | 96.78% | **48.87%** | 0.9530 | **0.4366** | 54.66%p |
| Combo LCR | 93.89% | 54.66% | 38.26% | 36.01% | 30.87% | 96.78% | 45.34% | 0.9496 | 0.4001 | 57.88%p |
| **Combo Logo Heavy** | 93.25% | 56.59% | 40.51% | 37.94% | **33.76%** | 96.14% | 48.55% | 0.9462 | 0.4274 | 55.31%p |
| Combo Clean | 93.57% | 54.66% | 38.91% | 36.98% | 30.55% | 97.11% | 46.30% | 0.9506 | 0.4126 | 56.59%p |

---

## 3. 핵심 결과

### Clean 이미지

Clean 조건에서는 여전히 **Baseline이 가장 높은 성능**을 기록했다.

- R@1: **95.18%**
- R@5: **98.07%**
- MRR@10: **0.9633**

하지만 신규 조합 중 `combo_lc`는:

- Clean R@1: **94.21%**
- Clean R@5: **96.78%**
- Clean MRR@10: **0.9530**

으로 Baseline 대비 Clean R@1이 단 **0.97%p 감소**했다.

기존 `logo_mask`가 Clean R@1 **90.03%**까지 떨어졌던 것과 비교하면, `combo_lc`는 clean 성능을 훨씬 잘 보존한다.

---

## 4. 약한 occlusion: 25%

신규 조합을 포함하면 **Combo LC가 가장 높은 R@1**을 기록했다.

| Regime | 25% R@1 |
|---|---:|
| **Combo LC** | **58.84%** |
| Cutout | 57.56% |
| Hide-and-Seek | 56.59% |
| Combo Logo Heavy | 56.59% |
| Random Erasing | 56.27% |
| GridMask | 55.31% |
| Part-based | 55.31% |
| Combo LCR | 54.66% |
| Combo Clean | 54.66% |
| Mixed | 53.70% |
| CutMix | 49.20% |
| Logo Mask | 45.98% |
| Baseline | 43.41% |

Baseline 대비:

**43.41% → 58.84%**

로 **+15.43%p 개선**되었다.

기존 최고였던 Cutout의 57.56%도 넘어섰다.

---

## 5. 중간 occlusion: Logo 50%

50% occlusion에서도 **Combo LC가 가장 높았다.**

| Regime | 50% R@1 |
|---|---:|
| **Combo LC** | **43.09%** |
| Combo Logo Heavy | 40.51% |
| Logo Mask | 40.19% |
| Combo Clean | 38.91% |
| Combo LCR | 38.26% |
| Part-based | 33.76% |
| Mixed | 33.44% |
| Cutout | 30.87% |
| GridMask | 27.97% |
| CutMix | 27.97% |
| Baseline | 27.65% |
| Random Erasing | 27.65% |
| Hide-and-Seek | 26.37% |

Baseline 대비:

**27.65% → 43.09%**

로 **+15.44%p 개선**되었다.

기존 최고인 Logo Mask보다도:

**40.19% → 43.09%**

로 **+2.90%p 높았다.**

---

## 6. 강한 occlusion: Logo 100%

Logo 100% occlusion에서도 **Combo LC가 가장 높은 R@1**을 기록했다.

| Regime | 100% R@1 |
|---|---:|
| **Combo LC** | **39.55%** |
| Combo Logo Heavy | 37.94% |
| Logo Mask | 37.62% |
| Combo Clean | 36.98% |
| Combo LCR | 36.01% |
| Mixed | 31.51% |
| Part-based | 30.23% |
| Cutout | 27.97% |
| CutMix | 26.37% |
| GridMask | 25.72% |
| Baseline | 25.40% |
| Random Erasing | 25.40% |
| Hide-and-Seek | 24.44% |

Baseline 대비:

**25.40% → 39.55%**

로 **+14.15%p 개선**되었다.

상대적으로는 약 **55.7% 높은 R@1**이다.

기존 최고였던 Logo Mask와 비교해도:

**37.62% → 39.55%**

로 **+1.93%p 개선**되었다.

---

## 7. Heavy Occlusion

Heavy 조건에서는 `combo_lc`와 `combo_logo_heavy`가 공동 최고 성능을 기록했다.

| Regime | Heavy R@1 |
|---|---:|
| **Combo LC** | **33.76%** |
| **Combo Logo Heavy** | **33.76%** |
| Logo Mask | 33.44% |
| Combo LCR | 30.87% |
| Combo Clean | 30.55% |
| Mixed | 27.65% |
| Part-based | 27.33% |
| Cutout | 25.40% |
| Baseline | 23.79% |
| GridMask | 23.47% |
| Hide-and-Seek | 23.47% |
| CutMix | 23.15% |
| Random Erasing | 22.83% |

Baseline 대비:

**23.79% → 33.76%**

로 **+9.97%p 개선**되었다.

---

## 8. Ranking 품질

Logo 100% occlusion에서 R@5와 MRR@10 역시 `combo_lc`가 가장 높은 결과를 기록했다.

| Regime | 100% R@5 | 100% MRR@10 |
|---|---:|---:|
| **Combo LC** | **48.87%** | **0.4366** |
| Combo Logo Heavy | 48.55% | 0.4274 |
| Logo Mask | 46.62% | 0.4136 |
| Combo Clean | 46.30% | 0.4126 |
| Combo LCR | 45.34% | 0.4001 |
| Part-based | 43.09% | 0.3567 |
| Mixed | 40.84% | 0.3560 |
| Random Erasing | 36.98% | 0.2982 |
| Cutout | 36.01% | 0.3145 |
| CutMix | 35.37% | 0.3037 |
| Baseline | 35.05% | 0.2965 |
| GridMask | 33.76% | 0.2935 |
| Hide-and-Seek | 32.48% | 0.2808 |

따라서 `combo_lc`의 개선은 Top-1 retrieval뿐 아니라 **Top-5와 전체 ranking 품질에서도 일관되게 나타났다.**

---

## 9. Robust Drop 비교

Clean R@1에서 Logo 100% R@1로 떨어지는 정도를 비교하였다.

| Regime | Robust Drop |
|---|---:|
| **Logo Mask** | **52.41%p** |
| Combo LC | 54.66%p |
| Combo Logo Heavy | 55.31%p |
| Combo Clean | 56.59%p |
| Combo LCR | 57.88%p |
| Mixed | 58.52%p |
| Part-based | 60.45%p |
| Hide-and-Seek | 63.99%p |
| Cutout | 65.27%p |
| CutMix | 65.59%p |
| GridMask | 66.88%p |
| Random Erasing | 67.20%p |
| Baseline | 69.77%p |

절대적인 Robust Drop만 보면 **Logo Mask가 가장 작다.**

하지만 Logo Mask의 Clean R@1은 **90.03%**이고, Combo LC는 **94.21%**이다.

즉:

- Logo Mask: Clean **90.03%** → Logo100 **37.62%**
- Combo LC: Clean **94.21%** → Logo100 **39.55%**

`combo_lc`는 Drop 자체는 2.25%p 더 크지만, **clean과 occluded 성능 모두 Logo Mask보다 높다.**

따라서 단순 Drop 수치만으로 최종 모델을 선택하기보다는 절대적인 Clean/Robust 성능을 함께 보는 것이 중요하다.

---

## 10. Combination 실험 비교

신규 조합만 비교하면 다음과 같다.

| Regime | Clean R@1 | 25% R@1 | 50% R@1 | 100% R@1 | Heavy R@1 |
|---|---:|---:|---:|---:|---:|
| **Combo LC** | **94.21%** | **58.84%** | **43.09%** | **39.55%** | **33.76%** |
| Combo LCR | 93.89% | 54.66% | 38.26% | 36.01% | 30.87% |
| Combo Logo Heavy | 93.25% | 56.59% | 40.51% | 37.94% | **33.76%** |
| Combo Clean | 93.57% | 54.66% | 38.91% | 36.98% | 30.55% |

가장 단순한 구성인:

**Clean 50% + Logo Mask 30% + Cutout 20%**

이 모든 주요 조건에서 가장 좋은 결과를 보였다.

Random Erasing과 CutMix를 추가한 `combo_lcr`은 오히려 성능이 감소했다.

또한 Logo Mask 비율을 40%까지 높인 `combo_logo_heavy`도 강한 occlusion에는 좋은 성능을 보였지만, `combo_lc`보다 전반적인 성능은 낮았다.

이는 augmentation 종류를 많이 추가하는 것 자체보다 **효과적인 augmentation의 종류와 비율을 적절히 조합하는 것이 중요함**을 보여준다.

---

## 11. 방법별 해석

### Baseline

- Clean 성능 최고
- Logo가 가려질 경우 큰 성능 저하
- Logo 및 logo 주변 visual cue에 크게 의존

### Random Erasing

- 25% 수준에서는 개선
- 강한 logo occlusion에는 효과 제한적

### Cutout

- 단독 augmentation 중 약한 occlusion에 매우 강함
- 25% R@1 57.56%
- Logo Mask와 결합했을 때 더 큰 효과를 보임

### GridMask / Hide-and-Seek / CutMix

- 약한 occlusion에서는 일부 개선
- 강한 logo-specific occlusion에는 효과 제한적
- 추가 augmentation이 항상 성능 향상으로 연결되지는 않음

### Logo Mask

- 강한 logo occlusion에 가장 직접적인 단일 augmentation
- Robust Drop이 가장 작음
- Clean 성능이 90.03%까지 하락하는 trade-off 존재

### Mixed

- 여러 augmentation을 단순 혼합
- 개별 augmentation보다 어느 정도 균형적
- 효과가 낮았던 Hide-and-Seek/CutMix 등이 섞이면서 최적 조합에는 미치지 못함

### Part-based

- 일부 ranking 성능 개선
- 단순 fixed crop 기반 part matching으로는 targeted augmentation보다 효과가 작음

### Combo LC

- **Clean 50% + Logo Mask 30% + Cutout 20%**
- 전체 실험에서 가장 균형 잡힌 결과
- Clean 성능 거의 유지
- 25%, 50%, 100% occlusion에서 모두 최고
- Heavy 조건에서도 공동 최고
- 100% R@5 및 MRR@10도 최고

### Combo LCR

- Random Erasing과 CutMix까지 추가
- Combo LC보다 모든 occlusion 조건에서 낮음
- augmentation을 추가한다고 반드시 robustness가 개선되는 것은 아님

### Combo Logo Heavy

- Logo Mask 비율을 40%로 증가
- Heavy R@1에서 공동 최고
- 그러나 Clean 및 중간 occlusion에서는 Combo LC보다 낮음

### Combo Clean

- Clean 비율을 55%까지 증가
- Clean R@1 93.57%로 높음
- 그러나 Combo LC보다 clean과 occlusion 성능 모두 낮음

---

## 12. 결론

초기 9-way 실험에서는 `logo_mask`가 강한 Logo Occlusion에 가장 효과적인 방법이었다.

그러나 추가 combination 실험 결과, **Logo Mask와 Cutout을 적절히 혼합하면서 충분한 Clean sample을 유지하는 방법이 더 좋은 결과**를 보였다.

### 최종적으로 가장 강한 조합

**Combo LC**

구성:

- Clean: **50%**
- Logo Mask: **30%**
- Cutout: **20%**

결과:

- Clean R@1: **94.21%**
- 25% R@1: **58.84%**
- 50% R@1: **43.09%**
- 100% R@1: **39.55%**
- Heavy R@1: **33.76%**
- 100% R@5: **48.87%**
- 100% MRR@10: **0.4366**

Baseline 대비:

| Metric | Baseline | Combo LC | 변화 |
|---|---:|---:|---:|
| Clean R@1 | **95.18%** | 94.21% | **-0.97%p** |
| 25% R@1 | 43.41% | **58.84%** | **+15.43%p** |
| 50% R@1 | 27.65% | **43.09%** | **+15.44%p** |
| 100% R@1 | 25.40% | **39.55%** | **+14.15%p** |
| Heavy R@1 | 23.79% | **33.76%** | **+9.97%p** |
| 100% R@5 | 35.05% | **48.87%** | **+13.82%p** |
| 100% MRR@10 | 0.2965 | **0.4366** | **+0.1401** |

즉 Clean R@1을 단 **0.97%p 희생**하면서 Logo 100% occlusion R@1을 **14.15%p 개선**하였다.

이번 실험에서는 단일 augmentation보다 **Clean + targeted Logo Mask + generic Cutout을 적절한 비율로 함께 학습하는 방식이 가장 효과적**이었다.

또한 Random Erasing, Hide-and-Seek, CutMix 등을 더 많이 섞는다고 추가적인 성능 향상이 발생하지 않았다.

따라서 현재 실험 조건에서 가장 실용적인 학습 전략은:

**50% Clean + 30% Logo-aware Masking + 20% Cutout**

으로 판단된다.

이는 clean image retrieval 성능을 대부분 유지하면서도 약한 occlusion부터 강한 logo occlusion까지 전반적으로 robustness를 개선하는 가장 좋은 trade-off를 보였다.
