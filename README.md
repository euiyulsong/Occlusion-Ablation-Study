# Brand Occlusion Robustness 9-Way 실험 결과

## 1. 실험 개요

브랜드/logo 영역이 가려졌을 때 image retrieval 성능이 얼마나 저하되는지 확인하고, 다양한 occlusion augmentation 및 retrieval 전략의 robustness를 비교하였다.

비교한 방법은 다음 9가지이다.

1. `baseline`
2. `random_erasing`
3. `cutout`
4. `gridmask`
5. `hide_and_seek`
6. `cutmix`
7. `logo_mask`
8. `mixed`
9. `part_based`

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
| **Cutout** | 93.25% | **57.56%** | 30.87% | 27.97% | 25.40% | 96.46% | 36.01% | 0.9469 | 0.3145 | 65.27%p |
| GridMask | 92.60% | 55.31% | 27.97% | 25.72% | 23.47% | 95.50% | 33.76% | 0.9365 | 0.2935 | 66.88%p |
| Hide-and-Seek | 88.42% | 56.59% | 26.37% | 24.44% | 23.47% | 92.28% | 32.48% | 0.9042 | 0.2808 | 63.99%p |
| CutMix | 91.96% | 49.20% | 27.97% | 26.37% | 23.15% | 94.53% | 35.37% | 0.9334 | 0.3037 | 65.59%p |
| **Logo Mask** | 90.03% | 45.98% | **40.19%** | **37.62%** | **33.44%** | 95.82% | **46.62%** | 0.9200 | **0.4136** | **52.41%p** |
| Mixed | 90.03% | 53.70% | 33.44% | 31.51% | 27.65% | 94.53% | 40.84% | 0.9188 | 0.3560 | 58.52%p |
| Part-based | 90.68% | 55.31% | 33.76% | 30.23% | 27.33% | 94.53% | 43.09% | 0.9238 | 0.3567 | 60.45%p |

---

## 3. 핵심 결과

### Clean 이미지

Clean 조건에서는 **Baseline이 가장 높았다.**

- R@1: **95.18%**
- R@5: **98.07%**
- MRR@10: **0.9633**

Occlusion augmentation을 추가하면 대부분 clean 성능은 소폭 감소했다.

특히 `logo_mask`는:

- Baseline: 95.18%
- Logo Mask: 90.03%

으로 약 **5.15%p의 clean R@1 감소**가 발생했다.

---

## 4. 약한 occlusion: 25%

25% occlusion에서는 **Cutout이 가장 높은 R@1**을 기록했다.

| Regime | 25% R@1 |
|---|---:|
| Cutout | **57.56%** |
| Hide-and-Seek | 56.59% |
| Random Erasing | 56.27% |
| GridMask | 55.31% |
| Part-based | 55.31% |
| Mixed | 53.70% |
| CutMix | 49.20% |
| Logo Mask | 45.98% |
| Baseline | 43.41% |

즉 비교적 약한 부분 가림에서는 **일반적인 random occlusion augmentation도 효과가 크다.**

Baseline 대비 Cutout은:

**43.41% → 57.56%**

로 **+14.15%p** 개선되었다.

---

## 5. 중간 이상 occlusion

Occlusion이 강해질수록 결과가 크게 달라졌다.

### Logo 50%

| Regime | R@1 |
|---|---:|
| **Logo Mask** | **40.19%** |
| Part-based | 33.76% |
| Mixed | 33.44% |
| Cutout | 30.87% |
| GridMask | 27.97% |
| CutMix | 27.97% |
| Baseline | 27.65% |
| Random Erasing | 27.65% |
| Hide-and-Seek | 26.37% |

`logo_mask`가 baseline보다 **+12.54%p** 높았다.

---

### Logo 100%

| Regime | R@1 |
|---|---:|
| **Logo Mask** | **37.62%** |
| Mixed | 31.51% |
| Part-based | 30.23% |
| Cutout | 27.97% |
| CutMix | 26.37% |
| GridMask | 25.72% |
| Baseline | 25.40% |
| Random Erasing | 25.40% |
| Hide-and-Seek | 24.44% |

여기서는 `logo_mask`의 효과가 매우 명확하다.

Baseline:

**25.40%**

Logo Mask:

**37.62%**

즉 **+12.22%p**, 상대적으로 약 **48% 높은 R@1**을 기록했다.

---

### Heavy Occlusion

Heavy 조건에서도 `logo_mask`가 가장 높았다.

| Regime | Heavy R@1 |
|---|---:|
| **Logo Mask** | **33.44%** |
| Mixed | 27.65% |
| Part-based | 27.33% |
| Cutout | 25.40% |
| Baseline | 23.79% |
| GridMask | 23.47% |
| Hide-and-Seek | 23.47% |
| CutMix | 23.15% |
| Random Erasing | 22.83% |

Baseline 대비:

**23.79% → 33.44%**

로 **+9.65%p** 개선되었다.

---

## 6. Ranking 품질에서도 Logo Mask가 가장 강함

Logo 100% occlusion에서 R@5와 MRR@10 역시 `logo_mask`가 가장 높았다.

| Regime | 100% R@5 | 100% MRR@10 |
|---|---:|---:|
| **Logo Mask** | **46.62%** | **0.4136** |
| Part-based | 43.09% | 0.3567 |
| Mixed | 40.84% | 0.3560 |
| Random Erasing | 36.98% | 0.2982 |
| Cutout | 36.01% | 0.3145 |
| CutMix | 35.37% | 0.3037 |
| Baseline | 35.05% | 0.2965 |
| GridMask | 33.76% | 0.2935 |
| Hide-and-Seek | 32.48% | 0.2808 |

따라서 `logo_mask`의 개선은 단순 Top-1 hit 증가뿐 아니라 **전체 ranking 품질에도 나타났다.**

---

## 7. Robust Drop 비교

Clean R@1에서 Logo 100% R@1로 떨어지는 정도를 비교하였다.

| Regime | Robust Drop |
|---|---:|
| **Logo Mask** | **52.41%p** |
| Mixed | 58.52%p |
| Part-based | 60.45%p |
| Hide-and-Seek | 63.99%p |
| Cutout | 65.27%p |
| CutMix | 65.59%p |
| GridMask | 66.88%p |
| Random Erasing | 67.20%p |
| Baseline | **69.77%p** |

Baseline은 가장 큰 성능 하락을 보였다.

`logo_mask`는:

**69.77%p → 52.41%p**

로 Robust Drop을 **17.36%p 감소**시켰다.

---

## 8. 방법별 해석

### Baseline

- Clean 성능은 가장 높음
- Logo가 가려지면 급격하게 성능 저하
- 모델이 logo/logo 주변 visual cue에 크게 의존하는 것으로 해석 가능

### Random Erasing

- 25% 수준의 가벼운 occlusion에는 효과적
- 50~100% logo occlusion에서는 baseline과 거의 차이가 없음
- random한 위치를 가리는 것만으로는 logo-specific robustness 확보가 어려움

### Cutout

- 25% occlusion에서 전체 최고
- 약하거나 중간 정도의 occlusion에는 효과가 있음
- 강한 logo-specific occlusion에서는 개선폭이 제한적

### GridMask / Hide-and-Seek / CutMix

- 일부 약한 occlusion에서는 baseline보다 개선
- 강한 logo occlusion에서는 큰 효과를 보이지 못함
- 특히 Hide-and-Seek은 clean 성능 감소폭이 상대적으로 큼

### Logo Mask

- Clean 성능은 일부 감소
- Logo 50%, 100%, Heavy 조건에서 모두 가장 높은 성능
- Logo occlusion robustness에 가장 직접적인 효과를 보임

### Mixed

- Logo Mask보다 clean 성능은 동일
- 25% occlusion에서는 Logo Mask보다 높음
- 강한 occlusion에서는 Logo Mask보다는 낮음
- 여러 유형의 occlusion을 고려할 때 균형적인 결과

### Part-based

- 25% occlusion에서 강한 성능
- 100% R@5가 Logo Mask 다음으로 높음
- 현재 단순 fixed crop 기반 part retrieval에서는 Logo Mask를 넘지 못함
- 향후 object/semantic part crop을 사용하면 추가 개선 가능성이 있음

---

## 9. 결론

실험 결과, occlusion 유형에 따라 효과적인 방법이 달랐다.

### 약한 가림

**Cutout**

- 25% R@1: **57.56%**
- Baseline 대비 **+14.15%p**

일반적인 작은 occlusion에 대한 robustness를 높이는 데 효과적이었다.

### 강한 Logo Occlusion

**Logo-aware Masking**

- 50% R@1: **40.19%**
- 100% R@1: **37.62%**
- Heavy R@1: **33.44%**
- 100% R@5: **46.62%**
- 100% MRR@10: **0.4136**
- Robust Drop: **52.41%p**

강한 브랜드/logo 가림 문제에서는 모든 지표에서 가장 안정적인 결과를 보였다.

### 종합

이번 실험에서는 **generic occlusion augmentation만 사용하는 것보다, 실제 logo 위치를 이용한 targeted masking이 강한 logo occlusion에 훨씬 효과적이었다.**

특히 Baseline의 Logo 100% R@1이 **25.40%**였던 것에 비해 Logo Mask는 **37.62%**로 증가하였다.

따라서 실제 서비스에서 문제의 핵심이 **브랜드/logo가 손, 물체, crop 등에 의해 가려지는 경우**라면, 현재 결과 기준으로는 **Logo-aware masking을 우선 적용하는 것이 가장 유효한 전략**이다.

다만 Clean R@1은 **95.18% → 90.03%**로 감소했으므로, production에서는 clean/logo-masked image 비율과 masking probability를 추가로 튜닝하여 clean 성능과 robustness 사이의 trade-off를 조정하는 것이 필요하다.
