# Brand Occlusion Robustness 실험 결과

## 1. 실험 설정

- Dataset: **LogoDet-3K**
- 대상: Clothing brand subset
- 학습 방식: **Baseline**
  - 원본 이미지로만 학습
  - 별도 occlusion augmentation 없음
- Retrieval 평가:
  - Clean
  - Logo 25% occlusion
  - Logo 50% occlusion
  - Logo 100% occlusion
  - Heavy occlusion
- Metrics:
  - Recall@1
  - Recall@5
  - MRR@10
  - Robust Drop

---

## 2. Baseline 결과

| Regime | Clean R@1 | 25% R@1 | 50% R@1 | 100% R@1 | Heavy R@1 | Clean R@5 | 100% R@5 | Clean MRR@10 | 100% MRR@10 | Robust Drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | **94.21%** | 43.41% | 28.94% | 26.69% | 24.44% | **97.75%** | 38.26% | **0.9613** | 0.3154 | **67.52%p** |

---

## 3. 결과 분석

Baseline은 clean image에서는 **Recall@1 94.21%**, **Recall@5 97.75%**로 매우 높은 retrieval 성능을 보였다.

반면 logo 영역을 가리기 시작하면 성능이 크게 감소했다.

- 25% occlusion: **R@1 43.41%**
- 50% occlusion: **R@1 28.94%**
- 100% occlusion: **R@1 26.69%**
- Heavy occlusion: **R@1 24.44%**

특히 Clean R@1과 100% occlusion R@1의 차이는 **67.52%p**로 매우 크다.

즉, augmentation 없이 학습한 baseline 모델은 clean image에서는 높은 성능을 보이지만, **brand/logo 영역이 가려지면 retrieval 성능이 크게 저하되는 경향**이 확인된다.

MRR@10 역시:

- Clean: **0.9613**
- 100% occlusion: **0.3154**

로 크게 감소해, 단순히 Top-1만 떨어지는 것이 아니라 전체 ranking 품질도 상당히 악화되는 것을 확인할 수 있다.

---

## 4. 핵심 요약

| 항목 | 결과 |
|---|---:|
| Clean R@1 | **94.21%** |
| 100% Occlusion R@1 | **26.69%** |
| Heavy Occlusion R@1 | **24.44%** |
| Clean R@5 | **97.75%** |
| 100% Occlusion R@5 | **38.26%** |
| Clean MRR@10 | **0.9613** |
| 100% Occlusion MRR@10 | **0.3154** |
| Robust Drop | **67.52%p** |

Baseline은 clean 조건에서는 강하지만, **logo occlusion에 대한 robustness는 매우 낮은 편**이다.

따라서 이후 실험에서는 Random Erasing, CutMix, Hide-and-Seek, logo-aware masking, mixed augmentation, part-based retrieval이 이 Robust Drop을 얼마나 줄이는지 비교하는 것이 핵심이다.
