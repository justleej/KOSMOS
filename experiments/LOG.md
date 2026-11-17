# 실험 기록

자동 생성 파일이다. 손으로 고치지 말고 각 실험의 `meta.json`을 고친 뒤 `python experiments/index.py`를 돌린다.
새 실험: `python experiments/new.py <이름> --author 이름 --question "..."`. 기각된 결과도 지우지 않는다.

| 날짜 | 실험 | 담당 | 질문 | 결론 | 상태 |
|---|---|---|---|---|---|
| 2026-11-15 | [20261115-baseline-calibration](20261115-baseline-calibration/) | 태우 | 현재 데이터에서 3모델의 β*와 validation TVD 기준선은? | β* 재현 일치. 보정 후 L0·L1 EXAONE 최저(0.243/0.248) | 완료 |
| 2026-11-15 | [20261115-baseline-calibration-k98](20261115-baseline-calibration-k98/) | 태우 | EXAONE k≈98로 늘린 DataFrame에서 3모델 β*와 validation TVD 기준선은 어떻게 바뀌는가? | EXAONE k 72→98에도 보정 후 TVD 변화 0.01 미만(L0 0.239, L1 0.246). DeepSeek·Haiku 재현 일치 | 완료 |
| 2026-11-15 | [20261115-k-curve](20261115-k-curve/) | 태우 | 모델별로 프롬프트당 표본 수 k를 늘리면 보정 후 validation TVD가 얼마나 줄어드는가? | k 50→90 감소 모두 0.01 미만(EXAONE L0 0.008). Haiku는 k=5부터 평평, 모델 순서 불변 | 완료 |
| 2026-11-17 | [20261117-gender-gap](20261117-gender-gap/) | justleej | L1에서 세 모델이 성별 차이를 구분하지 못하는 건 어떤 문항에서인가? 사람 성별 차이가 클 때는 잡는가? |  | 진행 |
