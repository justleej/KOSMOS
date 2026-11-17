# 실험 기록

자동 생성 파일이다. 손으로 고치지 말고 각 실험의 `meta.json`을 고친 뒤 `python experiments/index.py`를 돌린다.
새 실험: `python experiments/new.py <이름> --author 이름 --question "..."`. 기각된 결과도 지우지 않는다.

| 날짜 | 실험 | 담당 | 질문 | 결론 | 상태 |
|---|---|---|---|---|---|
| 2026-11-15 | [20261115-baseline-calibration](20261115-baseline-calibration/) | 태우 | 현재 데이터에서 3모델의 β*와 validation TVD 기준선은? | β* 재현 일치. 보정 후 L0·L1 EXAONE 최저(0.243/0.248) | 완료 |
| 2026-11-15 | [20261115-baseline-calibration-k98](20261115-baseline-calibration-k98/) | 태우 | EXAONE k≈98로 늘린 DataFrame에서 3모델 β*와 validation TVD 기준선은 어떻게 바뀌는가? | EXAONE k 72→98에도 보정 후 TVD 변화 0.01 미만(L0 0.239, L1 0.246). DeepSeek·Haiku 재현 일치 | 완료 |
| 2026-11-15 | [20261115-k-curve](20261115-k-curve/) | 태우 | 모델별로 프롬프트당 표본 수 k를 늘리면 보정 후 validation TVD가 얼마나 줄어드는가? | k 50→90 감소 모두 0.01 미만(EXAONE L0 0.008). Haiku는 k=5부터 평평, 모델 순서 불변 | 완료 |
| 2026-11-17 | [20261117-beta-stability](20261117-beta-stability/) | justleej | β*가 문항을 바꿔 뽑았을 때 얼마나 흔들리고, 흔들린 β*를 쓰면 validation TVD가 실제로 얼마나 나빠지는가? | β* 95% 구간 폭 0.05~0.095(9개 모두 기준 이내). validation 후회는 7/9가 0.005 미만, 초과는 c0의 EXAONE·DeepSeek. EXAONE vs DeepSeek 순서는 c0·L0에서 불안정 | 완료 |
