"""20261117-beta-stability — 실행: python experiments/20261117-beta-stability/run.py  (KOSMOS 루트에서, 1~3분)"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "research/design_v2_20261112/dataset"))
import silicon as sl  # noqa: E402

B, TOL, WIDTH_MAX, REGRET_MAX, ORDER_MIN = 1000, 0.005, 0.10, 0.005, 0.95
MODELS, LEVELS = ["exaone", "deepseek", "haiku"], sl.LEVELS
betas = sl.BETAS
rng = np.random.default_rng(20261117)

df = sl.load()                                          # test 49문항 제외
split_of = df.groupby("var")["split"].first()
cal_vars = sorted(split_of[split_of == "calibration"].index)
val_vars = sorted(split_of[split_of == "validation"].index)

# 1. (모델, 단계, 문항) × β 표: 페르소나 평균 TVD (level_mean과 같은 규칙)
curve = pd.concat([sl.score(sl.calibrate(df, b)).groupby(["model", "level", "var"])["tvd"].mean().rename(b)
                   for b in betas], axis=1)
cal = {(m, l): curve.loc[(m, l)].loc[cal_vars].to_numpy() for m in MODELS for l in LEVELS}   # [문항, β]
val = {k: curve.loc[k].loc[val_vars].to_numpy().mean(0) for k in cal}                       # [β]

# 2. sl.fit_beta와 같은 β*인지 확인
point = {k: int(v.mean(0).argmin()) for k, v in cal.items()}
ref = sl.fit_beta(df)
assert all(abs(betas[point[k]] - ref[k]) < 1e-9 for k in cal), (betas[list(point.values())], ref)

# 3. 문항 부트스트랩 (9개 모두 같은 추출 공유)
I = len(cal_vars)
idx = rng.integers(0, I, (B, I))
boot = {k: np.array([cal[k][i].mean(0).argmin() for i in idx]) for k in cal}

# 4. 표
ci_rows, regret_rows = [], []
for k in cal:
    m, l = k
    curve_mean = cal[k].mean(0)
    near = betas[curve_mean <= curve_mean.min() + TOL]
    b = betas[boot[k]]
    lo, hi = np.percentile(b, [2.5, 97.5])
    ci_rows.append(dict(model=m, level=l, beta_point=betas[point[k]], boot_median=np.median(b), ci_lo=lo, ci_hi=hi,
                        ci_width=hi - lo, near_opt_lo=near.min(), near_opt_hi=near.max(), near_opt_width=near.max() - near.min(),
                        edge_share=np.mean((b <= betas[0]) | (b >= betas[-1])), h1_ok=bool(hi - lo <= WIDTH_MAX)))
    reg = val[k][boot[k]] - val[k][point[k]]
    regret_rows.append(dict(model=m, level=l, regret_mean=reg.mean(), regret_p95=np.percentile(reg, 95), regret_max=reg.max(),
                            share_better=np.mean(reg < 0), h2_ok=bool(np.percentile(reg, 95) < REGRET_MAX),
                            val_best_beta=betas[val[k].argmin()], val_regret_of_point=val[k][point[k]] - val[k].min()))
beta_ci, regret = pd.DataFrame(ci_rows), pd.DataFrame(regret_rows)

order_rows = []
for l in LEVELS:
    for a, c in itertools.combinations(MODELS, 2):
        d0 = betas[point[(a, l)]] - betas[point[(c, l)]]
        d = betas[boot[(a, l)]] - betas[boot[(c, l)]]
        share = np.mean(np.sign(d) == np.sign(d0)) if d0 != 0 else np.nan
        order_rows.append(dict(level=l, a=a, b=c, point_a=betas[point[(a, l)]], point_b=betas[point[(c, l)]],
                               share_same_order=share, share_tie=np.mean(d == 0), h3_ok=bool(share >= ORDER_MIN)))
order = pd.DataFrame(order_rows)

draws = pd.DataFrame({f"{m}_{l}": betas[boot[(m, l)]] for m, l in cal})
tables = {"beta_ci": beta_ci, "val_regret": regret, "order": order, "draws": draws}
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
for name in ["beta_ci", "val_regret", "order"]:
    print(f"\n[{name}]\n{tables[name].round(4).to_string(index=False)}")

run = sl.save_run(HERE / "out", params={"B": B, "seed": 20261117, "tol": TOL, "width_max": WIDTH_MAX, "regret_max": REGRET_MAX,
                                        "order_min": ORDER_MIN, "n_cal": I, "n_val": len(val_vars)}, tables=tables)
print(f"\n기록: {HERE / 'out/run.json'} (commit {run['git_commit']}, dirty={run['git_dirty']})")
