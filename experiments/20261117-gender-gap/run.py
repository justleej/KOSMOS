"""20261117-gender-gap — 실행: python experiments/20261117-gender-gap/run.py  (KOSMOS 루트에서, 수십 초)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "research/design_v2_20261112/dataset"))
import silicon as sl  # noqa: E402

B, NB, MIN_PAIRS, EPS = 1000, 2000, 30, 1e-9      # 사람 재추출, 문항 부트스트랩, 층 최소 쌍 수, 동점 허용 오차
AGES = ["18-29", "30-39", "40-49", "50-59", "60+"]
MODELS = ["exaone", "deepseek", "haiku"]
rng = np.random.default_rng(20261117)

df = sl.load()                                     # test 49문항 제외
df = df[(df["level"] == "L1") & df["scale_type"].isin(["ordinal_monotone", "binary"])].copy()
df["code"] = pd.to_numeric(df["code"])
df = df.sort_values(["model", "persona", "var", "opt_idx"])

# ── 평균 점수: 모델은 보정 없는 원비율, 사람은 human_p ─────────────────────
df["rs"] = df["count"] / df["n_valid"].clip(lower=1) * df["code"]
df["hp"] = df["human_p"] / df.groupby(["model", "persona", "var"])["human_p"].transform("sum")
df["hs"] = df["hp"] * df["code"]
ms = df.groupby(["persona", "var", "model"])["rs"].sum().unstack("model")[MODELS]
hs = df.groupby(["persona", "var", "model"])["hs"].sum().unstack("model")
assert np.allclose(hs.max(1), hs.min(1)), "사람 분포가 모델마다 다르다"
hs = hs["exaone"]

# ── 사람 점수의 다항 재추출(유효 표본 크기 기준) ─────────────────────────
hb = {}
for (p, v), g in df[df["model"] == "exaone"].groupby(["persona", "var"]):
    neff = max(int(round(g["human_neff"].iloc[0])), 1)
    hb[(p, v)] = rng.multinomial(neff, g["hp"].to_numpy(), size=B) / neff @ g["code"].to_numpy()

meta = df[df["model"] == "exaone"].groupby("var")[["topic", "sensitivity_level", "split", "scale_type"]].first()

# ── 쌍 만들기 ──────────────────────────────────────────────────────────
def cell(sex, age):
    return f"L1_{sex}_{age}"


pairs = []
for v in meta.index:
    combos = [("gender", cell(2, a), cell(1, a)) for a in AGES]                     # 여 − 남 (같은 연령대)
    for sex in (1, 2):                                                                # 나이 많은 쪽 − 적은 쪽 (같은 성별)
        combos += [("age", cell(sex, AGES[j]), cell(sex, AGES[i])) for i in range(5) for j in range(i + 1, 5)]
    for kind, a, b in combos:
        lo, hi = np.percentile(hb[(a, v)] - hb[(b, v)], [2.5, 97.5])
        row = dict(var=v, kind=kind, a=a, b=b, dh=hs[(a, v)] - hs[(b, v)], sig=bool(lo > 0 or hi < 0))
        row.update({f"dm_{m}": ms.loc[(a, v), m] - ms.loc[(b, v), m] for m in MODELS})
        pairs.append(row)
pairs = pd.DataFrame(pairs).join(meta, on="var")
pairs["absdh"] = pairs["dh"].abs()


# ── 지표 ───────────────────────────────────────────────────────────────
def metrics(g, m):
    """g: 사람 차이가 유의한 쌍. AUC는 방향 일치(동점 0.5), 구간은 문항 단위 부트스트랩."""
    out = dict(n_pairs=len(g), n_items=g["var"].nunique())
    if len(g) < MIN_PAIRS:
        return out
    dm, dh = g[f"dm_{m}"].to_numpy(), g["dh"].to_numpy()
    tie = np.abs(dm) < EPS
    s = np.where(tie, 0.5, (np.sign(dm) == np.sign(dh)).astype(float))
    v = pd.factorize(g["var"])[0]
    nv = v.max() + 1
    hit, cnt = np.bincount(v, s, nv), np.bincount(v, minlength=nv)
    idx = rng.integers(0, nv, (NB, nv))
    lo, hi = np.percentile(hit[idx].sum(1) / cnt[idx].sum(1), [2.5, 97.5])
    out.update(auc=s.mean(), auc_lo=lo, auc_hi=hi, tie_rate=tie.mean(),
               size_ratio=np.abs(dm).mean() / np.abs(dh).mean(),
               acc_nonties=(s[~tie]).mean() if (~tie).any() else np.nan)
    return out


def table(groups, extra):
    """groups: [(라벨 dict, 유의 쌍 DataFrame)] → 모델별 지표 행."""
    return pd.DataFrame([{**lab, "model": m, **metrics(g, m)} for lab, g in groups for m in MODELS]).assign(**extra)


scopes = {"all197": pairs, "validation49": pairs[pairs["split"] == "validation"]}
sig = lambda d: d[d["sig"] & (d["dh"] != 0)]

# 1. 쌍 종류별 (성별 vs 연령)
rows = []
for sc, d in scopes.items():
    for kind, k in d.groupby("kind"):
        base = dict(scope=sc, kind=kind, n_all=len(k), sig_share=k["sig"].mean(),
                    human_pos_share=(sig(k)["dh"] > 0).mean())
        rows += [{**base, **r} for r in table([({}, sig(k))], {}).to_dict("records")]
by_kind = pd.DataFrame(rows)

# 2. 사람 차이 크기(|dh|) 3분위
rows = []
for kind, k in sig(pairs).groupby("kind"):
    k = k.assign(tercile=pd.qcut(k["absdh"], 3, labels=["low", "mid", "high"]))
    rows += table([({"kind": kind, "tercile": t}, g) for t, g in k.groupby("tercile", observed=True)], {}).to_dict("records")
by_tercile = pd.DataFrame(rows)

# 3. 주제·민감도 (성별 쌍만, 탐색용)
gen = sig(pairs[pairs["kind"] == "gender"])
by_topic = table([({"topic": t}, g) for t, g in gen.groupby("topic")], {})
sens_bin = pd.qcut(meta["sensitivity_level"], 3, labels=["low", "mid", "high"])    # 연속값이라 문항 단위 3분위로 묶는다
gen = gen.assign(sens_bin=gen["var"].map(sens_bin))
by_sens = table([({"sensitivity_tercile": t}, g) for t, g in gen.groupby("sens_bin", observed=True)], {})

tables = {"by_kind": by_kind, "by_tercile": by_tercile, "gender_by_topic": by_topic,
          "gender_by_sensitivity": by_sens, "pairs": pairs}
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
for name in ["by_kind", "by_tercile", "gender_by_topic", "gender_by_sensitivity"]:
    print(f"\n[{name}]\n{tables[name].round(3).to_string(index=False)}")

run = sl.save_run(HERE / "out", params={"B": B, "NB": NB, "min_pairs": MIN_PAIRS, "eps": EPS, "seed": 20261117,
                                        "calibration": "none (원비율)"}, tables=tables)
print(f"\n기록: {HERE / 'out/run.json'} (commit {run['git_commit']}, dirty={run['git_dirty']})")
