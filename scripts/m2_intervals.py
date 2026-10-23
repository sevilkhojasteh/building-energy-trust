"""M2: prediction intervals on all buildings, 5 methods, 4 levels.

Run from the project root with .venv active:
    python scripts/m2_intervals.py
"""
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.append(".")
from src.features import make_features

LEVELS = [0.5, 0.8, 0.9, 0.95]
GAMMA = 0.05
FEATS = ["hour", "weekday", "lag24_n", "lag168_n", "temp"]

# 1. Features for all buildings (same as train_gbm.py)
df_clean = pd.read_parquet("data/processed/electricity_clean.parquet")
temp = pd.read_parquet("data/processed/air_temperature.parquet")
buildings = pd.read_csv("configs/train_buildings.csv")["building"].tolist()

parts = []
for b in buildings:
    f = make_features(b, df_clean, temp)
    num = f.columns.drop("building")
    f[num] = f[num].astype("float32")
    parts.append(f)
data = pd.concat(parts)
data["building"] = data["building"].astype("category")
del parts, df_clean

# 2. Three time blocks: fit, calibration, test
year, month = data.index.year, data.index.month
fit = data[(year == 2016) & (month <= 10)].dropna(subset=["load_n"])
cal = data[(year == 2016) & (month >= 11)].dropna(subset=["load_n"])
test = data[year == 2017].dropna(subset=["load_n"])
del data
print(len(fit), "fit rows,", len(cal), "calibration rows,", len(test), "test rows", flush=True)

y_cal = cal["load_n"].to_numpy()
y = test["load_n"].to_numpy()
hours = test["hour"].to_numpy()
months = test.index.month.to_numpy()
days = test.index.normalize().to_numpy()
test_codes = test["building"].cat.codes.to_numpy()
names = test["building"].cat.categories
cal_idx = cal.groupby("building", observed=True).indices    # building -> row positions
test_idx = test.groupby("building", observed=True).indices
no_cal = [b for b in test_idx if b not in cal_idx]
print(len(no_cal), "test buildings without calibration rows (they use the global q)", flush=True)


def conformal_q(s_sorted, a):
    """The ceil((n+1)(1-a))-th smallest score, capped to the scores we have."""
    n = len(s_sorted)
    k = int(np.ceil((n + 1) * (1 - a)))
    return s_sorted[min(max(k, 1), n) - 1]


def per_building_q(scores, a):
    s_global = np.sort(scores)
    q = np.empty(len(test))
    for b, rows in test_idx.items():
        s_b = np.sort(scores[cal_idx[b]]) if b in cal_idx else s_global
        q[rows] = conformal_q(s_b, a)
    return q


def aci(scores, lo_t, hi_t, a0):
    """Adaptive conformal on top of per-building CQR, one update per day."""
    s_global = np.sort(scores)
    lo, hi = np.empty(len(test)), np.empty(len(test))
    for b, rows in test_idx.items():
        s_b = np.sort(scores[cal_idx[b]]) if b in cal_idx else s_global
        _, starts = np.unique(days[rows], return_index=True)
        bounds = list(starts) + [len(rows)]
        a = a0
        for k in range(len(starts)):
            idx = rows[bounds[k]:bounds[k + 1]]
            q = conformal_q(s_b, a)
            lo[idx], hi[idx] = lo_t[idx] - q, hi_t[idx] + q
            miss = 1 - ((y[idx] >= lo[idx]) & (y[idx] <= hi[idx])).mean()
            a += GAMMA * (a0 - miss)
    return lo, hi


rows_out, per_b_90, month_90 = [], {}, {}


def evaluate(name, level, lo, hi):
    covered = pd.Series((y >= lo) & (y <= hi))
    per_b = covered.groupby(test_codes).mean()
    by_month = covered.groupby(months).mean()
    rows_out.append({"method": name, "target": level,
                     "coverage": covered.mean(),
                     "rel_width": np.mean(hi - lo),
                     "buildings_10pt_below": int((per_b < level - 0.1).sum()),
                     "per_building_std": per_b.std(),
                     "worst_building": per_b.min(),
                     "worst_hour": covered.groupby(hours).mean().min(),
                     "worst_month": by_month.min()})
    if level == 0.9:
        per_b_90[name] = pd.Series(per_b.to_numpy(), index=names[per_b.index])
        month_90[name] = by_month


# 3. Point model for split conformal (trained once)
t0 = time.time()
point = HistGradientBoostingRegressor(random_state=0).fit(fit[FEATS], fit["load_n"])
res_cal = np.sort(np.abs(y_cal - point.predict(cal[FEATS])))
pred = point.predict(test[FEATS])
print("point model done in", round(time.time() - t0), "s", flush=True)

# 4. Every level: quantile models, then the 5 methods
for level in LEVELS:
    t0 = time.time()
    a0 = 1 - level
    lo_m = HistGradientBoostingRegressor(loss="quantile", quantile=a0 / 2, random_state=0)
    hi_m = HistGradientBoostingRegressor(loss="quantile", quantile=1 - a0 / 2, random_state=0)
    lo_m.fit(fit[FEATS], fit["load_n"])
    hi_m.fit(fit[FEATS], fit["load_n"])
    lo_t, hi_t = lo_m.predict(test[FEATS]), hi_m.predict(test[FEATS])
    scores = np.maximum(lo_m.predict(cal[FEATS]) - y_cal, y_cal - hi_m.predict(cal[FEATS]))

    q_split = conformal_q(res_cal, a0)
    q_cqr = conformal_q(np.sort(scores), a0)
    q_b = per_building_q(scores, a0)

    evaluate("Split conformal", level, pred - q_split, pred + q_split)
    evaluate("Quantile GBM", level, lo_t, hi_t)
    evaluate("CQR", level, lo_t - q_cqr, hi_t + q_cqr)
    evaluate("CQR per building", level, lo_t - q_b, hi_t + q_b)
    evaluate("ACI", level, *aci(scores, lo_t, hi_t, a0))
    print("level", level, "done in", round(time.time() - t0), "s", flush=True)

# 5. Save tables and one figure
table = pd.DataFrame(rows_out)
table.to_csv("results/tables/m2_intervals_all.csv", index=False)
pd.DataFrame(per_b_90).to_csv("results/tables/m2_coverage_per_building_90.csv")
print(table[table["target"] == 0.9].drop(columns="target").round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot([0.45, 1], [0.45, 1], color="gray", ls="--", label="perfect calibration")
for name in ["Split conformal", "CQR", "CQR per building", "ACI"]:
    t = table[table["method"] == name]
    axes[0].plot(t["target"], t["coverage"], marker="o", label=name)
    axes[1].plot(month_90[name], marker="o", label=name)
axes[1].axhline(0.9, color="gray", ls="--")
axes[0].set(xlabel="target coverage", ylabel="coverage on 2017", title="Reliability diagram")
axes[1].set(xlabel="month (2017)", ylabel="coverage", title="Coverage by month, 90% intervals")
axes[0].legend()
fig.tight_layout()
fig.savefig("results/figures/m2_calibration.png", dpi=150)
print("saved results/tables/m2_intervals_all.csv, results/tables/m2_coverage_per_building_90.csv,"
      " results/figures/m2_calibration.png")
