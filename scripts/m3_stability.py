"""M3: explanation stability on all buildings.

Exact interventional SHAP for 4 sources of variation: seeds, building bootstrap,
training period, input noise. Run from the project root with .venv active:
    python scripts/m3_stability.py
"""
import sys
import time
from itertools import combinations
from math import factorial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.append(".")
from src.features import make_features

FEATS = ["hour", "weekday", "lag24_n", "lag168_n", "temp"]
GROUPS = {"past load": [2, 3], "calendar": [0, 1], "temperature": [4]}
N_EXPLAIN, N_BACKGROUND = 2000, 100

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

train = data[data.index.year == 2016].dropna(subset=["load_n"])
test = data[data.index.year == 2017].dropna(subset=["load_n"])
del data
print(len(train), "train rows,", len(test), "test rows", flush=True)

X_explain = test[FEATS].sample(N_EXPLAIN, random_state=0).to_numpy(dtype="float64")
background = train[FEATS].sample(N_BACKGROUND, random_state=0).to_numpy(dtype="float64")
y_test = test["load_n"].to_numpy()


def shapley(model, X, groups):
    """Exact interventional Shapley values; each group of columns is one player."""
    names = list(groups)
    g, (n, p), b = len(names), X.shape, len(background)
    v = {}
    for size in range(g + 1):
        for G in combinations(range(g), size):
            cols = [c for k in G for c in groups[names[k]]]
            Z = np.repeat(background[None, :, :], n, axis=0)
            Z[:, :, cols] = X[:, None, cols]
            Zdf = pd.DataFrame(Z.reshape(n * b, p), columns=FEATS)
            v[G] = model.predict(Zdf).reshape(n, b).mean(axis=1)
    phi = np.zeros((n, g))
    for j in range(g):
        others = [k for k in range(g) if k != j]
        for size in range(g):
            w = factorial(size) * factorial(g - size - 1) / factorial(g)
            for G in combinations(others, size):
                phi[:, j] += w * (v[tuple(sorted(G + (j,)))] - v[G])
    pred = model.predict(pd.DataFrame(X, columns=FEATS))
    assert np.abs(phi.sum(axis=1) + v[()] - pred).max() < 1e-8, "additivity check failed"
    return phi


SINGLE = {f: [i] for i, f in enumerate(FEATS)}


def fit(rows, seed=0):
    return HistGradientBoostingRegressor(random_state=seed).fit(rows[FEATS], rows["load_n"])


def cosine(A, B):
    return (A * B).sum(axis=1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))


summary, importances = [], []


def evaluate(source, models, labels):
    t0 = time.time()
    phis = [shapley(m, X_explain, SINGLE) for m in models]
    gphis = [shapley(m, X_explain, GROUPS) for m in models]
    for label, m, ph, gph in zip(labels, models, phis, gphis):
        mae = np.abs(y_test - m.predict(test[FEATS])).mean()
        importances.append({"source": source, "model": label, "test_mae": mae,
                            **dict(zip(FEATS, np.abs(ph).mean(axis=0))),
                            **dict(zip(GROUPS, np.abs(gph).mean(axis=0)))})
    pairs = list(combinations(range(len(models)), 2))
    imp = [np.abs(ph).mean(axis=0) for ph in phis]
    top2 = [set(np.argsort(imp[a])[-2:]) == set(np.argsort(imp[b])[-2:]) for a, b in pairs]
    cos = np.concatenate([cosine(phis[a], phis[b]) for a, b in pairs])
    diff = np.mean([np.abs(phis[a] - phis[b]).sum(axis=1) for a, b in pairs], axis=0)
    gdiff = np.mean([np.abs(gphis[a] - gphis[b]).sum(axis=1) for a, b in pairs], axis=0)
    rho = [spearmanr(imp[a], imp[b])[0] for a, b in pairs]
    summary.append({"source": source, "models": len(models),
                    "spearman_mean": np.mean(rho), "spearman_min": np.min(rho),
                    "top2_agreement": np.mean(top2),
                    "cosine_median": np.median(cos), "cosine_p5": np.percentile(cos, 5),
                    "diff_median": np.median(diff), "diff_p95": np.percentile(diff, 95),
                    "group_diff_median": np.median(gdiff), "group_diff_p95": np.percentile(gdiff, 95)})
    print(source, "done in", round(time.time() - t0), "s", flush=True)
    return phis


# 2. Seeds
t0 = time.time()
seed_models = [fit(train, seed=s) for s in range(5)]
print("seed models trained in", round(time.time() - t0), "s", flush=True)
seed_phis = evaluate("seed", seed_models, [f"seed {s}" for s in range(5)])

# 3. Bootstrap of buildings
rng_b = np.random.default_rng(1)
by_building = train.groupby("building", observed=True).indices
names = list(by_building)
boot_models = []
for i in range(5):
    picked = rng_b.choice(names, size=len(names), replace=True)
    rows = np.concatenate([by_building[b] for b in picked])
    boot_models.append(fit(train.iloc[rows]))
    print("bootstrap model", i, "trained:", len(set(picked)), "different buildings", flush=True)
evaluate("building bootstrap", boot_models, [f"bootstrap {i}" for i in range(5)])
del boot_models

# 4. Training period: 6-month windows of 2016
windows = {"Jan-Jun": (1, 6), "Mar-Aug": (3, 8), "May-Oct": (5, 10), "Jul-Dec": (7, 12)}
month = train.index.month
period_models = [fit(train[(month >= a) & (month <= z)]) for a, z in windows.values()]
evaluate("training period", period_models, list(windows))
del period_models

# 5. Input noise on lag24_n, lag168_n, temp (seed-0 model)
q = train[FEATS].quantile([0.25, 0.75])
spread = ((q.loc[0.75] - q.loc[0.25]) / 1.349).to_numpy()
model0, phi0 = seed_models[0], seed_phis[0]
pred0 = model0.predict(pd.DataFrame(X_explain, columns=FEATS))
rng_n = np.random.default_rng(2)
noise_rows = []
for level in [0.01, 0.05, 0.10]:
    Xn = X_explain.copy()
    Xn[:, [2, 3, 4]] += rng_n.normal(0, 1, (len(Xn), 3)) * level * spread[[2, 3, 4]]
    d_pred = np.abs(model0.predict(pd.DataFrame(Xn, columns=FEATS)) - pred0)
    d_shap = np.abs(shapley(model0, Xn, SINGLE) - phi0).sum(axis=1)
    noise_rows.append({"noise": level, "median_d_forecast": np.median(d_pred),
                       "median_d_shap": np.median(d_shap), "p95_d_shap": np.percentile(d_shap, 95),
                       "hidden_reshuffles": np.mean((d_pred < 0.005) & (d_shap > 0.02))})
print("input noise done", flush=True)

# 6. Save tables and one figure
summary = pd.DataFrame(summary)
importances = pd.DataFrame(importances)
noise = pd.DataFrame(noise_rows)
summary.to_csv("results/tables/m3_stability_summary.csv", index=False)
importances.to_csv("results/tables/m3_importances.csv", index=False)
noise.to_csv("results/tables/m3_input_noise.csv", index=False)
print(summary.round(4).to_string(index=False))
print(noise.round(4).to_string(index=False))
print(importances.groupby("source")[FEATS + ["test_mae"]].agg(["min", "max"]).round(4).T.to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sources = list(summary["source"])
for k, src in enumerate(sources):
    sub = importances[importances["source"] == src]
    for i, f in enumerate(FEATS):
        axes[0].scatter(np.full(len(sub), i) + (k - 1) * 0.2, sub[f], s=18,
                        color=f"C{k}", label=src if i == 0 else None)
axes[0].set_xticks(range(len(FEATS)), FEATS)
axes[0].set(ylabel="global importance (mean |SHAP|)", title="Global importance, one dot per model")
axes[0].legend()
axes[1].bar(sources, summary["diff_median"], label="median")
axes[1].bar(sources, summary["diff_p95"] - summary["diff_median"], bottom=summary["diff_median"],
            alpha=0.4, label="up to 95th percentile")
axes[1].axhline(importances.loc[0, "test_mae"], color="gray", ls="--", label="forecast MAE (seed 0)")
axes[1].set(ylabel="total |SHAP difference| per hour", title="How much explanations change")
axes[1].legend()
fig.tight_layout()
fig.savefig("results/figures/m3_stability.png", dpi=150)
print("saved results/tables/m3_*.csv and results/figures/m3_stability.png")
