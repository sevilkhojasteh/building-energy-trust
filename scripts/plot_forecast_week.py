"""Forecast vs actual for 3 buildings, one week of 2017. Run after train_nn.py."""
import pandas as pd
import matplotlib.pyplot as plt

preds = pd.read_parquet("data/processed/preds_2017_seed0.parquet")
res = pd.read_csv("results/tables/m1_compare_all.csv", index_col=0)

best = res[["mae_yesterday", "mae_lastweek"]].min(axis=1)
imp = ((best - res["gbm_seed0"]) / best).sort_values()
n = len(imp)
picks = {"Among the best (90th percentile)": imp.index[int(n * 0.9)],
         "Typical (median)": imp.index[n // 2],
         "Among the worst (10th percentile)": imp.index[int(n * 0.1)]}

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for ax, (label, b) in zip(axes, picks.items()):
    p = preds[preds["building"] == b].loc["2017-03-06":"2017-03-12"].asfreq("h")
    ax.plot(p.index, p["load"], color="black", lw=2, label="actual")
    ax.plot(p.index, p["pred_gbm"], label="gradient boosting")
    ax.plot(p.index, p["pred_nn"], label="MLP")
    ax.plot(p.index, p["lag24"], color="gray", ls="--", label="yesterday baseline")
    ax.set_title(f"{label}: {b} (GBM improvement over best baseline: {imp[b]:.0%})")
    ax.set_ylabel("kWh")
axes[0].legend(ncol=4, loc="upper right")
fig.tight_layout()
fig.savefig("results/figures/forecast_week.png", dpi=150)
print("saved results/figures/forecast_week.png")
