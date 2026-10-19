"""MLP and gradient boosting on all buildings, scored on the same test rows.

Run from the project root with .venv active:
    python scripts/train_nn.py val   # pick the number of epochs (Nov-Dec 2016 validation)
    python scripts/train_nn.py       # final run: 5 seeds of each model
"""
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.append(".")
from src.features import make_features

EPOCHS = 5            # change this after the validation run
MODE = sys.argv[1] if len(sys.argv) > 1 else "final"
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

# 2. NN inputs as one numpy matrix: sin/cos time, clipped lags, temperature
h = data["hour"].to_numpy()
wd = data["weekday"].to_numpy()
X = np.column_stack([
    np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
    np.sin(2 * np.pi * wd / 7), np.cos(2 * np.pi * wd / 7),
    data["lag24_n"].clip(upper=5).to_numpy(),
    data["lag168_n"].clip(upper=5).to_numpy(),
    data["temp"].to_numpy(),
]).astype("float32")
y = data["load_n"].clip(upper=5).to_numpy(dtype="float32")

# 3. Rows where every NN input and the target exist; split by time
ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
year, month = data.index.year, data.index.month
is_train = ok & (year == 2016)
is_test = ok & (year == 2017)
is_fit = is_train & (month <= 10)
is_val = is_train & (month >= 11)

# 4. Standardise the last 3 columns with statistics from the rows we train on
fit_rows = is_fit if MODE == "val" else is_train
mu = X[fit_rows, 4:].mean(axis=0)
sd = X[fit_rows, 4:].std(axis=0)
X[:, 4:] = (X[:, 4:] - mu) / sd


def make_mlp():
    return nn.Sequential(nn.Linear(7, 64), nn.ReLU(),
                         nn.Linear(64, 64), nn.ReLU(),
                         nn.Linear(64, 1))


def predict(model, Xt):
    """Predict in chunks, so 9 million rows don't fill the memory at once."""
    model.eval()
    with torch.no_grad():
        out = [model(Xt[i:i + 100_000]) for i in range(0, len(Xt), 100_000)]
    return torch.cat(out).squeeze(1).numpy()


def train_mlp(seed, Xtr, ytr, epochs, Xval=None, yval=None):
    torch.manual_seed(seed)
    model = make_mlp()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 1024):
            idx = perm[i:i + 1024]
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if Xval is not None:
            val_loss = float(np.mean((predict(model, Xval) - yval) ** 2))
            print("epoch", epoch, "validation loss", round(val_loss, 4), flush=True)
    return model


# 5a. Validation mode: choose the number of epochs, then stop
if MODE == "val":
    Xf = torch.from_numpy(X[is_fit])
    yf = torch.from_numpy(y[is_fit]).unsqueeze(1)
    Xv = torch.from_numpy(X[is_val])
    print(len(Xf), "fit rows,", len(Xv), "validation rows", flush=True)
    train_mlp(0, Xf, yf, epochs=15, Xval=Xv, yval=y[is_val])
    sys.exit()

# 5b. Final mode: both models, 5 seeds, the same test rows
Xtr = torch.from_numpy(X[is_train])
ytr = torch.from_numpy(y[is_train]).unsqueeze(1)
Xte = torch.from_numpy(X[is_test])
test = data.loc[is_test, ["building", "load", "lag24", "lag168", "scale"]]
print(len(Xtr), "train rows,", len(Xte), "test rows", flush=True)


def mae_by_building(pred):
    return (test["load"] - pred).abs().groupby(test["building"], observed=True).mean()


res = pd.DataFrame({"mae_yesterday": mae_by_building(test["lag24"]),
                    "mae_lastweek": mae_by_building(test["lag168"])})
scale = test["scale"].to_numpy()

for seed in range(5):
    t0 = time.time()
    mlp = train_mlp(seed, Xtr, ytr, epochs=EPOCHS)
    pred_nn = predict(mlp, Xte) * scale

    gbm = HistGradientBoostingRegressor(random_state=seed)
    gbm.fit(data.loc[is_train, FEATS], data.loc[is_train, "load_n"])
    pred_gbm = gbm.predict(data.loc[is_test, FEATS]) * scale

    res[f"nn_seed{seed}"] = mae_by_building(pred_nn)
    res[f"gbm_seed{seed}"] = mae_by_building(pred_gbm)
    if seed == 0:
        out = test[["building", "load", "lag24", "lag168"]].copy()
        out["pred_nn"] = pred_nn
        out["pred_gbm"] = pred_gbm
        out.to_parquet("data/processed/preds_2017_seed0.parquet")
    print("seed", seed, "done in", round(time.time() - t0), "s", flush=True)

res.to_csv("results/tables/m1_compare_all.csv")

# 6. Summary: share of buildings beating the stronger baseline, median improvement
best = res[["mae_yesterday", "mae_lastweek"]].min(axis=1)
rows = []
for name in ["nn", "gbm"]:
    for seed in range(5):
        imp = (best - res[f"{name}_seed{seed}"]) / best
        rows.append({"model": name, "seed": seed,
                     "share": (imp > 0).mean(), "median": imp.median()})
summary = pd.DataFrame(rows)
print(summary.pivot(index="seed", columns="model", values=["share", "median"]).round(3))
print(summary.groupby("model")[["share", "median"]].agg(["mean", "std"]).round(3))
