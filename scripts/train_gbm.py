import sys
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.append(".")
from src.features import make_features

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
del parts

FEATS = ["hour", "weekday", "lag24_n", "lag168_n", "temp"]
train = data[data.index.year == 2016].dropna(subset=["load_n"])
test = data[data.index.year == 2017].dropna(subset=["load", "scale"])
print(len(train), "train rows,", len(test), "test rows", flush=True)

def mae_by_building(pred):
    return (test["load"] - pred).abs().groupby(test["building"], observed=True).mean()

res = pd.DataFrame({"mae_yesterday": mae_by_building(test["lag24"]),
                    "mae_lastweek": mae_by_building(test["lag168"])})

for seed in range(5):
    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(train[FEATS], train["load_n"])
    pred = model.predict(test[FEATS]) * test["scale"].to_numpy()
    res[f"mae_seed{seed}"] = mae_by_building(pred)
    print("seed", seed, "done", flush=True)

res.to_csv("results/tables/m1_gbm_all.csv")

best = res[["mae_yesterday", "mae_lastweek"]].min(axis=1)
seeds = res.filter(like="mae_seed")
improve = seeds.apply(lambda col: (best - col) / best)
summary = pd.DataFrame({"share_beats_best": (improve > 0).mean(),
                        "median_improve": improve.median()})
print(summary.round(3))
print(summary.agg(["mean", "std"]).round(3))
