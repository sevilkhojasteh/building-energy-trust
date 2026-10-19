import pandas as pd

def make_features(b, df_clean, temp):
    """Feature table for one building: one row per hour."""
    site = b.split("_")[0]
    s = df_clean[b]
    recent = s.shift(24).rolling(168, min_periods=24).mean()
    feat = pd.DataFrame({"building": b, "load": s, "hour": s.index.hour,
                         "weekday": s.index.weekday, "lag24": s.shift(24),
                         "lag168": s.shift(168), "temp": temp[site],
                         "scale": recent.where(recent > 0.01)})
    for col in ["load", "lag24", "lag168"]:
        feat[col + "_n"] = feat[col] / feat["scale"]
    return feat
