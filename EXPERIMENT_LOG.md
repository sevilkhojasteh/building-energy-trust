# Experiment log

| date | change | result | note |
|---|---|---|---|
| 2026-10-15 | Repo structure created | - | - |

## 2026-10-19: global gradient boosting, 100 random buildings (seed 0)
- Features: hour, weekday, lag24, lag168, temp. Default HistGradientBoostingRegressor.
- Raw kWh: beats stronger baseline for 58% of buildings, median improvement 2.5%.
- Normalised by 2016 mean per building: 75%, median 9.9%.
- Note: 1 run, 100 buildings only. Not final.

## 2026-10-19: fixed 2016 scale vs recent 7-day scale (3 random samples of 100 buildings)
- Share beating stronger baseline: fixed 0.75/0.75/0.75, recent 0.73/0.72/0.79
- Median improvement: fixed 0.099/0.115/0.082, recent 0.103/0.098/0.100
- Difference changes sign across seeds -> no accuracy difference. Chose recent scale:
  Robust to level shifts (Rat_assembly_Pam -6.5 -> -0.3) and works for Swan (no 2016 data).