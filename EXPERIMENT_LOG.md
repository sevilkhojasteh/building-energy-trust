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


The global gradient-boosting model beats the stronger naive baseline for 76.3% ± 0.3% of buildings, with a median improvement of 10.7% ± 0.1% (5 seeds).
## 2026-10-19: MLP vs gradient boosting, 100 buildings
- Same 100 buildings (rng seed 0), same train/test rows, 5 seeds each.
- MLP 7→64→64→1 (sin/cos hour and weekday, lags clipped at 5, inputs standardised on 2016),
  Adam lr 1e-3, batch 1024, 7 epochs chosen on Nov–Dec 2016 validation.
- GBM: 79.0% of buildings beat the stronger baseline, median improvement 12.2%.
- MLP: 73.2 ± 4.3%, median improvement 8.1 ± 1.0%.
- Conclusion: GBM is better and far more stable across seeds.