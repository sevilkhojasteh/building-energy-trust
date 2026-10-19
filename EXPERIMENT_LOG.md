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

## 2026-10-19: MLP vs gradient boosting, all 1,083 buildings (scripts/train_nn.py)
- Validation (Jan–Oct fit, Nov–Dec 2016 val): loss flat from epoch 1 (0.0455–0.0465);
  used 5 epochs, the minimum. The choice barely matters.
- Same rows (hours with every input), 5 seeds each:
  GBM 78.3 ± 0.3% / 11.5 ± 0.2%; MLP 78.2 ± 1.1% / 11.0 ± 0.4%.
- The MLP caught up with more data (73.2% on 100 buildings). GBM is still more stable.
- Figure: both models weak on weekends; the worst buildings are strictly weekly ones.






## Ideas (not done yet)
- Weekend/weekly-building weakness (seen in 2017 figure): try building type
  (primaryspaceusage), a "how weekly is this building" feature, and holidays.
  Evaluate on Nov–Dec 2016 validation, not 2017.
