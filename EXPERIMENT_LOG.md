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

## 2026-10-22: M2 first intervals, 100 buildings (notebooks/03_uncertainty.ipynb)
- Fit Jan–Oct 2016, calibrate Nov–Dec 2016, test 2017; 90% intervals; 1 seed.
- Overall: quantile GBM 89.1% / relative width 0.446; split conformal 89.5% / 0.511
  (q_hat 0.255, same width everywhere); CQR 89.0% / 0.444 (q_cqr ≈ 0: nothing to correct).
- By hour: split conformal 94% at night, 85% at 14:00. Quantile GBM and CQR 88.5–90% at
  every hour, because their width adapts (0.30 at night, 0.58 at midday).
- By month: all methods worst in April–May (quantile 88.4%); split conformal swings most.
  My prediction "summer worst" was wrong.
- By building: very spread out (0.46–1.00). 16/100 buildings below 80% (quantile, CQR),
  13/100 (split conformal). My prediction "tightly packed" was wrong.
- Conclusion: overall coverage looks fine but hides large differences between buildings.

- Per building (same notebook): CQR per building 88.5% / width 0.475 / 4 of 100 buildings < 80% / std 0.047 (best). Split conformal per building 88.4% / 0.506 / 10 / 0.067. Overall coverage dropped ~0.5 points: per-building Nov–Dec calibration is less representative.

- ACI (γ=0.05, daily update, on CQR per building): 90.0% / width 0.508 / 0 buildings < 80% (by construction). Wolf: sawtooth (bursts of misses); Hog_science_Max: α up to 0.37 in summer (Nov–Dec calibration too wide).

- ACI by hour 89.1–91.2%, by month 89.3–90.8%. CQR per building drops to 86.6% in June (Nov–Dec calibration doesn't fit summer); ACI fixes it. My prediction "ACI still low at 14:00" was wrong.

- Reliability diagram (100 buildings): CQR per building 1.2–1.6 points under target at every
  level (48.8 / 78.4 / 88.5 / 93.5%); ACI on target at every level (49.9 / 80.0 / 90.0 / 95.0%).
  Width 50% → 95%: about ×4.3 (0.163 → 0.705 for ACI), more than a normal distribution
  would need (×2.9): heavy tails.

## 2026-10-23: M2 on all 1,083 buildings (scripts/m2_intervals.py)
- Fit Jan–Oct 2016, calibrate Nov–Dec 2016, test 2017; 1 seed; ACI γ = 0.05, daily update.
- 90% intervals: split conformal 89.9% / width 0.535 / 160 buildings below 80% (worst 0.29);
  quantile GBM 89.4% / 0.467 / 127; CQR 89.8% / 0.474 / 116; CQR per building 89.4% / 0.500 / 31;
  ACI 90.0% / 0.524 / 0 (worst building 0.888, worst hour 0.886, worst month 0.893).
- Same ranking as on 100 buildings. The sample over-estimated the share of bad buildings for
  global CQR (16% on 100 vs 10.7% on all).
- Reliability diagram: every method within about 1 point of target at 50/80/90/95%
  overall; the differences are in the groups (buildings, hours, months).
- Code fix: the script picks the exact k-th calibration score; the notebook's
  np.quantile(method="higher") picked one position higher (effect < 0.1 point).

## 2026-10-23: M3 stability metrics (defined before computing)
- Explained: 1,000 fixed hours of 2017 (100-building sample), background: 100 fixed hours of 2016.
- Exact interventional SHAP (all 32 subsets of the 5 inputs).
- Sources of variation in Step 3.1: random_state 0–4 only.
- Global: Spearman rank correlation of mean |SHAP| between all 10 seed pairs (mean, min);
  top-2 agreement (share of pairs with the same two most important inputs).
- Local: cosine similarity of the 5-number SHAP vector of the same hour between seed pairs
  (median, 5th percentile over hours).

- Step 3.1b (defined before computing): 5 models, each trained on a bootstrap sample of the 100
  buildings (random_state 0 for the model, bootstrap rng seed 1). Same metrics as above.

## 2026-10-25: M3 Step 3.1 results (100 buildings, 1,000 explained hours)
- Both lags missing: 375 of 864,321 test hours (0.04%, 22 buildings), only 600 such training
  hours. Error there 1.666 vs 0.113 elsewhere (15× worse): the model should not forecast
  these hours. Known limitation for M1/M2 (those hours are included in train_gbm.py and
  m2_intervals.py).
- Seeds 0–4: Spearman 1.000, top-2 10/10, local cosine median 0.9989, p5 0.9855.
  Weak test: seeds only change the 10% held-out rows.
- Bootstrap of buildings: Spearman mean 0.940, min 0.900 (hour and temp swap places);
  top-2 always lag24_n and lag168_n; local cosine median 0.9907, p5 0.896; some hours
  negative (explanations pointing in opposite directions).
- Post-hoc check (added after seeing the results): smallest 20% of explanations have median
  min-cosine 0.915 vs 0.981 for the rest; hours with a missing lag 0.875. Cosine is partly a
  metric problem on near-average hours, partly a real problem on missing-lag hours.
- Second metric (added after seeing the results): total |SHAP difference| per hour between two
  models. Seeds: median 0.011, p95 0.038. Bootstrap: median 0.030, p95 0.125. For 5% of hours,
  two equally accurate models' explanations differ by more than the typical forecast error (0.113).
- Step 3.2a (defined before computing): group SHAP with 3 groups, past load = {lag24_n, lag168_n},
  calendar = {hour, weekday}, temperature = {temp}, on the same 5 bootstrap models. Same metrics.
  Hypothesis: grouped explanations are more stable, because the model can split credit between
  lag24_n and lag168_n differently from one model to the next.


- Step 3.2a result: past-load share going to lag24_n 0.691–0.698 across bootstrap models
  (stable). Grouped total |SHAP difference| median 0.023 vs 0.030 individual, p95 0.110 vs 0.125.
  Grouping helps a little; the hypothesis "instability is mainly credit-splitting between the
  two lags" is mostly rejected. Temperature (0.009–0.018) and calendar (0.033–0.041) importance
  still vary between bootstrap samples: they depend on which buildings are in the data.
- Step 3.2b (defined before computing): 4 models trained on 6-month windows of 2016
  (Jan–Jun, Mar–Aug, May–Oct, Jul–Dec), random_state 0, all 100 buildings. Same explained hours,
  same background, same metrics (individual and grouped), plus each model's test MAE (normalised).

- Step 3.2b result (6-month training windows of 2016): test MAE 0.1141–0.1153 (equally accurate).
  Spearman mean 0.950, min 0.900; top-2 6/6; cosine median 0.9916, p5 0.902; total |SHAP difference|
  median 0.027, p95 0.114. Temperature 0.0105–0.0141 (highest May–Oct), hour 0.0105–0.0174.
  Training period changes explanations about as much as the building sample, a bit less.
- Summary of Step 3.1–3.2b (median / p95 total |SHAP difference|): seeds 0.011 / 0.038,
  training period 0.027 / 0.114, building bootstrap 0.030 / 0.125.
- Step 3.2c (defined before computing): seed-0 model, same 1,000 hours and background. Add
  Gaussian noise of 1%, 5% and 10% of each input's robust spread (IQR / 1.349) to lag24_n,
  lag168_n and temp only. Report median |Δ forecast|, median and p95 total |Δ SHAP|, and the
  share of "hidden reshuffles" (forecast moves < 0.005 but explanation moves > 0.02).






## Ideas (not done yet)
- Weekend/weekly-building weakness (seen in 2017 figure): try building type
  (primaryspaceusage), a "how weekly is this building" feature, and holidays.
  Evaluate on Nov–Dec 2016 validation, not 2017.
