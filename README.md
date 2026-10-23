# Trustworthy building-energy forecasting

Calibrated uncertainty, stable explanations, and federated and edge deployment for
day-ahead building electricity forecasts, on public data.

**Status:** M1 (forecasting) and M2 (uncertainty) done. M3 (explanations) is next.

| Part | Question | Status |
|---|---|---|
| M1 Forecasting | Can a model beat strong naive baselines, per building? | Done |
| M2 Uncertainty | Are the prediction intervals calibrated? | Done |
| M3 Explanations | Why does the model predict this, and is the explanation stable? | Planned |
| M4 Federated learning | Can sites train one model together without sharing data (simulated)? | Planned |

## Data

[Building Data Genome Project 2 (BDG2)](https://github.com/buds-lab/building-data-genome-project-2),
hourly electricity meters for 2016–2017, licence CC BY-SA 4.0. The raw files are not
in this repository: download `electricity_cleaned.csv`, `weather.csv` and `metadata.csv`
into `data/raw/`.

Selection (details and plots in `notebooks/01_explore.ipynb`):

- Runs of identical values longer than 24 h are treated as missing (stuck or dead meters).
- Buildings are kept if at most 20% of hours are missing in **both** 2016 and 2017:
  **1,083 buildings on 14 sites** (`configs/train_buildings.csv`).
- Sites Panther and Swan are held out as unseen test sites. Shrew and Lamb are
  excluded because they have no usable 2017 data.
- Weather gaps of up to 24 h are interpolated; longer gaps stay missing.
- Timestamps are local time.

## Task and evaluation

- **Task:** each midnight, forecast the next 24 hourly values for each building.
- **Split by time:** train on 2016, test once on 2017. Model choices (for example the
  number of MLP epochs) are made on a validation period, November–December 2016.
- **Inputs:** hour, weekday, the load 24 h and 168 h earlier, and outdoor air temperature.
  Load values are divided by the building's mean load over the previous 7 days, so one
  global model can serve buildings of very different sizes.
- **Metric:** mean absolute error (MAE, kWh) per building, on hours with real values.
- **Baselines:** "yesterday" (same hour, 24 h earlier) and "last week" (same hour,
  168 h earlier). A model wins for a building only if it beats the **stronger** of the
  two for that building. Yesterday is the stronger baseline for 62% of buildings.

## Results (M1)

| Model | Buildings where it beats the stronger baseline | Median improvement over it |
|---|---|---|
| Gradient boosting (scikit-learn `HistGradientBoostingRegressor`) | 78.3 ± 0.3% | 11.5 ± 0.2% |
| MLP 7→64→64→1 (PyTorch) | 78.2 ± 1.1% | 11.0 ± 0.4% |

Mean ± standard deviation over 5 seeds, all 1,083 buildings. Both models are trained and
scored on exactly the same rows: hours where every input exists. On all test hours,
gradient boosting beats the stronger baseline for 76.3 ± 0.3% of buildings, because the
hours right after data gaps are harder.

![Forecast vs actual, one week of March 2017](results/figures/forecast_week.png)

The three buildings are picked by rule (90th percentile, median and 10th percentile of
improvement), not by hand.

**Takeaways**

- The two models are equally often better than the baselines. Gradient boosting has a
  slightly higher median improvement and is 3–4 times more stable across seeds.
- On a 100-building sample the MLP was clearly behind (73.2 ± 4.3%). With all buildings
  it caught up: the neural network needed much more data than the trees.
- Both models are weakest on weekends and on buildings with a strict weekly pattern. A
  single global model learns one average way to combine "yesterday" and "last week",
  and for an office that repeats every week this is worse than just copying last week.
  This motivates personalised models in M4.
- The MLP matters for later work: federated averaging needs model weights that can be
  averaged, which trees don't have.

**Limitations**

- Temperature is the measured value, not a forecast, so results are an upper bound on
  what a real weather forecast would allow. Removing temperature changed MAE very little
  on the building checked.
- One test year only. The held-out sites (Panther, Swan) are not evaluated yet.

## Results (M2: prediction intervals)

**Question.** A "90% interval" should contain the real value in 90% of hours. Is that true
for every building, every hour of the day and every season, not only on average?

**Setup.** Gradient boosting, all 1,083 buildings. Fit on January–October 2016, calibrate
on November–December 2016, test once on 2017. Width is relative to the building's mean load
over the previous 7 days (0.50 = the interval is half a normal hour's load wide). One seed.

**Methods.**
- *Split conformal*: point forecast ± one global error size from the calibration months.
- *Quantile GBM*: two models trained with the pinball loss for the 5% and 95% quantiles.
- *CQR* (conformalised quantile regression): the quantile interval, corrected on the
  calibration months.
- *CQR per building*: the same correction, computed separately for each building.
- *ACI* (adaptive conformal inference) on top of CQR per building: after each day, the
  interval for the next day gets wider if the building's interval missed more than 10% of
  hours, narrower if it missed less.

| Method (90% intervals) | Coverage | Mean width | Buildings below 80% | Worst building | Worst month |
|---|---|---|---|---|---|
| Split conformal | 89.9% | 0.535 | 160 | 29% | 88.7% |
| Quantile GBM | 89.4% | 0.467 | 127 | 61% | 89.0% |
| CQR | 89.8% | 0.474 | 116 | 61% | 89.4% |
| CQR per building | 89.4% | 0.500 | 31 | 70% | 88.8% |
| ACI on CQR per building | 90.0% | 0.524 | 0 | 89% | 89.3% |

![Reliability diagram and coverage by month](results/figures/m2_calibration.png)

**Takeaways**

- **Average coverage hides failures.** Split conformal covers 89.9% overall, but 160 of
  1,083 buildings get less than 80%. On a 100-building sample it covered 94% at night and
  only 85% at 14:00, because one fixed width is too wide for quiet hours and too narrow for
  busy ones.
- **Each fix addresses one group.** Quantile models adapt the width to the hour.
  Calibrating per building cuts the buildings below 80% from 116 to 31. ACI fixes the
  season: calibration on November–December does not represent summer, and per-building CQR
  drops to about 88.8% from June to September, while ACI stays between 89.3% and 90.5% in
  every month.
- **The price is width.** ACI intervals are about 10% wider than CQR (0.524 vs 0.474).
- **Heavy tails.** On the 100-building sample, 95% intervals were about 4.3 times wider
  than 50% intervals; a normal distribution would need about 2.9 times.

**Limitations**

- ACI's per-building coverage is close to 90% partly by construction: it steers each
  building's long-run miss rate. The informative checks are by hour and by month.
- ACI updates once per day (the forecasts are made at midnight for the next day), so it
  reacts to a bad day only afterwards. When it asks for more than the largest calibration
  error, the interval is capped there instead of becoming infinite.
- Apart from ACI, no method has a coverage guarantee on 2017: the guarantee of split
  conformal needs calibration and test data to be exchangeable, and time series are not.
- One seed; measured temperature instead of a weather forecast (as in M1).

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. Run `notebooks/01_explore.ipynb`. It writes the cleaned data to `data/processed/`.
2. `python scripts/train_gbm.py`: gradient boosting on all test hours, 5 seeds.
3. `python scripts/train_nn.py val`: validation run to choose the number of MLP epochs.
4. `python scripts/train_nn.py`: MLP and gradient boosting on the same rows, 5 seeds each.
5. `python scripts/plot_forecast_week.py`: the forecast figure.
6. `python scripts/m2_intervals.py`: M2 intervals, 5 methods × 4 levels, and the M2 figure.

Every experiment, including the ones that didn't work, is recorded in `EXPERIMENT_LOG.md`.

## Repository

```
configs/        building list
notebooks/      01_explore (data cleaning), 02_first_model (first models)
src/features.py feature table for one building
scripts/        training and plotting scripts
results/        tables (per-building MAE) and figures
data/           raw and processed data (not tracked by git)
```

## Data citation

Miller, C., Kathirgamanathan, A., Picchetti, B. et al. *The Building Data Genome Project 2,
energy meter data from the ASHRAE Great Energy Predictor III competition.*
Scientific Data 7, 368 (2020).
