# Baseline (issue #2)

**Straight linear interpolation in time** from ERA5 at t0 to the FCN3 4-member ensemble mean at t6, for hours
1–6. The code is `linear_interpolation` in `src/balloon/baselines.py`.

FCN3 only has 50, 100 and 150 of the target levels. At 30, 70, 125 and 175 hPa, the t6 endpoint is ERA5's t0
value plus FCN3's 6h change, interpolated in log pressure. Above 50 hPa the 50 hPa change is used unchanged.

## How to reproduce

```bash
pip install -e .                                               # once
python scripts/evaluate_baseline.py --split test --loss mse    # or --split val
python scripts/compare_runs.py --split test                    # compare with other saved runs
```

The loss is chosen by name from `LOSSES` in `src/balloon/metrics.py`. Only `mse` exists for now. Results go to
`runs/baseline/` (gitignored), as `config.json` and `scores_<split>.nc`, which holds the score per (hour, var,
level).

## Results on test (2021-01 and 2021-07, 124 windows)

MSE in m²/s², averaged over u and v, on the full 201×241 domain:

| hour \ hPa | 30 | 50 | 70 | 100 | 125 | 150 | 175 | all |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.09 | 0.51 | 0.78 | 0.45 | 0.60 | 0.50 | 0.64 | 0.65 |
| 2 | 3.85 | 1.30 | 2.73 | 1.14 | 1.96 | 1.26 | 1.89 | 2.02 |
| 3 | 7.55 | 1.78 | 5.38 | 1.53 | 3.59 | 1.69 | 3.10 | 3.52 |
| 4 | 11.85 | 1.87 | 8.46 | 1.54 | 5.28 | 1.70 | 4.07 | 4.97 |
| 5 | 16.65 | 1.77 | 11.78 | 1.39 | 6.98 | 1.56 | 4.88 | 6.43 |
| 6 | 21.87 | 1.88 | 15.20 | 1.48 | 8.75 | 1.67 | 5.76 | 8.09 |

The mean over all hours and levels is **4.28**. Validation (2020, days 26–31) gives 4.41.

**The levels FCN3 doesn't have dominate the error.** By hour 6, MSE at 30 and 70 hPa is about 10× the MSE at the
FCN3 levels. The winds have vertical structure finer than FCN3's levels, so this is where a learned model has the
most room to improve.
