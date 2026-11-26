# Data

Raw data for the FCN3 → hourly wind refinement model.

with `xarray.open_dataset` and index with `.sel(...)`. All 17 files were verified against Drive checksums on
2026-11-23.

```
data/
├── fcn3_region/              ← training / evaluation data
│   ├── fcn3_YYYY-MM.nc         FCN3 4-member forecasts (input)
│   └── era5_hourly_YYYY-MM.nc  hourly ERA5 winds (ground truth)
├── fcn3_global/              ← one global init per month, sanity checks only
│   ├── fcn3_global_YYYY-MM-15T00.nc
│   └── era5_global_YYYY-MM-15T00.nc
└── loon/loon_region_2020.parquet   real Loon balloon flights (later phase)
```
Months available: **2020-01, 2020-07, 2021-01, 2021-07** (one Southern Hemisphere summer and one winter per year).

## Region and grid (all regional files)
- **10°N to 40°S, 260°E to 320°E** (eastern Pacific / South America), 0.25° → **201 lat × 241 lon**.
- `lat` is **descending** (10 → -40) and `lon` is 0–360 (not -180–180).
- The FCN3 and ERA5 regional grids are identical (checked), so no regridding is needed.

## `fcn3_region/fcn3_YYYY-MM.nc` (model input)
Variable `fcn3`, float32, dims:

| dim | size | values |
|---|---|---|
| `init` | 62 | forecast start times, 00Z and 12Z every day of the month (no gaps) |
| `member` | 4 | ensemble members 0–3 (differ only in FCN3's internal noise) |
| `lead` | 9 | 0, 6, 12, …, 48 hours |
| `var` | 4 | `u`, `v` (m/s), `z` (geopotential, m²/s²), `t` (K) |
| `level` | 5 | 50, 100, 150, 200, 250 hPa |
| `lat`, `lon` | 201, 241 | |

Full shape `(62, 4, 9, 4, 5, 201, 241)`, ~9 GB uncompressed per month.

- **Lead 0 is the ERA5 initial state.** All 4 members are identical there and match ERA5 exactly. At lead 0 there is
  effectively **1** sample, not 4.
- Members diverge from lead 6 onward. The maximum pointwise member difference at +6h is ~14–16 m/s.
- `z` and `t` are not targets, but they are useful extra input channels.

## `fcn3_region/era5_hourly_YYYY-MM.nc` (ground truth)
Variable `era5`, float32, dims `(time=793, var=2, level=10, lat=201, lon=241)`.
- `time`: hourly, from the 1st 00Z to the 3rd of the next month 00Z (793 h, no gaps). That covers every valid
  time up to the last init + 48h.
- `var`: `u`, `v` (m/s) only.
- `level`: **30, 50, 70, 100, 125, 150, 175, 200, 225, 250 hPa**.

⚠️ **Level mismatch:** ERA5 has 10 levels but FCN3 only has 5 of them (**50/100/150/200/250**). The other levels
(30, 70, 125, 175 and 225 hPa) have no FCN3 input. The Loon balloons fly at 48–144 hPa (median 77), mostly
*between* FCN3 levels, so 70 and 125 hPa matter downstream. Decide in issue #4 whether the model predicts 5 levels
or all 10. Predicting all 10 means also interpolating vertically, and 30 hPa means extrapolating above FCN3's top
level.

## `fcn3_global/` (sanity checks only)
- `fcn3_global_*`: `(member=4, lead=9, channel=72, lat=721, lon=1440)`. All 72 FCN3 channels, global.
  Attributes include `seed`. The regional crop matches `fcn3_region` exactly.
- `era5_global_*`: `(time=5, channel=72, 721, 1440)`. ERA5 at 00/12Z only, for leads 0–48.
- 72 channels: `u10m v10m u100m v100m t2m msl tcwv`, plus `u v z t q` at 13 levels (50 to 1000 hPa).

## `loon/loon_region_2020.parquet`
468,551 rows × `flight_id, time, lat, lon, pressure, acs, u, v`. 169 flights across all of 2020, all inside the
region, sampled every ~10 min. Pressure is 47.6–144.5 hPa. `acs` ∈ {-1, 0, 1} probably means altitude-control
state (descending/holding/ascending); **unconfirmed**. Jan 2020 has 41 flights and Jul 2020 has 36. This is
in-situ wind and position data for later trajectory validation, not part of phase-1 training.

---

## How many FCN3 samples per time step

**Per lead time**, for each month:

| lead (h) | inits | members | forecasts | distinct |
|---|---|---|---|---|
| 0 | 62 | 4 | 248 | 62 (members identical = ERA5) |
| 6, 12, …, 48 (each) | 62 | 4 | 248 | 248 |

Across all 4 months: **248 inits → 992 member forecasts per lead** (lead 0: 248 distinct states).

**Per valid (wall-clock) time.** Forecasts from different inits overlap. For example, 2020-01-03 00Z is forecast by
the 00Z inits of Jan 1, 2 and 3 (leads 48, 24 and 0) and by the 12Z inits of Jan 1 and 2 (leads 36 and 12).

| valid hour | FCN3 member-forecasts | from |
|---|---|---|
| 00Z, 12Z | 20 (16 + 4 identical lead-0 copies) | leads 0/12/24/36/48 × 4 members |
| 06Z, 18Z | 16 | leads 6/18/30/42 × 4 members |
| hourly in between (01–05Z, …) | **0** (ERA5 truth only) | — |

The first ~2 days of each month have fewer, because earlier inits are missing.


## Train / validation / test split

**Principle (team decision):** training on dates inside FCN3's training period is acceptable, since we are learning
finer time resolution and not re-forecasting. **Testing must be outside it.**

- FCN3's training data was **ERA5 1980–2016**; fine-tuning used 2012–2016. FCN3's own evaluation used 2020.
- **All local data (2020, 2021) is outside FCN3's training window.** Every month here is valid for testing, and
  nothing we have touches the FCN3 training years. If more training data is needed, it could be generated from
  1980–2016 (allowed under the principle above) or from 2017–2019.

**Proposed split (by contiguous time blocks; never random samples):**

| split | data | inits | phase-1 windows |
|---|---|---|---|
| **train** | 2020-01 and 2020-07, inits on days 1–23 | 46 × 2 = 92 | 92 (× 4 members) |
| *buffer* | days 24–25 (dropped) | 4 × 2 | — |
| **validation** | 2020-01 and 2020-07, inits on days 26–31 | 12 × 2 = 24 | 24 |
| **test** | all of 2021-01 and 2021-07 | 124 | 124 |


⚠️ **The stratospheric winds reverse between years.** The quasi-biennial oscillation (QBO) flips tropical
stratospheric winds roughly every 14 months, and it shows up clearly here:

| mean u (m/s) | 30 hPa | 50 hPa | 70 hPa |
|---|---|---|---|
| 2020-07 | +7.4 | −1.0 | +2.7 |
| 2021-07 | −2.5 | +7.8 | +8.6 |

So the 2021 test set has a different wind regime aloft than the 2020 training set. This makes the split a real
generalization test, but it will also inflate the test error at upper levels. Report per-level metrics, and keep
this in mind before blaming the model. Training on several years would cover both QBO phases.

## Processed data (`notebooks/data_pipeline.ipynb`)
- `data/processed/windows/YYYY-MM.nc`: the full-domain phase-1 windows per month (`t0_uv`, `t0_zt`, `t6`, `y`).
  ~1.9 GB per month, ~3 min to build all 4. It doesn't depend on the split, and existing months are skipped.
- `data/processed/stats.nc`: per-field mean/std from the **train** windows only. It is rebuilt automatically when
  the train set changes.
- `make_patch_dataset(split, window_size, label_size, n_per_window | stride, member_mode, target_levels)` returns
  `X (N, C, k, k)`, `Y (N, 6, 2, L, l, l)` and `meta`. `window_size=1` is pointwise.
- Split rules live in `SPLIT_RULES` in the notebook. To add data, drop new month pairs into `data/fcn3_region/`
  and re-run.

## Per-level scale (ERA5 u/v std, m/s; varies by month)
u std grows from ~6–18 at 30–70 hPa to ~14–23 at 200–250 hPa. v std grows from ~3–4 to ~10–13. Normalize per
variable **and** per level.
