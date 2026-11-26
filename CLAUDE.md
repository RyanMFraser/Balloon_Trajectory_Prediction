# CLAUDE.md — Balloon Trajectory Prediction (Deep Learning class project)

## Project goal (phase 1)

Learn the **hourly states between FourCastNet 3 (FCN3) 6-hour steps**:

- **Input:** ERA5 state at `t=0` + `n` stochastic FCN3 ensemble samples of `t=6h` (`t6_1 … t6_n`).
- **Output:** `t=1,2,3,4,5h` **and a corrected "true" `t=6h`**. All targets are ERA5 hourly reanalysis.
- Downstream use, which is out of scope for now: winds drive a weather balloon trajectory model, which is why the
  targets are **u/v winds at stratospheric and upper-tropospheric levels** over a regional box.

This is **temporal interpolation plus endpoint correction / ensemble post-processing**. It is not forecasting. The
model must (a) fill in sub-6h dynamics and (b) turn an uncertain, biased ensemble at t=6 into a better estimate of
the truth.

### Open GitHub issues (phase 1 work list)
| # | Title | Notes |
|---|---|---|
| 1 | Train/Test/Validation pipeline + docs | Save datasets once and document date ranges. Avoid leakage (see "Data splits"). |
| 2 | Baseline model on test set | Interpolate over FCN3 data. Score with MSE on u, v across interpolated hours and levels. |
| 3 | Architecture research/docs | Candidates: U-Net, Bi-directional LSTM, plus whatever else we find (see below). |
| 4 | Input/output specification | Exact tensor shapes and features. **Agreed framing: medium-sized spatial patches** (see "Patch-based framing"). |
| 5 | Loss metrics | MSE is likely. Also consider latitude weighting and probabilistic losses. |

---

## Local data (`data/`, gitignored, ~60 GB; never commit these)

**See `data/data.md`** for full shapes, per-lead sample counts, baseline numbers and the agreed train/val/test split.

Source: a teammate's shared Google Drive folder `fcn3`. `data/` mirrors its layout:
`data/fcn3_region/` (regional FCN3 + hourly ERA5 truth), `data/fcn3_global/` (one global init per month), and
`data/loon/loon_region_2020.parquet`. The Loon file holds real balloon flights, columns `flight_id, time, lat,
lon, pressure, acs, u, v`: 468k rows, 169 flights, all of 2020, inside the project region, ~10-min sampling.
Pressure is 48–144 hPa (median 77), so the flights mostly sit **between** FCN3's 50/100/150 hPa levels. `acs` is
-1/0/1 and probably means altitude-control state (descending/holding/ascending). Jan 2020 has 41 flights and Jul
2020 has 36. These are in-situ winds, useful later for validating trajectories and winds. Months available: **2020-01, 2020-07,
2021-01, 2021-07**, with the same file naming for each.

Sync and verify with rclone. The remote `gdrive` is read-only and configured in `~/.config/rclone/rclone.conf`:
`rclone copy gdrive:fcn3 data/ --drive-shared-with-me -P`, then `rclone check gdrive:fcn3 data/ --drive-shared-with-me`.
Use rclone rather than browser downloads, because a browser download once silently truncated `fcn3_2020-01.nc`.

All files are NetCDF4/HDF5, zlib-compressed. Open them lazily with `xarray.open_dataset` and pull only the slices
you need, because the global files are about 11 GB each. Generated with **earth2studio 0.19.0**.

### Regional files (the actual training/eval data)
Region: **10°N–40°S, 260°–320°E** (eastern Pacific / South America), 0.25°, grid **201 lat × 241 lon**. Lat is
descending (10 → -40) and lon is 0–360. The lat/lon grids of the FCN3 and ERA5 regional files match exactly.

| File | Var | Dims | Contents |
|---|---|---|---|
| `fcn3_2020-07.nc` | `fcn3` | `(init=62, member=4, lead=9, var=4, level=5, lat, lon)` | FCN3 4-member forecasts. Inits at 00Z/12Z every day of July 2020. lead = 0,6,…,48 h. var = u,v,z,t. levels = **50,100,150,200,250 hPa** |
| `fcn3_2020-01.nc` (and `fcn3_2021-*.nc`) | `fcn3` | same layout | Jan 2020 (and Jan/Jul 2021) |
| `era5_hourly_2020-01.nc` / `-07.nc` | `era5` | `(time=793, var=2, level=10, lat, lon)` | **Ground truth**: hourly ERA5 u, v from the 1st of the month to the 3rd of the next month. levels = **30,50,70,100,125,150,175,200,225,250 hPa** |

### Global files (one init each, for sanity checks)
| File | Dims | Notes |
|---|---|---|
| `fcn3_global_2020-{01,07}-15T00.nc` | `(member=4, lead=9, channel=72, 721, 1440)` | All 72 FCN3 channels. Has `seed` attr. The note says the regional crop matches the stored regional data exactly. |
| `era5_global_2020-{01,07}-15T00.nc` | `(time=5, channel=72, 721, 1440)` | ARCO ERA5 at 00/12Z only, for lead 0–48h. |

### Gotchas found while inspecting the data
- **Level mismatch.** ERA5 truth has 10 levels but FCN3 only has 5 (50/100/150/200/250). The levels 30, 70, 125,
  175 and 225 hPa are **not FCN3 channels at all**, and 30 hPa sits above FCN3's top level. Issue #4 must decide
  between two options: (a) only predict the 5 shared levels, or (b) also do vertical interpolation/extrapolation
  to the extra levels. Option (b) is harder, especially at 30 hPa.
- **FCN3 lead 0 == ERA5 exactly** (RMSE 0.000 checked on 2020-07-11T00), so the lead-0 input is just the ERA5 state.
- FCN3 also gives `z` and `t` at the 5 levels. These are useful extra input channels (geostrophic/thermal-wind
  information), even though only u/v are targets.
- The ensemble has only **4 members**. FCN3 was trained with 16 and the paper uses 50 for evaluation, so 4-member
  spread/mean statistics are noisy.
- Leads up to 48 h exist. Phase 1 only needs lead 0→6. The other windows (6→12, …) are extra training samples but
  have different error statistics, because their start state is an FCN3 forecast rather than ERA5.

### Reference numbers (one init, 2020-07-11T00, u+v over the 5 shared levels, region; indicative only)
| Quantity | RMSE (m/s) |
|---|---|
| Single FCN3 member at +6h vs ERA5 | 1.91 |
| FCN3 4-member mean at +6h vs ERA5 | 1.56 |
| FCN3 ensemble spread (std) at +6h | 0.97 |
| Linear interp between **true** ERA5 0h & 6h, at h = 1/2/3/4/5 | 0.88 / 1.33 / 1.45 / 1.29 / 0.83 |
| FCN3 member at +12h | 2.75 |

**Implication:** error at the FCN3 t=6 endpoint is about as large as, or larger than, the interpolation error
itself. The biggest gains likely come from **correcting t=6** (ensemble mean beats a single member, and spread <
error means the ensemble is under-dispersive at this point). Recompute these over the full test set in issue #2.

---

## FourCastNet 3 (FCN3): key facts

Paper: Bonev, Kurth et al., *"FourCastNet 3: A geometric approach to probabilistic machine-learning weather
forecasting at scale"*, arXiv:2507.12144 (Jul 2025). Code: NVIDIA `makani`. Weights: NGC / HuggingFace
`nvidia/fourcastnet3` (Apache-2.0, ~710M params). Run via **earth2studio** `earth2studio.models.px.FCN3`.

- **Grid/step:** global 0.25° equiangular, 721×1440, **6-hour** autoregressive step, stable out to 60 days.
- **72 channels:** surface `u10m v10m u100m v100m t2m msl tcwv`, plus `u v z t q` at 13 levels
  `50 100 150 200 250 300 400 500 600 700 850 925 1000` hPa. The channel order matches the global files here.
- **Architecture:** fully convolutional **spherical neural operator**, made of an encoder, 8 neural-operator
  blocks and a decoder. It uses spherical signal processing (spherical harmonic transforms and discrete-continuous
  convolutions from `torch-harmonics`), predicts the next state directly rather than a tendency, and keeps water
  channels positive with a spline output activation. Cos zenith angle is an auxiliary input.
- **Stochasticity:** a hidden-Markov formulation. Each step takes the previous state plus a **latent noise field
  z_n**, drawn from diffusion processes on the sphere with several length and time scales. Ensemble members differ
  **only** in noise realization. There is no input perturbation, and each member is generated in one step (unlike
  diffusion models such as GenCast).
- **Training loss:** ensemble **CRPS** (pointwise/spatial) + a **spectral CRPS** term that matches the
  distribution of spherical-harmonic coefficients at all wavelengths. This keeps spectra realistic (no blurring).
- **Training data:** ERA5 **1980–2016**. Pre-training used hourly-initialized 6h pairs. Fine-tuning used 6-hourly
  data from 2012–2016. **Evaluation was on 2020**, with 12-hourly inits. 2017–2019 were not used for training
  (probably validation, though the paper doesn't say).
- **Skill:** beats IFS-ENS, matches GenCast, and is 8–60× faster. A 15-day forecast takes ~60 s on one H100.
  Spread-skill ratio approaches 1 for 50 members, but it is **slightly over-dispersive up to 24h and then
  under-dispersive**.
- **Requirements:** ≥ ~60 GB GPU memory for inference.

### Data split implications (issue #1)
- Issue #1 says FCN3 was trained "up till 2020". The paper actually says **training ended at 2016**. 2017–2019 are
  out-of-sample for FCN3's weights, but they may have been used for its validation/model selection. **2020+ is the
  safest test/val period** and matches FCN3's own evaluation year.
- Our own model's train/val/test must also be split by **contiguous time blocks** (e.g., whole weeks or months),
  not random samples. Consecutive inits 12h apart are highly correlated, and windows overlap at 48h lead.
- Seasonality: the data covers Jan and Jul of **2020 and 2021**. A natural split is by year (e.g., train/val on
  2020 and test on 2021), which keeps both seasons in every split. Document the choice.

---

## ERA5

- ECMWF's 5th-generation reanalysis (Copernicus C3S). It runs hourly from 1940 to present at 0.25° (~31 km), with
  37 pressure levels. It is the standard "truth" for ML weather models (WeatherBench2, FCN, GraphCast, Pangu).
- Our data came from **ARCO-ERA5** (Google, analysis-ready Zarr):
  `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` (anonymous access, us-central1). In
  earth2studio, use `earth2studio.data.ARCO`. The CDS API is the slower alternative.
- Caveats for balloons: ERA5 winds in the tropical lower stratosphere have known errors compared with
  superpressure-balloon measurements (Strateole-2, ACP 2026: only a 10–20% improvement since 2010). Median balloon
  trajectory errors after 24h are ~260 km, with a heavy tail (>600 km for 10%). ERA5 "truth" is itself imperfect,
  which caps how small our errors can get.

---

## Related work and techniques

### Most directly relevant: NVIDIA ModAFNO temporal interpolation
Leinonen & Bonev, *"Modulated Adaptive Fourier Neural Operators for Temporal Interpolation of Weather Forecasts"*,
arXiv:2410.18904. Implemented in PhysicsNeMo (`examples/weather/temporal_interpolation`) and earth2studio
(`InterpModAFNO`, which is paired with 6-hourly models to produce hourly output).
- **Inputs:** normalized states at t0 and t1 (73 vars each), sin/cos lat/lon, land-sea mask, orography, and cos
  zenith at t0, t and t1. That is 155 channels plus a **scalar target time**.
- **Time conditioning:** the normalized offset `(t-t0)/(t1-t0)` goes through a sinusoidal embedding and an MLP.
  Per-block MLPs turn that into **scale/shift (FiLM-style) modulation** inside each AFNO block. One network
  handles every intermediate hour.
- **Loss:** cos(latitude)-weighted MSE. Trained on ERA5 1980–2016 and tested on 2017.
- **Result:** ~**50% lower RMSE than linear interpolation** at intermediate hours. It avoids the "ghosting" of
  fast-moving features that linear interpolation produces. It does produce artifacts at non-trained fractional
  times. It is deterministic, and **it was only evaluated with ERA5 endpoints, not forecast endpoints**. Our
  setup (an uncertain FCN3 endpoint) is the gap we fill.
- A pretrained checkpoint could serve as a strong reference baseline, but it expects 73 global variables, not
  FCN3's 72. Check the variable compatibility before relying on it.

### Other temporal interpolation work
- **WeatherBridge** (Research Square preprint rs-11183024): compares linear interpolation against six
  architectures at 6h/12h gaps. Key finding: **models trained on ERA5 endpoints lose their advantage when given
  forecast endpoints**, and adapting to forecast data recovers only ~7% over linear. Takeaway: **train on FCN3
  endpoints, not ERA5 pairs.**
- **TemDeep** (GMD 18:2427, 2025): self-supervised temporal downscaling at arbitrary resolutions, trained without
  dense high-res ground truth.
- **Video frame interpolation** is the closest CV analogue: Super SloMo (bidirectional flow + any-time
  interpolation), RIFE, FILM, and DAIN. The flow/warping idea maps to advecting fields with the wind itself.

### Patch-based framing (team decision, issue #4)
The team considered three framings: single gridpoint, patch, and whole field. They chose **medium-sized patches**,
which give spatial context and many training examples.
- **Why not pointwise:** wind at a point between t0 and t6 depends on upwind features moving through. Those are
  only visible in neighboring cells. A pointwise model (MLP/BiLSTM on one column) is still useful as a second
  baseline, because it measures how much spatial context adds.
- **Context needed:** a feature is at most ~3h of travel from either endpoint. At 30 m/s that is ~13 cells
  (0.25°), and at 56 m/s (Loon max) ~24 cells. **Start with 64×64**, and ablate 32/96/128.
- **Training:** random crops cut on the fly from the full saved windows (don't precompute patches). Compute the
  loss on the **interior only** (e.g., the central 48×48, an 8-cell context halo). Add lat/lon (sin/cos)
  channels, since crops lose absolute position.
- **Evaluation:** tile the full 201×241 domain with overlapping patches, keep the interiors and stitch them, so
  metrics are on the same full grid as the baselines.
- **Splits stay by time.** All patches from one window go in the same split. Never split by patch.

### Candidate architectures (issue #3)
| Arch | Fit for our problem | Required changes |
|---|---|---|
| **U-Net (2D)** | Strong default for regional gridded fields (201×241). | Stack t0, the t6 members (or their mean/std) and z/t as channels. Pad/crop to a size divisible by 2^depth (e.g., 208×248 or 192×240). Either output all 6 hours × 2 vars × L levels at once, or condition on target hour via FiLM/an extra channel. |
| **Bi-directional (Conv)LSTM** | Natural for "fill a sequence between two endpoints". | A plain LSTM on 50k gridpoints is too big, so use ConvLSTM or a U-Net encoder → BiConvLSTM over the 7 time slots → decoder. Endpoints are known and the middle is masked/learned. |
| **FiLM / time-conditioned U-Net** (ModAFNO idea) | One network for any target hour and fewer params. | Sinusoidal embedding of h/6 → MLP → per-block scale/shift. |
| **Residual-on-linear-interp** | Easy win. Predict `y_h − linear(t0, mean(t6), h)`. | Wrap any of the above. Usually trains faster and never does worse than the baseline at init. |
| **Neural operator (AFNO/FNO/SFNO)** | Matches FCN3/ModAFNO. Spectral, resolution-flexible. | Regional (non-periodic) domain → padding. SFNO is for global spherical data and is overkill for a regional crop. |
| **Probabilistic head** (optional) | The output could be an ensemble rather than a point estimate. | Use a CRPS loss with noise inputs, or map each FCN3 member to its own trajectory. |

**Ensemble handling:** members are exchangeable, so the model should be **permutation-invariant** over them. Three
ways to get that: use summary statistics (mean, std); use a shared encoder per member followed by mean/max pooling
(DeepSets-style); or apply the interpolator per member and average.

### Losses and metrics (issues #2, #5)
- **Baseline metric:** MSE/RMSE on u, v, reported **per target hour (1–6) and per level**. Also report wind speed
  and vector error, since balloon drift depends on the full wind vector.
- **Latitude weighting** (∝ cos φ) is the standard in WeatherBench2. Over 10°N–40°S, cos φ ranges from 0.77 to
  1.0. The effect is mild, but it is cheap to include.
- **Normalize per variable/level** (z-score using training-set mean/std). Stratospheric and tropospheric wind
  magnitudes differ.
- **Blurring caveat:** MSE rewards smooth averages. Ensemble-mean-like outputs look good on RMSE but lose sharp
  features (jet cores). Possible additions: gradient/spectral loss terms, or a check on power spectra.
- **Probabilistic options:** use CRPS (or its fair/ensemble version) if we output samples. Use spread-skill ratio
  and rank histograms to check calibration (FCN3 uses all of these).
- **Downstream proxy** (later): trajectory error from advecting a parcel through the predicted hourly winds vs ERA5.

### Baseline (issue #2, team decision: exactly one)
Straight linear interpolation in time between ERA5 t0 and the **FCN3 ensemble mean** at t6, scored with MSE (the
loss is a parameter so it can change later). It is in `src/balloon/baselines.py`, and results are in
`docs/baselines.md`. Don't add more baselines.

---

## Conventions
- Environment: `.venv/` (Python 3.14, `pip install -r requirements.txt`). Select it as the notebook kernel.
- Code lives in the `balloon` package (`src/balloon/`, `pip install -e .`):
  - `data.py`: splits, normalization stats, `make_patch_dataset` (patches) and `load_winds` (full domain).
    Models get their data through these, never by reading raw files, so splits and normalization stay consistent.
    `notebooks/data_pipeline.ipynb` runs and explains it (issue #1).
  - `baselines.py`: the one baseline (`linear_interpolation`). `metrics.py`: `evaluate(prediction, truth, loss)` returns per
    (hour, var, level) scores. `runs.py`: local run records in `runs/<name>/` (gitignored).
  - Results and reproduction steps are in `docs/baselines.md`. **Target levels: 30, 50, 70, 100, 125, 150, 175 hPa.**
- Python. Use `xarray` + `netCDF4`/`h5netcdf` to read data and PyTorch for models. Open the large files lazily and
  index with `.sel(...)`.
- Raw data lives in `data/`, which is gitignored along with `*.nc` and `*.parquet`. Save processed datasets
  (issue #1) in a separate gitignored directory (e.g., `data/processed/`), and record the split dates in the docs.

## Sources
- FCN3 paper: https://arxiv.org/abs/2507.12144 · NVIDIA blog: https://developer.nvidia.com/blog/fourcastnet-3-enables-fast-and-accurate-large-ensemble-weather-forecasting-with-scalable-geometric-ml/
- FCN3 model card: https://huggingface.co/nvidia/fourcastnet3 · NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/earth-2/models/fourcastnet3
- earth2studio FCN3: https://nvidia.github.io/earth2studio/main/modules/generated/models/px/FCN3/ · scorecard: https://nvidia.github.io/earth2studio/main/scorecard/generated/fcn3/
- ModAFNO interpolation: https://arxiv.org/abs/2410.18904 · PhysicsNeMo example: https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/weather/temporal_interpolation/README.html
- WeatherBridge preprint: https://www.researchsquare.com/article/rs-11183024/v1 · TemDeep: https://gmd.copernicus.org/articles/18/2427/2025/
- ARCO-ERA5: https://github.com/google-research/arco-era5 · https://cloud.google.com/storage/docs/public-datasets/era5
- Balloon/ERA5 stratospheric winds: https://acp.copernicus.org/articles/26/1665/2026/ · https://egusphere.copernicus.org/preprints/2025/egusphere-2025-3208/egusphere-2025-3208.pdf
