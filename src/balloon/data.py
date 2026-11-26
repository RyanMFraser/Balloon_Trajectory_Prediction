"""Data for the FCN3 -> hourly wind model. Each window is one init: ERA5 at t0, the FCN3 ensemble at t0+6h,
and the ERA5 truth at t0+1..6h.

You only need these:

    from balloon.data import TARGET_LEVELS, make_patch_dataset, load_winds, denormalize_y

    # Patches for training a model (normalized). X is (N, C, 64, 64), Y is (N, 6 hours, 2 u/v, levels, 48, 48)
    train = make_patch_dataset("train", window_size=64, label_size=48, n_per_window=8,
                               member_mode="meanstd", target_levels=TARGET_LEVELS)
    val = make_patch_dataset("val", window_size=64, label_size=48, stride=48,
                             member_mode="meanstd", target_levels=TARGET_LEVELS)
    Y_ms = denormalize_y(val["Y"], val["target_levels"])          # back to m/s

    # Full domain in m/s, for baselines and for scoring with balloon.metrics.evaluate
    winds = load_winds("test")        # dict with "t0", "t6" and "truth"

Splits are "train", "val" and "test" (see SPLIT_RULES). Everything else in this file builds the cached window
files and normalization stats; it runs on its own the first time. Details are in docs/data.md.
"""
from pathlib import Path
import hashlib
import json
import re

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "data" / "fcn3_region"
OUT_DIR = REPO / "data" / "processed"
WINDOW_DIR = OUT_DIR / "windows"
STATS_PATH = OUT_DIR / "stats.nc"

HOURS = [1, 2, 3, 4, 5, 6]
WINDOW_LEAD = 6
TARGET_LEVELS = [30, 50, 70, 100, 125, 150, 175]
FCN3_TRAIN_END = pd.Timestamp("2016-12-31T23:59")  # FCN3 trained on ERA5 1980-2016 (arXiv:2507.12144)

SPLIT_RULES = {
    "test_years": [2021],
    "val_days": list(range(26, 32)),
    "buffer_days": [24, 25],
}


# ---------- raw months and splits ----------

def discover_months(raw_dir=RAW_DIR):
    fcn3 = {_month_of(p, "fcn3") for p in raw_dir.glob("fcn3_????-??.nc")}
    era5 = {_month_of(p, "era5_hourly") for p in raw_dir.glob("era5_hourly_????-??.nc")}
    if fcn3 ^ era5:
        print("Skipping months without both files:", sorted(fcn3 ^ era5))
    return sorted(fcn3 & era5)


def _month_of(path, prefix):
    return re.search(rf"{prefix}_(\d{{4}}-\d{{2}})\.nc$", path.name).group(1)


def assign_split(init, rules=SPLIT_RULES):
    """'train', 'val', 'test', or None for buffer days (dropped)."""
    init = pd.Timestamp(init)
    if init.year in rules["test_years"]:
        if init <= FCN3_TRAIN_END:
            raise ValueError(f"test year {init.year} overlaps FCN3's training data (<= 2016)")
        return "test"
    if init <= FCN3_TRAIN_END:
        return "train"
    if init.day in rules["buffer_days"]:
        return None
    if init.day in rules["val_days"]:
        return "val"
    return "train"


def window_index(rules=SPLIT_RULES, months=None):
    """One row per extracted window: month, init, split."""
    rows = []
    for month in months or discover_months():
        with open_windows(month) as ds:
            inits = ds.init.values
        rows += [dict(month=month, init=pd.Timestamp(t), split=assign_split(t, rules)) for t in inits]
    return pd.DataFrame(rows)


def split_windows(split, rules=SPLIT_RULES, index=None):
    index = window_index(rules) if index is None else index
    windows = index[index.split == split]
    if windows.empty:
        raise ValueError(f"no windows in split {split!r}")
    return windows


# ---------- stage 1: extract windows (cached) ----------

def extract_all(months=None, overwrite=False):
    return [extract_month(m, overwrite) for m in (months or discover_months())]


def extract_month(month, overwrite=False):
    out = WINDOW_DIR / f"{month}.nc"
    if out.exists() and not overwrite:
        return out
    WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    # Close files explicitly: netCDF4/HDF5 can segfault if the GC closes one during another HDF5 call.
    with xr.open_dataset(RAW_DIR / f"fcn3_{month}.nc") as f, \
            xr.open_dataset(RAW_DIR / f"era5_hourly_{month}.nc") as e:
        n = _write_windows(f.fcn3, e.era5, month, out)
    print(f"{month}: wrote {n} windows -> {out.relative_to(REPO)}")
    return out


def open_windows(month):
    """Use as `with open_windows(m) as ds:` so the file is closed deterministically."""
    return xr.open_dataset(WINDOW_DIR / f"{month}.nc")


def _complete_inits(fcn3, era5):
    era5_times = set(pd.to_datetime(era5.time.values))
    needed = [0, *HOURS]
    return [pd.Timestamp(t) for t in fcn3.init.values
            if all(pd.Timestamp(t) + pd.Timedelta(h, "h") in era5_times for h in needed)]


def _write_windows(fcn3, era5, month, out):
    if not (np.array_equal(fcn3.lat, era5.lat) and np.array_equal(fcn3.lon, era5.lon)):
        raise ValueError(f"{month}: FCN3 and ERA5 grids differ")
    inits = _complete_inits(fcn3, era5)
    tmp = out.with_suffix(".nc.partial")
    with netCDF4.Dataset(tmp, "w") as nc:
        variables = _create_window_file(nc, fcn3, era5, month)
        for k, t in enumerate(inits):
            _write_one_window(nc, variables, k, t, fcn3.sel(init=t), era5)
    tmp.rename(out)
    return len(inits)


def _create_window_file(nc, fcn3, era5, month):
    shared = [int(l) for l in fcn3.level.values]
    dims = dict(init=None, member=fcn3.sizes["member"], hour=len(HOURS), uv=2, zt=2, var4=4,
                level=era5.sizes["level"], level5=len(shared), lat=fcn3.sizes["lat"], lon=fcn3.sizes["lon"])
    for d, n in dims.items():
        nc.createDimension(d, n)
    nc.createVariable("init", "f8", ("init",)).units = "hours since 1970-01-01"
    coords = [("lat", fcn3.lat.values), ("lon", fcn3.lon.values), ("level", era5.level.values),
              ("level5", np.array(shared)), ("hour", np.array(HOURS)), ("member", fcn3.member.values)]
    for name, vals in coords:
        nc.createVariable(name, vals.dtype, (name,))[:] = vals

    def field(name, dims):
        chunks = [1] + [nc.dimensions[d].size for d in dims[1:]]
        return nc.createVariable(name, "f4", dims, zlib=True, complevel=1, chunksizes=chunks)

    nc.setncatts(dict(month=month, var4="u,v,z,t", uv="u,v", zt="z,t",
                      description="Phase-1 windows: ERA5 t0 + FCN3 lead-6 ensemble -> ERA5 t0+1..6h"))
    return dict(t0_uv=field("t0_uv", ("init", "uv", "level", "lat", "lon")),
                t0_zt=field("t0_zt", ("init", "zt", "level5", "lat", "lon")),
                t6=field("t6", ("init", "member", "var4", "level5", "lat", "lon")),
                y=field("y", ("init", "hour", "uv", "level", "lat", "lon")))


def _write_one_window(nc, variables, k, t, forecast, era5):
    nc["init"][k] = (t - pd.Timestamp("1970-01-01")) / pd.Timedelta(1, "h")
    variables["t0_uv"][k] = era5.sel(time=t).values
    variables["t0_zt"][k] = forecast.sel(lead=0, var=["z", "t"]).isel(member=0).values
    variables["t6"][k] = forecast.sel(lead=WINDOW_LEAD, var=["u", "v", "z", "t"]).values
    variables["y"][k] = era5.sel(time=[t + pd.Timedelta(h, "h") for h in HOURS]).values


# ---------- full-domain winds for evaluation ----------

def load_winds(split, rules=SPLIT_RULES):
    """u/v of every window in a split, full domain, in m/s.

    Returns a dict of DataArrays:
      t0    (init, var, level, lat, lon)          ERA5 at t0, all 10 levels
      t6    (init, member, var, level, lat, lon)  FCN3 at +6h, its 5 levels
      truth (init, hour, var, level, lat, lon)    ERA5 at t0+1..6h, TARGET_LEVELS
    """
    months = [_load_month_winds(m, group.init.values)
              for m, group in split_windows(split, rules).groupby("month")]
    return {key: xr.concat([m[key] for m in months], "init") for key in ("t0", "t6", "truth")}


def _load_month_winds(month, inits):
    with open_windows(month) as ds:
        ds = ds.sel(init=inits)
        winds = dict(t0=ds.t0_uv.rename(uv="var"),
                     t6=ds.t6.isel(var4=[0, 1]).rename(var4="var", level5="level"),
                     truth=ds.y.sel(level=TARGET_LEVELS).rename(uv="var"))
        return {key: arr.load().assign_coords(var=["u", "v"]) for key, arr in winds.items()}


# ---------- normalization statistics (train split only) ----------

def load_stats(index=None, rules=SPLIT_RULES):
    """Per-field mean/std ('u50', 'z100', ...) from the train windows' t0, cached in stats.nc."""
    index = window_index(rules) if index is None else index
    signature, n_train = _train_signature(index)
    cached = _cached_stats(signature)
    if cached is not None:
        return cached
    stats = _compute_stats(index[index.split == "train"])
    stats.attrs.update(train_signature=signature, n_train_windows=n_train)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_netcdf(STATS_PATH)
    print(f"computed stats from {n_train} train windows")
    return stats


def _train_signature(index):
    train = index[index.split == "train"].init.astype(str).tolist()
    return hashlib.sha1(json.dumps(train).encode()).hexdigest(), len(train)


def _cached_stats(signature):
    if not STATS_PATH.exists():
        return None
    with xr.open_dataset(STATS_PATH) as f:
        stats = f.load()
    return stats if stats.attrs.get("train_signature") == signature else None


def _compute_stats(train_windows):
    sums, squares, count = {}, {}, 0
    for month, group in train_windows.groupby("month"):
        with open_windows(month) as ds:
            for t in group.init.values:
                w = ds.sel(init=t)
                for name, arr in _named_t0_fields(w["t0_uv"].values, w["t0_zt"].values, ds).items():
                    arr = arr.astype(np.float64)
                    sums[name] = sums.get(name, 0.0) + arr.sum()
                    squares[name] = squares.get(name, 0.0) + (arr ** 2).sum()
                count += arr.size
    names = list(sums)
    mean = np.array([sums[k] / count for k in names])
    std = np.sqrt(np.array([squares[k] / count for k in names]) - mean ** 2)
    return xr.Dataset({"mean": ("field", mean), "std": ("field", std)}, coords={"field": names})


def _named_t0_fields(t0_uv, t0_zt, ds):
    fields = {f"{v}{int(lev)}": t0_uv[i, j] for i, v in enumerate("uv") for j, lev in enumerate(ds.level.values)}
    fields |= {f"{v}{int(lev)}": t0_zt[i, j] for i, v in enumerate("zt") for j, lev in enumerate(ds.level5.values)}
    return fields


def _stat(stats, field):
    return float(stats["mean"].sel(field=field)), float(stats["std"].sel(field=field))


def _normalize(arr, stats, field, center=True):
    mean, std = _stat(stats, field)
    return (arr - mean) / std if center else arr / std


def denormalize_y(Y, target_levels, stats=None):
    """Normalized targets (N, 6, 2, L, h, w) -> m/s."""
    stats = load_stats() if stats is None else stats
    out = Y.copy()
    for i, v in enumerate("uv"):
        for j, lev in enumerate(target_levels):
            mean, std = _stat(stats, f"{v}{lev}")
            out[:, :, i, j] = out[:, :, i, j] * std + mean
    return out


# ---------- stage 2: patch datasets ----------

POSITION_CHANNELS = ["sin_lat", "sin_lon", "cos_lon"]


def make_patch_dataset(split, window_size, label_size=None, n_per_window=None, stride=None,
                       member_mode="stack", target_levels=None, normalize=True, seed=0,
                       rules=SPLIT_RULES, max_gb=6.0):
    """Cut patches from every window of a split. See notebooks/data_pipeline.ipynb for the full description.

    Returns dict(X (N, C, k, k), Y (N, 6, 2, L, l, l), meta, x_names, target_levels, label_offset, normalized).
    """
    k, l, offset = _patch_geometry(window_size, label_size)
    index = window_index(rules)
    windows = split_windows(split, rules, index)
    stats = load_stats(index, rules) if normalize else None
    rng = np.random.default_rng(seed)
    layout = _window_layout(windows.month.iloc[0])
    target_levels = layout["levels"] if target_levels is None else list(target_levels)
    level_idx = [layout["levels"].index(v) for v in target_levels]

    n_pos = len(_positions(layout["rows"], layout["cols"], k, n_per_window, stride, rng)[0])
    n_chan = _n_channels(member_mode, layout["members"])
    N = len(windows) * n_pos
    gb = N * 4 * (n_chan * k * k + len(HOURS) * 2 * len(target_levels) * l * l) / 1e9
    if gb > max_gb:
        raise MemoryError(f"dataset would be {gb:.1f} GB (> max_gb={max_gb}); use fewer/smaller patches")

    X = np.empty((N, n_chan, k, k), np.float32)
    Y = np.empty((N, len(HOURS), 2, len(target_levels), l, l), np.float32)
    meta, n = [], 0
    for month, group in windows.groupby("month", sort=True):
        with open_windows(month) as ds:
            _check_members(ds, month, member_mode, layout["members"])
            for t in group.init.values:
                x_full, y_full, x_names = _window_fields(ds.sel(init=t).load(), layout, member_mode, stats)
                I, J = _positions(layout["rows"], layout["cols"], k, n_per_window, stride, rng)
                X[n:n + len(I)] = _cut(x_full, I, J, k)
                Y[n:n + len(I)] = _cut(y_full[:, :, level_idx], I + offset, J + offset, l)
                meta += [dict(month=month, init=pd.Timestamp(t), split=split, i=i, j=j) for i, j in zip(I, J)]
                n += len(I)
    print(f"{split}: {len(windows)} windows x {n_pos} patches = {N} examples, "
          f"X {X.shape}, Y {Y.shape}, {gb:.2f} GB")
    return dict(X=X, Y=Y, meta=pd.DataFrame(meta), x_names=x_names, target_levels=target_levels,
                label_offset=offset, normalized=normalize)


def _patch_geometry(window_size, label_size):
    label_size = window_size if label_size is None else label_size
    if label_size > window_size or (window_size - label_size) % 2:
        raise ValueError("label_size must be <= window_size with an even difference")
    return window_size, label_size, (window_size - label_size) // 2


def _window_layout(month):
    with open_windows(month) as ds:
        return dict(levels=[int(v) for v in ds.level.values], levels5=[int(v) for v in ds.level5.values],
                    members=ds.sizes["member"], rows=ds.sizes["lat"], cols=ds.sizes["lon"])


def _n_channels(member_mode, n_members):
    t6 = {"stack": 20 * n_members, "meanstd": 40, "mean": 20}[member_mode]
    return 30 + t6 + len(POSITION_CHANNELS)


def _check_members(ds, month, member_mode, n_members):
    if member_mode == "stack" and ds.sizes["member"] != n_members:
        raise ValueError(f"{month} has {ds.sizes['member']} members, expected {n_members}; "
                         "use member_mode='meanstd' or 'mean' to mix months")


def _positions(n_rows, n_cols, k, n_per_window, stride, rng):
    """Top-left corners of patches: random (n_per_window) or a regular grid touching the far edges (stride)."""
    if (n_per_window is None) == (stride is None):
        raise ValueError("pass exactly one of n_per_window (random) or stride (grid)")
    if n_per_window is not None:
        return rng.integers(0, n_rows - k + 1, n_per_window), rng.integers(0, n_cols - k + 1, n_per_window)
    rows = sorted(set(range(0, n_rows - k + 1, stride)) | {n_rows - k})
    cols = sorted(set(range(0, n_cols - k + 1, stride)) | {n_cols - k})
    I, J = np.meshgrid(rows, cols, indexing="ij")
    return I.ravel(), J.ravel()


def _cut(field, I, J, size):
    """Patches of the last two axes at corners (I, J), with the patch axis moved to the front."""
    windows = np.lib.stride_tricks.sliding_window_view(field, (size, size), axis=(-2, -1))
    patches = windows[..., I, J, :, :]
    return np.moveaxis(patches, -3, 0)


def _window_fields(w, layout, member_mode, stats):
    """(C, H, W) inputs, (6, 2, 10, H, W) targets and channel names for one window."""
    channels = _input_channels(w, layout["levels"], layout["levels5"], member_mode)
    arrays = [arr if stats is None else _normalize(arr, stats, field, center) for _, field, arr, center in channels]
    arrays += _position_fields(w.lat.values, w.lon.values)
    names = [name for name, *_ in channels] + POSITION_CHANNELS
    y = w["y"].values if stats is None else _normalize_targets(w["y"].values, layout["levels"], stats)
    return np.stack(arrays).astype(np.float32), y.astype(np.float32), names


def _input_channels(w, levels, levels5, member_mode):
    """(channel name, stats field, array, subtract mean?) for every non-position input channel."""
    t0_uv, t0_zt = w["t0_uv"].values, w["t0_zt"].values
    channels = [(f"t0_{v}{lev}", f"{v}{lev}", t0_uv[i, j], True)
                for i, v in enumerate("uv") for j, lev in enumerate(levels)]
    channels += [(f"t0_{v}{lev}", f"{v}{lev}", t0_zt[i, j], True)
                 for i, v in enumerate("zt") for j, lev in enumerate(levels5)]
    for tag, t6, center in _member_groups(w["t6"].values, member_mode):
        channels += [(f"t6_{tag}_{v}{lev}", f"{v}{lev}", t6[i, j], center)
                     for i, v in enumerate("uvzt") for j, lev in enumerate(levels5)]
    return channels


def _member_groups(t6, member_mode):
    if member_mode == "stack":
        return [(f"m{k}", t6[k], True) for k in range(t6.shape[0])]
    if member_mode == "meanstd":
        return [("mean", t6.mean(0), True), ("std", t6.std(0, ddof=1), False)]
    if member_mode == "mean":
        return [("mean", t6.mean(0), True)]
    raise ValueError(f"member_mode must be 'stack', 'meanstd' or 'mean', got {member_mode!r}")


def _position_fields(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)[:, None] * np.ones((1, len(lon_deg)))
    lon = np.deg2rad(lon_deg)[None, :] * np.ones((len(lat_deg), 1))
    return [np.sin(lat), np.sin(lon), np.cos(lon)]


def _normalize_targets(y, levels, stats):
    y = y.copy()
    for i, v in enumerate("uv"):
        for j, lev in enumerate(levels):
            y[:, i, j] = _normalize(y[:, i, j], stats, f"{v}{lev}")
    return y
