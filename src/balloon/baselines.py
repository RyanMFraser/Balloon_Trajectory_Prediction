"""Baseline (issue #2): straight linear interpolation in time from ERA5 at t0 to the FCN3 ensemble mean at t6.

    from balloon.data import load_winds
    from balloon.baselines import linear_interpolation
    from balloon.metrics import evaluate

    winds = load_winds("test")
    scores = evaluate(linear_interpolation(winds), winds["truth"], loss="mse")
"""
import numpy as np
import xarray as xr

from balloon.data import HOURS, TARGET_LEVELS, WINDOW_LEAD


def linear_interpolation(winds):
    """Prediction in m/s with dims (init, hour, var, level, lat, lon) on TARGET_LEVELS."""
    start = winds["t0"].sel(level=TARGET_LEVELS)
    end = fcn3_endpoint(winds["t0"], winds["t6"].mean("member"))
    fraction = xr.DataArray(np.array(HOURS, np.float32) / WINDOW_LEAD, dims="hour", coords={"hour": HOURS})
    prediction = (end - start) * fraction
    prediction += start
    return prediction.transpose("init", "hour", "var", "level", "lat", "lon")


def fcn3_endpoint(t0, fcn3_t6):
    """FCN3's t6 on TARGET_LEVELS. FCN3 only has 50/100/150 of them, so for the others its 6h change is
    interpolated in log pressure (held constant above 50 hPa) and added to ERA5 t0."""
    change = fcn3_t6 - t0.sel(level=fcn3_t6.level)
    return t0.sel(level=TARGET_LEVELS) + interpolate_levels(change, TARGET_LEVELS)


def interpolate_levels(field, levels):
    source = field.level.values
    weights = xr.DataArray(log_pressure_weights(source, levels), dims=("new_level", "level"),
                           coords={"new_level": levels, "level": source})
    return xr.dot(weights, field, dim="level").rename(new_level="level")


def log_pressure_weights(source_levels, target_levels):
    """Matrix W with W @ values_at_source = values interpolated (in log p) to the target levels."""
    unit_vectors = np.eye(len(source_levels))
    columns = [np.interp(np.log(target_levels), np.log(source_levels), unit) for unit in unit_vectors]
    return np.stack(columns, axis=1).astype(np.float32)
