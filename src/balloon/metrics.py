"""Scoring a prediction against truth. Both are DataArrays with dims (init, hour, var, level, lat, lon), in m/s."""
import xarray as xr

AVERAGED_DIMS = ("init", "lat", "lon")


def mse(prediction, truth):
    squared_error = prediction - truth
    squared_error **= 2
    return squared_error.mean(AVERAGED_DIMS, skipna=False)


LOSSES = {"mse": mse}


def evaluate(prediction, truth, loss="mse"):
    """Score per (hour, var, level)."""
    prediction, truth = xr.align(prediction, truth, join="exact")
    return LOSSES[loss](prediction, truth).rename(loss)


def score_table(scores):
    """Scores averaged over u and v: one row per hour, one column per level, plus an all-level column."""
    by_level = scores.mean("var").to_pandas().astype(float)
    by_level["all"] = scores.mean(["var", "level"]).to_pandas()
    return by_level
