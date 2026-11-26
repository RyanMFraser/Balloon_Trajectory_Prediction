"""Local experiment records: runs/<name>/config.json and runs/<name>/scores_<split>.nc."""
import json

import xarray as xr

from balloon.data import REPO

RUNS_DIR = REPO / "runs"


def save_run(name, split, scores, config):
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    scores.to_netcdf(run_dir / f"scores_{split}.nc")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    return run_dir


def load_scores(split):
    """{run name: scores} for every run that has been scored on this split."""
    paths = sorted(RUNS_DIR.glob(f"*/scores_{split}.nc"))
    return {path.parent.name: _read_scores(path) for path in paths}


def _read_scores(path):
    with xr.open_dataarray(path) as scores:
        return scores.load()
