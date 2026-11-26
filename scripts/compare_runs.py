"""One table comparing every saved run on a split: score over all target levels, per hour.

    python scripts/compare_runs.py --split test
"""
import argparse

import pandas as pd

from balloon.metrics import score_table
from balloon.runs import load_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    runs = load_scores(args.split)
    table = pd.DataFrame({name: score_table(scores)["all"] for name, scores in runs.items()}).T
    table["mean"] = table.mean(axis=1)
    print(f"Scores on {args.split}, all target levels; columns = hour")
    print(table.sort_values("mean").round(3).to_string())


if __name__ == "__main__":
    main()
