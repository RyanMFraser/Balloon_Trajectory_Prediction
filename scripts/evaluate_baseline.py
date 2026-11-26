"""Score the linear interpolation baseline on one split and save it under runs/baseline/.

    python scripts/evaluate_baseline.py --split test --loss mse
"""
import argparse

from balloon.baselines import linear_interpolation
from balloon.data import load_winds
from balloon.metrics import LOSSES, evaluate, score_table
from balloon.runs import save_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--loss", default="mse", choices=list(LOSSES))
    args = parser.parse_args()

    winds = load_winds(args.split)
    scores = evaluate(linear_interpolation(winds), winds["truth"], loss=args.loss)
    save_run("baseline", args.split, scores, {"model": "linear_interpolation", "loss": args.loss})
    print(f"baseline on {args.split}: {args.loss}, rows = hour, columns = level (hPa)")
    print(score_table(scores).round(2).to_string())


if __name__ == "__main__":
    main()
