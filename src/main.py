"""
Run the comparison.

    python src/main.py all (full study)

Results are written to out/ as csv files.
"""

import argparse
from pathlib import Path

import pandas as pd

import datasets
import experiments

ROOT = Path(__file__).resolve().parent.parent


def load(out, name):
    path = out / name
    if not path.exists():
        raise SystemExit(f"{path} is missing - run the earlier stage first")
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["hidden", "all"])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--problems", nargs="+", default=None)
    arguments = parser.parse_args()

    out = ROOT / "out"
    out.mkdir(parents=True, exist_ok=True)
    problems = arguments.problems or datasets.ALL
    stage, jobs = arguments.stage, arguments.jobs

    # stage A: number of hidden units
    if stage in ("hidden", "all"):
        experiments.stage_hidden(problems, jobs, out)

    hidden = experiments.select_hidden(load(out, "stage_a_hidden_units.csv"), out)
    print("hidden units:", hidden, flush=True)


if __name__ == "__main__":
    main()
