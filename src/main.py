"""
Run the comparison.

    python src/main.py all (full study)
    python src/main.py analyse (statistics and figures from the csv files)
    python src/main.py all --activation relu --out out/out_relu (the study with ReLU units)

Results are written to out/ (or --out) as csv files and pdf figures.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import datasets
import experiments
import plots
import stats as statistics
from nn import ACTIVATIONS

ROOT = Path(__file__).resolve().parent.parent


def load(out, name):
    path = out / name
    if not path.exists():
        raise SystemExit(f"{path} is missing - run the earlier stage first")
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["hidden", "params", "confirm", "compare",
                                          "analyse", "all"])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--problems", nargs="+", default=None)
    parser.add_argument("--activation", choices=list(ACTIVATIONS), default="sigmoid",
                        help="hidden unit activation")
    parser.add_argument("--out", default="out", help="output folder, relative to the project")
    parser.add_argument("--hidden-grid", type=int, nargs="+", default=experiments.HIDDEN_GRID,
                        help="hidden layer sizes tried in stages A and A2")

    arguments = parser.parse_args()

    out = ROOT / arguments.out
    out.mkdir(parents=True, exist_ok=True)
    problems = arguments.problems or (["iris", "spirals", "digits", "sinc", "diabetes"]
                                      if arguments.quick else datasets.ALL)
    stage, jobs, activation = arguments.stage, arguments.jobs, arguments.activation

    # stage A: number of hidden units
    if stage in ("hidden", "all"):
        experiments.stage_hidden(problems, jobs, out, activation, arguments.hidden_grid)

    hidden = experiments.select_hidden(load(out, "stage_a_hidden_units.csv"), out)
    print("hidden units:", hidden, flush=True)
    if stage == "hidden":
        return

    # stage B: control parameters
    if stage in ("params", "all"):
        experiments.stage_params(problems, hidden, jobs, out, activation, arguments.quick)

    params = experiments.select_params(load(out, "stage_b_control_parameters.csv"), out)
    for (problem, algorithm), setting in sorted(params.items()):
        print(f"  {problem:10s} {algorithm:9s} {setting}", flush=True)
    if stage == "params":
        return

    # stage A2: hidden units with the tuned parameters
    if stage in ("confirm", "all"):
        experiments.stage_confirm(problems, params, jobs, out, activation, arguments.hidden_grid)

    confirmed = experiments.select_hidden(load(out, "stage_a2_hidden_units_tuned.csv"),
                                          out, "confirmed_hidden_units.csv")
    print("hidden units with tuned parameters:", confirmed, flush=True)
    if stage == "confirm":
        return

    # stage C: the comparison
    if stage in ("compare", "all"):
        experiments.stage_compare(problems, hidden, params, jobs, out, activation)

    curves = dict(np.load(out / "learning_curves.npz"))
    comparison = experiments.add_convergence_speed(load(out, "stage_c_comparison.csv"),
                                                   curves)
    print(comparison.groupby(["problem", "algorithm"])[["test_error", "cu_to_target"]]
          .median().to_string())

    if stage == "compare":
        return

    pd.DataFrame([dict(problem=p.name, kind=p.kind, patterns=len(p.X),
                       inputs=p.X.shape[1], outputs=p.T.shape[1],
                       hidden=hidden[p.name], budget=p.budget, description=p.description)
                  for p in map(datasets.get, problems)]).to_csv(
        out / "problems.csv", index=False)

    print("\n" + statistics.report(comparison, out))
    plots.hidden_units(load(out, "stage_a_hidden_units.csv"),
                       load(out, "selected_hidden_units.csv").set_index("problem"), out)
    plots.sgd_stability(load(out, "stage_a2_hidden_units_tuned.csv"), params, out, activation)
    plots.leapfrog_stall(hidden, params, out, activation)
    plots.learning_curves(comparison, curves, out)
    print(f"\nwrote csv files and figures to {out}")


if __name__ == "__main__":
    main()
