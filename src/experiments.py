"""The three empirical stages.

  A: architecture, how many hidden units does each problem need?
  B: control, what are the best control parameters for each algorithm?
  A2: confirmation, does the choice of A survive the tuning of B?
  C: comparison, which algorithm generalises best, under equal work?

Stages A and B select on the validation set; stage C uses fresh seeds, so that
the final comparison is not measured on the same resamples that were used to
make the choices."""

import itertools
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from evaluation import run_trial
from optimizers import DEFAULTS

HIDDEN_GRID = [1, 2, 4, 6, 8, 12, 16, 24, 32]

HIDDEN_SEEDS = range(5)
PARAM_SEEDS = range(10)
COMPARE_SEEDS = range(100, 130)

PARAM_GRIDS = {
    "sgd": dict(learning_rate=[0.05, 0.1, 0.5, 1.0, 2.0],
                momentum=[0.0, 0.5, 0.9],
                batch_size=[8, 32, 128]),
    "scg": dict(sigma=[1e-6, 1e-5, 1e-4, 1e-3],
                lambda_init=[1e-8, 1e-6, 1e-4]),
    "leapfrog": dict(dt=[0.1, 0.5, 1.0],
                     delta=[0.5, 2.0, 10.0, 50.0],
                     m=[3, 5, 8],
                     delta1=[1e-3, 1e-2]),
}

QUICK_GRIDS = {
    "sgd": dict(learning_rate=[0.05, 0.5], momentum=[0.0, 0.9], batch_size=[32]),
    "scg": dict(sigma=[1e-5, 1e-3], lambda_init=[1e-6]),
    "leapfrog": dict(dt=[0.25, 1.0], delta=[1.0, 5.0], m=[3], delta1=[1e-3]),
}


def grid(spec):
    keys = list(spec)
    return [dict(zip(keys, values)) for values in itertools.product(*spec.values())]


def _work(job):
    params = json.loads(job["params"])
    result = run_trial(job["problem"], job["algorithm"], job["hidden"], params,
                       job["seed"], curve=job.get("curve", False))

    result["params"] = job["params"]
    result["stage"] = job["stage"]
    return result


def run_jobs(jobs, jobs_n, label):
    # One BLAS thread per worker: there is already one worker per core
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    print(f"{label}: {len(jobs)} runs on {jobs_n} processes", flush=True)
    results = []
    with Pool(jobs_n) as pool:
        for i, result in enumerate(pool.imap_unordered(_work, jobs, chunksize=4), 1):
            results.append(result)
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)
    return results


def one_se_rule(frame, value="val_error"):
    """
    Mean, standard error and the one standard error threshold of the best row.
    """
    stats = frame.groupby(level=0)[value].agg(["mean", "std", "count"])
    stats["se"] = stats["std"] / np.sqrt(stats["count"])

    best = stats["mean"].idxmin()
    return stats, stats.loc[best, "mean"] + stats.loc[best, "se"], best


# ----------------------- stage A ------------------------
def stage_hidden(problems, jobs_n, out):
    jobs = [dict(stage="hidden", problem=p, algorithm=a, hidden=h,
                 params=json.dumps(DEFAULTS[a]), seed=s)
            for p in problems for a in DEFAULTS for h in HIDDEN_GRID for s in HIDDEN_SEEDS]

    frame = pd.DataFrame(run_jobs(jobs, jobs_n, "stage A (hidden units)"))
    frame.to_csv(out / "stage_a_hidden_units.csv", index=False)
    return frame


def select_hidden(frame, out, name="selected_hidden_units.csv"):
    """
    Optimal = fewest hidden units whose mean validation error is within one
    standard error of the best, averaged over the three algorithms.
    """

    rows = []
    for problem, block in frame.groupby("problem"):
        stats, threshold, best = one_se_rule(block.set_index("hidden"))
        chosen = min(h for h in stats.index if stats.loc[h, "mean"] <= threshold)

        per_algorithm = {a: int(g.groupby("hidden")["val_error"].mean().idxmin())
                         for a, g in block.groupby("algorithm")}

        rows.append(dict(problem=problem, hidden=int(chosen), unconstrained_best=int(best),
                         val_error=stats.loc[chosen, "mean"], threshold=threshold,
                         **{f"best_for_{a}": h for a, h in per_algorithm.items()}))

    selected = pd.DataFrame(rows)
    selected.to_csv(out / name, index=False)
    return dict(zip(selected["problem"], selected["hidden"]))


# ----------------------- stage B ------------------------
def stage_params(problems, hidden, jobs_n, out, quick=False):
    grids = QUICK_GRIDS if quick else PARAM_GRIDS
    jobs = [dict(stage="params", problem=p, algorithm=a, hidden=hidden[p],
                 params=json.dumps(c), seed=s)
            for p in problems for a in grids for c in grid(grids[a]) for s in PARAM_SEEDS]

    frame = pd.DataFrame(run_jobs(jobs, jobs_n, "stage B (control parameters)"))
    frame.to_csv(out / "stage_b_control_parameters.csv", index=False)
    return frame


def select_params(frame, out):
    """
    Best = lowest mean validation error; among settings within one standard
    error of it, the one that reaches its best solution in the least work.
    """
    rows = []
    for (problem, algorithm), block in frame.groupby(["problem", "algorithm"]):
        stats, threshold, _ = one_se_rule(block.set_index("params"))
        stats["cu_at_best"] = block.groupby("params")["cu_at_best"].mean()

        tied = stats[stats["mean"] <= threshold]
        chosen = tied["cu_at_best"].idxmin()

        rows.append(dict(problem=problem, algorithm=algorithm, params=chosen,
                         val_error=stats.loc[chosen, "mean"], n_tied=len(tied),
                         spread=stats["mean"].max() - stats["mean"].min()))

    selected = pd.DataFrame(rows)
    selected.to_csv(out / "selected_control_parameters.csv", index=False)
    return {(r.problem, r.algorithm): json.loads(r.params) for r in selected.itertuples()}


# ----------------------- stage A2 ------------------------
def stage_confirm(problems, params, jobs_n, out):
    """
    Stage A again, with every algorithm at its tuned control parameters.
    Stage A had to choose the architecture before the parameters were known;
    this checks that the choice does not depend on that order.
    """

    jobs = [dict(stage="confirm", problem=p, algorithm=a, hidden=h,
                 params=json.dumps(params[(p, a)]), seed=s)
            for p in problems for a in DEFAULTS for h in HIDDEN_GRID for s in HIDDEN_SEEDS]

    frame = pd.DataFrame(run_jobs(jobs, jobs_n, "stage A2 (hidden units, tuned)"))
    frame.to_csv(out / "stage_a2_hidden_units_tuned.csv", index=False)
    return frame


# ----------------------- stage C ------------------------
def stage_compare(problems, hidden, params, jobs_n, out):
    jobs = [dict(stage="compare", problem=p, algorithm=a, hidden=hidden[p],
                 params=json.dumps(params[(p, a)]), seed=s, curve=True)
            for p in problems for a in DEFAULTS for s in COMPARE_SEEDS]
    results = run_jobs(jobs, jobs_n, "stage C (comparison)")

    curves = {f"{r['problem']}|{r['algorithm']}|{r['seed']}": r.pop("curve")
              for r in results}
    np.savez_compressed(out / "learning_curves.npz", **curves)

    frame = pd.DataFrame(results)
    frame.to_csv(out / "stage_c_comparison.csv", index=False)
    return frame


def add_convergence_speed(frame, curves):
    """
    Work each run needs to reach a common quality
    """

    frame = frame.copy()
    curve = [curves[f"{r.problem}|{r.algorithm}|{r.seed}"] for r in frame.itertuples()]

    # exact, unlike the csv copy
    frame["best"] = [c[:, 2].min() for c in curve]
    frame["target"] = frame.groupby(["problem", "seed"])["best"].transform("max")
    frame["cu_to_target"] = [c[c[:, 2] <= t, 0][0] for c, t in zip(curve, frame["target"])]
    return frame.drop(columns="best")
