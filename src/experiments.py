"""
The empirical stages.

  A: how many hidden units does each problem need?

Stage A selects on the validation set, averaging over the three algorithms as
well as over its seeds.
"""
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from evaluation import run_trial
from optimizers import DEFAULTS

HIDDEN_GRID = [1, 2, 4, 6, 8, 12, 16, 24, 32]

HIDDEN_SEEDS = range(5)


def _work(job):
    params = json.loads(job["params"])
    result = run_trial(job["problem"], job["algorithm"], job["hidden"], params,
                       job["seed"])

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


def select_hidden(frame, out):
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
    selected.to_csv(out / "selected_hidden_units.csv", index=False)
    return dict(zip(selected["problem"], selected["hidden"]))
