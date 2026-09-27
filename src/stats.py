"""
Statistical comparison of the three algorithms.
"""

import numpy as np
import pandas as pd
from scipy import stats as sps

# studentised range / sqrt(2) at alpha = 0.05, indexed by number of algorithms
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}


def summary(frame, criteria=("test_error", "test_mse", "test_accuracy", "train_cost",
                             "val_cost", "cu_to_target", "cu_at_best", "seconds",
                             "iterations")):
    
    rows = []
    for (problem, algorithm), block in frame.groupby(["problem", "algorithm"]):
        row = dict(problem=problem, kind=block["kind"].iloc[0], algorithm=algorithm,
                   hidden=block["hidden"].iloc[0], runs=len(block))
        for criterion in criteria:
            row[f"{criterion}_mean"] = block[criterion].mean()
            row[f"{criterion}_std"] = block[criterion].std()
            row[f"{criterion}_median"] = block[criterion].median()
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["problem", "algorithm"])


def shapiro(d):
    """
    Shapiro-Wilk p-value for normality of the paired differences.
    """
    return sps.shapiro(d).pvalue if len(np.unique(d)) > 2 else 0.0


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))

    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def hodges_lehmann(d):
    """
    The location shift that goes with the Wilcoxon signed rank test: the
    median of the pairwise averages of the differences.
    """
    walsh = (d[:, None] + d[None, :]) / 2
    return float(np.median(walsh[np.triu_indices(len(d))]))


def pairwise(frame, criterion="test_error"):
    """
    Paired Wilcoxon signed rank tests within each problem.
    """

    rows = []
    for problem, block in frame.groupby("problem"):
        wide = block.pivot(index="seed", columns="algorithm", values=criterion).dropna()
        algorithms = sorted(wide.columns)

        tests = []
        for i, a in enumerate(algorithms):
            for b in algorithms[i + 1:]:
                d = wide[a].to_numpy() - wide[b].to_numpy()
                if np.allclose(d, 0):
                    p, statistic = 1.0, np.nan          # identical on every run
                else:
                    statistic, p = sps.wilcoxon(wide[a], wide[b])

                tests.append(dict(problem=problem, criterion=criterion, a=a, b=b,
                                  runs=len(d), a_better=int((d < 0).sum()),
                                  b_better=int((d > 0).sum()),
                                  median_a=np.median(wide[a]), median_b=np.median(wide[b]),
                                  shift=hodges_lehmann(d), shapiro_p=shapiro(d),
                                  statistic=statistic, p=p))

        for test, adjusted in zip(tests, holm([t["p"] for t in tests])):
            test["p_holm"] = adjusted
            test["winner"] = ("tie" if adjusted >= 0.05 or test["shift"] == 0 else
                              (test["a"] if test["shift"] < 0 else test["b"]))

        rows += tests

    return pd.DataFrame(rows)


def friedman(frame, criterion="test_error"):
    """
    Friedman test over problems, with the Iman-Davenport correction (the
    Friedman statistic itself is conservative for so few problems) and the
    Nemenyi critical difference.
    """
    scores = (frame.groupby(["problem", "algorithm"])[criterion].mean()
              .unstack("algorithm").dropna())
    ranks = scores.rank(axis=1)
    n, k = scores.shape

    statistic, p = sps.friedmanchisquare(*[scores[c] for c in scores.columns])

    f_statistic = (n - 1) * statistic / (n * (k - 1) - statistic)
    f_p = sps.f.sf(f_statistic, k - 1, (k - 1) * (n - 1))

    cd = NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n))
    mean_ranks = ranks.mean().sort_values()

    return dict(criterion=criterion, problems=n, algorithms=k, statistic=statistic,
                p=p, f_statistic=f_statistic, f_p=f_p, critical_difference=cd,
                mean_ranks=mean_ranks, scores=scores, ranks=ranks)


def report(frame, out, criteria=("test_error", "test_mse", "cu_to_target",
                                 "seconds")):
    summary(frame).to_csv(out / "summary.csv", index=False)

    tests = pd.concat([pairwise(frame, c) for c in criteria])
    tests.to_csv(out / "pairwise_tests.csv", index=False)

    errors = tests[tests["criterion"] == "test_error"]
    lines = [f"normality of paired test error differences (Shapiro-Wilk): rejected at "
             f"0.05 in {(errors['shapiro_p'] < 0.05).sum()} of {len(errors)} comparisons, "
             f"so rank based tests are used throughout"]

    rows = []
    for criterion in criteria:
        result = friedman(frame, criterion)
        result["ranks"].to_csv(out / f"friedman_ranks_{criterion}.csv")

        lines.append(f"{criterion}: over {result['problems']} problems, "
                     f"Friedman chi2={result['statistic']:.3f} p={result['p']:.4g}, "
                     f"Iman-Davenport F={result['f_statistic']:.3f} "
                     f"p={result['f_p']:.4g}, Nemenyi CD={result['critical_difference']:.3f}")
        lines.append("  mean ranks: " + ", ".join(
            f"{a}={r:.2f}" for a, r in result["mean_ranks"].items()))

        best, rank = result["mean_ranks"].index[0], result["mean_ranks"].iloc[0]
        for algorithm, other in result["mean_ranks"].items():
            rows.append(dict(criterion=criterion, algorithm=algorithm, mean_rank=other,
                             friedman_p=result["p"], iman_davenport_p=result["f_p"],
                             differs_from_best=bool(other - rank > result["critical_difference"])))
        lines.append(f"  best: {best}; algorithms not significantly worse: " + ", ".join(
            a for a, r in result["mean_ranks"].items()
            if r - rank <= result["critical_difference"]))

    pd.DataFrame(rows).to_csv(out / "friedman.csv", index=False)

    text = "\n".join(lines)
    (out / "friedman.txt").write_text(text + "\n")
    return text
