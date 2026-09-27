"""
How the study changes when the hidden units are ReLU instead of sigmoid.

    python src/compare_activations.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

import plots
import stats

ROOT = Path(__file__).resolve().parent.parent
ACTIVATIONS = ("sigmoid", "relu")


def paired(a, b):
    """
    Wilcoxon p-value and Hodges-Lehmann shift of b - a, with win counts.
    """

    d = b - a
    p = 1.0 if np.allclose(d, 0) else sps.wilcoxon(a, b).pvalue
    return p, stats.hodges_lehmann(d), int((d > 0).sum()), int((d < 0).sum())


def winner(p, shift, lower_wins="relu", higher_wins="sigmoid"):
    return "tie" if p >= 0.05 or shift == 0 else (lower_wins if shift < 0 else higher_wins)


def compare(folders):
    runs = {a: (pd.read_csv(folders[a] / "stage_c_comparison.csv")
                .set_index(["problem", "algorithm", "seed"]).sort_index())
            for a in ACTIVATIONS}
    curves = {a: dict(np.load(folders[a] / "learning_curves.npz")) for a in ACTIVATIONS}

    rows = []
    for (problem, algorithm), sig in runs["sigmoid"].groupby(level=[0, 1]):
        rel = runs["relu"].loc[sig.index]

        # work to a common quality: the worse of the two best validation costs of a seed
        cu = {a: [] for a in ACTIVATIONS}
        for seed in sig.index.get_level_values("seed"):
            curve = {a: curves[a][f"{problem}|{algorithm}|{seed}"] for a in ACTIVATIONS}
            target = max(c[:, 2].min() for c in curve.values())
            for a, c in curve.items():
                cu[a].append(c[c[:, 2] <= target, 0][0])

        error_p, error_shift, sigmoid_better, relu_better = paired(
            sig["test_error"].to_numpy(), rel["test_error"].to_numpy())
        speed_p, speed_shift, sigmoid_faster, relu_faster = paired(
            np.array(cu["sigmoid"]), np.array(cu["relu"]))

        rows.append(dict(
            problem=problem, algorithm=algorithm,
            hidden_sigmoid=int(sig["hidden"].iloc[0]), hidden_relu=int(rel["hidden"].iloc[0]),
            error_sigmoid=sig["test_error"].mean(), error_relu=rel["test_error"].mean(),
            error_shift=error_shift, relu_better=relu_better, sigmoid_better=sigmoid_better,
            error_p=error_p,
            train_cost_sigmoid=sig["train_cost"].mean(), train_cost_relu=rel["train_cost"].mean(),
            cu_sigmoid=np.median(cu["sigmoid"]), cu_relu=np.median(cu["relu"]),
            speed_shift=speed_shift, relu_faster=relu_faster, sigmoid_faster=sigmoid_faster,
            speed_p=speed_p,
            seconds_sigmoid=sig["seconds"].mean(), seconds_relu=rel["seconds"].mean()))

    table = pd.DataFrame(rows)
    for test in ("error", "speed"):
        table[f"{test}_p_holm"] = table.groupby("algorithm")[f"{test}_p"].transform(
            lambda p: stats.holm(p.to_numpy()))
        table[f"{test}_winner"] = [winner(p, s) for p, s in
                                   zip(table[f"{test}_p_holm"], table[f"{test}_shift"])]
    return table, runs


def verdicts(folders):
    """
    The across-problem comparison of the algorithms, under each activation.
    """

    frames = []
    for a in ACTIVATIONS:
        f = pd.read_csv(folders[a] / "friedman.csv")
        wide = f.pivot(index="criterion", columns="algorithm", values="mean_rank")
        wide["friedman_p"] = f.groupby("criterion")["friedman_p"].first()
        wide["iman_davenport_p"] = f.groupby("criterion")["iman_davenport_p"].first()
        frames.append(wide.assign(activation=a).reset_index())

    return pd.concat(frames).set_index(["criterion", "activation"]).sort_index()


def figure(runs, out):
    """
    Generalisation error of each algorithm under each activation, per problem.
    """
    
    problems = sorted(runs["sigmoid"].index.get_level_values("problem").unique())
    rows = int(np.ceil(len(problems) / 3))
    figure, axes = plt.subplots(rows, 3, figsize=(7.0, 1.9 * rows), squeeze=False)

    for axis, problem in zip(axes.flat, problems):
        for i, algorithm in enumerate(plots.ALGORITHMS):
            for j, a in enumerate(ACTIVATIONS):
                data = runs[a].loc[(problem, algorithm), "test_error"]
                box = axis.boxplot([data], positions=[3 * i + j], widths=0.8,
                                   patch_artist=True, showfliers=False)
                box["boxes"][0].set_facecolor(plots.COLOUR[algorithm])
                box["boxes"][0].set_alpha(0.35 if a == "sigmoid" else 0.9)

        # the activation under each box, the algorithm under each pair
        axis.set_xticks([3 * i + j for i in range(3) for j in range(2)],
                        [a[:3] for _ in range(3) for a in ACTIVATIONS], fontsize=7)
        axis.set_xticks([3 * i + 0.5 for i in range(3)],
                        [plots.LABEL[a] for a in plots.ALGORITHMS], minor=True, fontsize=8)
        axis.tick_params(axis="x", which="minor", length=0, pad=15)
        axis.set(title=plots.NAME[problem], ylabel="Test error")

    # hide the panels left over in the last row
    for axis in axes.flat[len(problems):]:
        axis.set_visible(False)

    plots._save(figure, out / "sigmoid_vs_relu.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigmoid", default="out")
    parser.add_argument("--relu", default="out/out_relu")
    arguments = parser.parse_args()
    folders = {"sigmoid": ROOT / arguments.sigmoid, "relu": ROOT / arguments.relu}

    table, runs = compare(folders)
    table.to_csv(folders["relu"] / "sigmoid_vs_relu.csv", index=False)

    ranks = verdicts(folders)
    ranks.to_csv(folders["relu"] / "verdicts_sigmoid_vs_relu.csv")

    figure(runs, folders["relu"])

    pd.set_option("display.width", 200)
    print("hidden units (sigmoid -> relu):",
          {r.problem: f"{r.hidden_sigmoid}->{r.hidden_relu}"
           for r in table[table["algorithm"] == "sgd"].itertuples()})

    print("\nper algorithm, over the nine problems (Holm-corrected paired Wilcoxon):")
    for algorithm, block in table.groupby("algorithm"):
        e, s = block["error_winner"].value_counts(), block["speed_winner"].value_counts()
        print(f"  {algorithm:9s} generalisation: relu better {e.get('relu', 0)}, "
              f"sigmoid better {e.get('sigmoid', 0)}, tie {e.get('tie', 0)}   |   "
              f"learning speed: relu faster {s.get('relu', 0)}, sigmoid faster "
              f"{s.get('sigmoid', 0)}, tie {s.get('tie', 0)}")

    print("\n" + table[["problem", "algorithm", "error_sigmoid", "error_relu", "error_winner",
                        "cu_sigmoid", "cu_relu", "speed_winner"]].round(4).to_string(index=False))
    print("\nmean ranks of the algorithms under each activation:\n" + ranks.round(4).to_string())

    print(f"\nwrote sigmoid_vs_relu.csv, verdicts_sigmoid_vs_relu.csv and sigmoid_vs_relu.png "
          f"to {folders['relu']}")


if __name__ == "__main__":
    main()
