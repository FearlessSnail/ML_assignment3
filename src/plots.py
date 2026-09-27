"""
Plotting functions for report.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import datasets
from evaluation import run_trial
from experiments import COMPARE_SEEDS, HIDDEN_SEEDS
from nn import ACTIVATIONS, Net, add_bias

# names and styles
NAME = {"iris": "Iris", "wine": "Wine", "cancer": "Cancer", "digits": "Digits",
        "spirals": "Spirals", "sinc": "Sinc", "sin2d": "Sin2D", "friedman": "Friedman1",
        "diabetes": "Diabetes"}
ALGORITHMS = ["sgd", "scg", "leapfrog"]
LABEL = {"sgd": "SGD", "scg": "SCG", "leapfrog": "LeapFrog"}

COLOUR = {"sgd": "#0072B2", "scg": "#D55E00", "leapfrog": "#009E73"}
STYLE = {"sgd": dict(ls="-", marker="o"), "scg": dict(ls="--", marker="s"),
         "leapfrog": dict(ls=":", marker="^")}
PROBLEM_COLOUR = {"sin2d": "#56B4E9", "friedman": "#CC79A7", "diabetes": "#E69F00",
                  "digits": "#000000"}

# figure sizes in inches, as printed
THIRD = (2.3, 1.62) # three sub-figures across the page
QUARTER = (1.72, 1.5) # four across the page, or two in a column

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{mathptmx}",
    "font.family": "serif",
    "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3,
    "legend.frameon": False,
})


def _save(figure, path):
    figure.tight_layout(pad=0.2)
    figure.savefig(path)
    plt.close(figure)


def _hidden_units_axis(axis, ticks):
    axis.set_xscale("log")
    axis.set_xlabel("Hidden units")
    axis.set_xticks(ticks)
    axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axis.minorticks_off()


def _legend(handles, path):
    figure = plt.figure(figsize=(7.0, 0.25))
    figure.legend(handles=handles, loc="center", ncol=len(handles), handlelength=2.2,
                  columnspacing=1.6)
    figure.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def _algorithm_lines():
    return [Line2D([], [], color=COLOUR[a], label=LABEL[a], **STYLE[a]) for a in ALGORITHMS]


# ----------------- figure 1 -------------------
def hidden_units(stage_a, selected, out):
    """
    Mean validation error against the number of hidden units, per problem, with
    the one-standard-error threshold and the chosen size.
    """

    for problem, block in stage_a.groupby("problem"):
        figure, axis = plt.subplots(figsize=THIRD)

        for algorithm in ALGORITHMS:
            curve = block[block["algorithm"] == algorithm].groupby("hidden")["val_error"].mean()
            axis.plot(curve.index, curve, color=COLOUR[algorithm], alpha=0.8, **STYLE[algorithm])

        pooled = block.groupby("hidden")["val_error"].agg(["mean", "std", "count"])
        se = pooled["std"] / np.sqrt(pooled["count"])
        axis.errorbar(pooled.index, pooled["mean"], yerr=se, color="black", lw=1.4, capsize=1.5)

        choice = selected.loc[problem]
        axis.axhline(choice["threshold"], color="grey", lw=0.7, ls="-.")
        axis.axvline(choice["hidden"], color="black", lw=0.7, ls="--")

        _hidden_units_axis(axis, [1, 2, 4, 8, 16, 32])
        axis.set(yscale="log", ylabel="Validation error")
        axis.minorticks_off()
        _save(figure, out / f"hidden_{problem}.pdf")

    extra = [Line2D([], [], color="black", lw=1.4, label="Pooled mean"),
             Line2D([], [], color="grey", lw=0.7, ls="-.", label="Threshold"),
             Line2D([], [], color="black", lw=0.7, ls="--", label="Chosen size")]
    _legend(_algorithm_lines() + extra, out / "legend_hidden.pdf")


# ----------------- figure 2 -------------------
def output_curvature(problem, hidden, activation):
    """
    Largest eigenvalue of the Hessian with respect to the linear output weights,
    at the initial weights, averaged over the seeds of stage A2.
    """

    hidden_function, _ = ACTIVATIONS[activation]

    values = []
    for seed in HIDDEN_SEEDS:
        split = datasets.split(datasets.get(problem), seed)
        net = Net(split.Xtr.shape[1], hidden, 1, linear_output=True, activation=activation)
        U, _ = net.unpack(net.initial_weights(np.random.default_rng(seed)))

        Z = add_bias(hidden_function(add_bias(split.Xtr) @ U))
        values.append(np.linalg.eigvalsh(Z.T @ Z / len(Z)).max())

    return np.mean(values)


def sgd_stability(stage_a2, params, out, activation="sigmoid"):
    """
    Why tuned SGD fails on larger networks.
    """

    problems = [p for p in ("sin2d", "friedman", "diabetes") if p in set(stage_a2["problem"])]
    if not problems:
        return
    sizes = sorted(stage_a2["hidden"].unique())

    # (a) the stability ratio at the initial weights
    figure, axis = plt.subplots(figsize=QUARTER)
    for problem in problems:
        sgd = params[(problem, "sgd")]
        bound = 2 * (1 + sgd["momentum"])
        ratio = [sgd["learning_rate"] * output_curvature(problem, h, activation) / bound
                 for h in sizes]
        axis.plot(sizes, ratio, color=PROBLEM_COLOUR[problem], marker="o", label=NAME[problem])

    axis.axhline(1, color="black", lw=0.7, ls="--")
    _hidden_units_axis(axis, [1, 4, 16])
    axis.set_ylabel("Stability ratio")
    axis.legend(loc="upper left", handlelength=1.5)
    _save(figure, out / "stability_ratio.pdf")

    # (b) the validation error of tuned SGD, with SCG dotted
    figure, axis = plt.subplots(figsize=QUARTER)
    for problem in problems:
        block = stage_a2[stage_a2["problem"] == problem]
        for algorithm, style in (("sgd", dict(marker="o")), ("scg", dict(ls=":", lw=0.8))):
            curve = block[block["algorithm"] == algorithm].groupby("hidden")["val_error"].mean()
            axis.plot(curve.index, curve, color=PROBLEM_COLOUR[problem], **style)

    _hidden_units_axis(axis, [1, 4, 16])
    axis.set(yscale="log", ylabel="Validation error")
    axis.minorticks_off()
    _save(figure, out / "stability_error.pdf")


# ----------------- figure 3 -------------------
def leapfrog_stall(hidden, params, out, activation="sigmoid"):
    """
    One LeapFrog run on Digits, where it stalled, and on Sin2D, where it did
    well: the lowest training cost so far, and the time step.
    """

    problems = [p for p in ("digits", "sin2d") if p in hidden]
    if not problems:
        return

    figure_cost, cost_axis = plt.subplots(figsize=QUARTER)
    figure_step, step_axis = plt.subplots(figsize=QUARTER)
    for problem in problems:
        steps = []
        run = run_trial(problem, "leapfrog", hidden[problem],
                        {**params[(problem, "leapfrog")], "trace": steps},
                        COMPARE_SEEDS[0], curve=True, activation=activation)
        curve, steps = run["curve"], np.array(steps)

        cost_axis.plot(curve[:, 0] / curve[-1, 0], np.minimum.accumulate(curve[:, 1]),
                       color=PROBLEM_COLOUR[problem], label=NAME[problem])
        step_axis.plot(steps[:, 0] / steps[-1, 0], steps[:, 1], color=PROBLEM_COLOUR[problem])

        print(f"  LeapFrog on {problem}: step limit active on {100 * steps[:, 2].mean():.0f}"
              f" percent of the steps, largest time step {steps[:, 1].max():.1f}")

    cost_axis.set(yscale="log", xlabel="Fraction of budget", ylabel="Training cost")
    cost_axis.legend(loc="upper right", handlelength=1.5)
    step_axis.set(yscale="log", xlabel="Fraction of budget", ylabel="Time step")
    cost_axis.minorticks_off()
    step_axis.minorticks_off()

    _save(figure_cost, out / "leapfrog_cost.pdf")
    _save(figure_step, out / "leapfrog_step.pdf")


# ----------------- figure 4 -------------------
def learning_curves(comparison, curves, out, problems=("digits", "spirals", "sinc", "sin2d")):
    """
    Median and interquartile range of the lowest validation cost so far against
    the work done, over the runs of stage C.
    """

    for problem in problems:
        block = comparison[comparison["problem"] == problem]
        if block.empty:
            continue
        common = np.linspace(3, block["cu_used"].max(), 300)

        figure, axis = plt.subplots(figsize=QUARTER)
        for algorithm in ALGORITHMS:
            traces = []
            for seed in block[block["algorithm"] == algorithm]["seed"]:
                c = curves[f"{problem}|{algorithm}|{seed}"]
                traces.append(np.interp(common, c[:, 0], np.minimum.accumulate(c[:, 2])))

            lower, median, upper = np.percentile(traces, [25, 50, 75], axis=0)
            axis.plot(common, median, color=COLOUR[algorithm], ls=STYLE[algorithm]["ls"])
            axis.fill_between(common, lower, upper, color=COLOUR[algorithm], alpha=0.15, lw=0)

        axis.set(xscale="log", yscale="log", xlabel="Complexity units", ylabel="Validation cost")
        axis.minorticks_off()
        _save(figure, out / f"curves_{problem}.pdf")

    _legend(_algorithm_lines(), out / "legend_curves.pdf")