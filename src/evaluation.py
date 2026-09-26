"""
Running one training trial and measuring it.
"""
import time

import numpy as np

import datasets
from nn import Net
from optimizers import ALGORITHMS, DEFAULTS, Objective


class Monitor:
    """
    Stops on the cu budget and remembers the best validation weights.
    """

    def __init__(self, obj, net, split, budget, check_every=3.0):
        self.obj, self.net, self.split = obj, net, split
        self.budget, self.check_every = budget, check_every
        self.next_check = 0.0

        self.best_error = np.inf
        self.best_w = None
        self.best_cu = 0.0

        self.overhead = 0.0
        self.curve = []

    def __call__(self, w):
        if self.obj.cu >= self.next_check:
            started = time.perf_counter()
            self.next_check = self.obj.cu + self.check_every

            train = self.net.error(w, self.split.Xtr, self.split.Ttr)
            val = self.net.error(w, self.split.Xva, self.split.Tva)
            if not np.isfinite(val):
                val = np.inf
            self.curve.append((self.obj.cu, train, val))

            if val < self.best_error:
                self.best_error, self.best_cu = val, self.obj.cu
                self.best_w = w.copy()

            self.overhead += time.perf_counter() - started

        return self.obj.cu >= self.budget


def measure(net, w, X, T, y, kind):
    """
    Primary generalisation criterion plus the secondary one.
    """

    Y = net.predict(w, X)
    mse = float(np.mean((Y - T) ** 2))

    if kind == "classification":
        accuracy = float(np.mean(np.argmax(Y, axis=1) == y))
        return dict(error=1.0 - accuracy, accuracy=accuracy, mse=mse)
    return dict(error=mse, accuracy=np.nan, mse=mse)


def run_trial(problem_name, algorithm, hidden, params, seed, budget=None):
    """
    One independent training run: split, initialise, train, measure.
    """
    # split
    problem = datasets.get(problem_name)
    split = datasets.split(problem, seed)

    # initialise
    net = Net(split.Xtr.shape[1], hidden, split.Ttr.shape[1],
              linear_output=(problem.kind == "regression"))
    w0 = net.initial_weights(np.random.default_rng(seed))

    # train
    obj = Objective(net, split.Xtr, split.Ttr)
    monitor = Monitor(obj, net, split, budget or problem.budget)

    start = time.perf_counter()
    with np.errstate(over="ignore", invalid="ignore"):
        # A control parameter setting that diverges overflows the network; the
        # monitor rejects the non finite costs, so the run simply scores badly.
        iterations = ALGORITHMS[algorithm](obj, w0.copy(), monitor,
                                           np.random.default_rng(seed + 10_000),
                                           **params)
    seconds = time.perf_counter() - start - monitor.overhead

    # measure
    w = monitor.best_w
    result = dict(problem=problem_name, kind=problem.kind, algorithm=algorithm,
                  hidden=hidden, seed=seed,
                  iterations=iterations, seconds=seconds,
                  cu_used=obj.cu, cu_at_best=monitor.best_cu,
                  train_cost=net.error(w, split.Xtr, split.Ttr),
                  val_cost=monitor.best_error)

    for name, part in (("test", (split.Xte, split.Tte, split.yte)),
                       ("val", (split.Xva, split.Tva, split.yva))):
        for key, value in measure(net, w, *part, problem.kind).items():
            result[f"{name}_{key}"] = value

    result.update({f"p_{k}": v for k, v in params.items()})

    return result


if __name__ == "__main__":
    for problem in ("iris", "sinc"):
        for algorithm, params in DEFAULTS.items():
            r = run_trial(problem, algorithm, 4, params, seed=0)
            print(f"{problem:6s} {algorithm:9s} test error {r['test_error']:.4f}  "
                  f"best after {r['cu_at_best']:5.0f} of {r['cu_used']:5.0f} cu  "
                  f"{r['seconds']:.2f}s")
