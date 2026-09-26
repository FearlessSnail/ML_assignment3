"""
Benchmark problems and preprocessing.
"""
from dataclasses import dataclass

import numpy as np
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split

LO, HI = 0.1, 0.9

@dataclass
class Problem:
    name: str
    kind: str # "classification" or "regression"
    X: np.ndarray
    T: np.ndarray # network targets
    y: np.ndarray # class labels (classification only)
    budget: float # complexity units allowed per training run
    description: str


@dataclass
class Split:
    Xtr: np.ndarray
    Ttr: np.ndarray
    ytr: np.ndarray

    Xva: np.ndarray
    Tva: np.ndarray
    yva: np.ndarray

    Xte: np.ndarray
    Tte: np.ndarray
    yte: np.ndarray


def _one_hot(y, n_classes):
    T = np.full((len(y), n_classes), LO)
    T[np.arange(len(y)), y] = HI
    return T


def _classification(name, X, y, budget, description):
    y = y.astype(int)
    return Problem(name, "classification", X.astype(float), _one_hot(y, y.max() + 1),
                   y, budget, description)


def _regression(name, X, t, budget, description):
    return Problem(name, "regression", X.astype(float), t.reshape(-1, 1).astype(float),
                   None, budget, description)


def _spirals(n_per_class=200, turns=1.25, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)

    r = np.linspace(0.2, 1.0, n_per_class)
    theta = np.linspace(0, turns * 2 * np.pi, n_per_class)
    inner = np.c_[r * np.cos(theta), r * np.sin(theta)]
    outer = -inner

    X = np.vstack([inner, outer]) + rng.normal(0, noise, (2 * n_per_class, 2))
    y = np.r_[np.zeros(n_per_class), np.ones(n_per_class)]
    return X, y


def _sinc(n=300, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-10, 10, n)
    return x.reshape(-1, 1), np.sinc(x / np.pi) + rng.normal(0, noise, n)


def _sin2d(n=600, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 2))
    t = np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])
    return X, t + rng.normal(0, noise, n)


BUILDERS = {
    # classification
    "iris": lambda: _classification(
        "iris", *skd.load_iris(return_X_y=True), 600,
        "150 patterns, 4 inputs, 3 classes - small and almost linearly separable"),
    "wine": lambda: _classification(
        "wine", *skd.load_wine(return_X_y=True), 900,
        "178 patterns, 13 inputs, 3 classes - features on wildly different scales"),
    "cancer": lambda: _classification(
        "cancer", *skd.load_breast_cancer(return_X_y=True), 900,
        "569 patterns, 30 inputs, 2 classes - higher dimensional, correlated inputs"),
    "digits": lambda: _classification(
        "digits", *skd.load_digits(return_X_y=True), 2400,
        "1797 patterns, 64 inputs, 10 classes - largest and highest dimensional"),
    "spirals": lambda: _classification(
        "spirals", *_spirals(), 6000,
        "400 patterns, 2 inputs, 2 classes - two interlocking spirals of 1.25 turns, "
        "a highly non-linear boundary with many local minima"),

    # function approximation
    "sinc": lambda: _regression(
        "sinc", *_sinc(), 3000,
        "300 patterns, 1 input - sin(x)/x on [-10,10] plus Gaussian noise (sd 0.05)"),
    "sin2d": lambda: _regression(
        "sin2d", *_sin2d(), 4000,
        "600 patterns, 2 inputs - sin(2 pi x1)cos(2 pi x2) on the unit square plus "
        "Gaussian noise (sd 0.05)"),
    "friedman": lambda: _regression(
        "friedman", *skd.make_friedman1(600, noise=1.0, random_state=0), 2000,
        "600 patterns, 10 inputs of which 5 are irrelevant - Friedman #1 function plus "
        "Gaussian noise (sd 1)"),
    "diabetes": lambda: _regression(
        "diabetes", *skd.load_diabetes(return_X_y=True), 900,
        "442 patterns, 10 inputs - real world, very noisy, low attainable accuracy"),
}

CLASSIFICATION = ["iris", "wine", "cancer", "digits", "spirals"]
REGRESSION = ["sinc", "sin2d", "friedman", "diabetes"]
ALL = CLASSIFICATION + REGRESSION

_cache = {}


def get(name):
    if name not in _cache:
        _cache[name] = BUILDERS[name]()
    return _cache[name]


def split(problem, seed, val_frac=0.2, test_frac=0.2):
    """
    Stratified (classification) 60/20/20 split, standardised on the training part.
    """
    idx = np.arange(len(problem.X))
    strat = problem.y
    rest_frac = val_frac + test_frac

    tr, rest = train_test_split(idx, test_size=rest_frac, random_state=seed,
                                stratify=strat)
    va, te = train_test_split(rest, test_size=test_frac / rest_frac, random_state=seed,
                              stratify=None if strat is None else strat[rest])

    X, T = problem.X, problem.T
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd

    if problem.kind == "regression":
        tmu, tsd = T[tr].mean(0), T[tr].std(0)
        T = (T - tmu) / tsd
    y = problem.y if problem.y is not None else np.zeros(len(X), int)

    return Split(Xs[tr], T[tr], y[tr],
                 Xs[va], T[va], y[va],
                 Xs[te], T[te], y[te])


if __name__ == "__main__":
    for name in ALL:
        p = get(name)
        print(f"{p.name:10s} {p.kind:14s} X{p.X.shape} T{p.T.shape} "
              f"budget={p.budget:5.0f}  {p.description}")
