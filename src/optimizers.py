"""
The training algorithms.
"""
import numpy as np


class Objective:
    """
    Training cost and gradient, with complexity unit accounting.
    """

    def __init__(self, net, X, T):
        self.net, self.X, self.T = net, X, T
        self.P = len(X)
        self.cu = 0.0

    def error(self, w):
        self.cu += 1.0
        return self.net.error(w, self.X, self.T)

    def error_grad(self, w, idx=None):
        X, T = (self.X, self.T) if idx is None else (self.X[idx], self.T[idx])
        self.cu += 3.0 * len(X) / self.P
        return self.net.error_grad(w, X, T)


def sgd(obj, w, monitor, rng, learning_rate=0.1, momentum=0.9, batch_size=32):
    """
    Stochastic gradient descent with momentum; monitored once per epoch.
    """

    v = np.zeros_like(w)
    batch_size = min(batch_size, obj.P)
    epoch = 0

    while True:
        order = rng.permutation(obj.P)
        for start in range(0, obj.P, batch_size):
            _, g = obj.error_grad(w, order[start:start + batch_size])
            v = momentum * v - learning_rate * g
            w += v

        epoch += 1
        if monitor(w):
            return epoch


ALGORITHMS = {"sgd": sgd}

DEFAULTS = {
    "sgd": dict(learning_rate=0.1, momentum=0.9, batch_size=32),
}
