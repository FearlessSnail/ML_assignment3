"""
Single hidden layer feedforward network.

The weights live in one flat vector so that the network can be handed to any
unconstrained optimiser as a plain function w -> (E, dE/dw).
"""
import numpy as np


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a, -500.0, 500.0)))

ACTIVATIONS = {
    "sigmoid": (sigmoid, lambda g, z: g * z * (1.0 - z))
}

def add_bias(X):
    return np.hstack([X, np.ones((len(X), 1))])


class Net:
    def __init__(self, n_in, n_hidden, n_out, linear_output=False):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.linear_output = linear_output

        self.n_weights = (n_in + 1) * n_hidden + (n_hidden + 1) * n_out
        self._cut = (n_in + 1) * n_hidden

    def initial_weights(self, rng):
        w1 = rng.uniform(-1, 1, self._cut) / np.sqrt(self.n_in + 1)
        w2 = rng.uniform(-1, 1, self.n_weights - self._cut) / np.sqrt(self.n_hidden + 1)
        return np.concatenate([w1, w2])

    def unpack(self, w):
        W1 = w[:self._cut].reshape(self.n_in + 1, self.n_hidden)
        W2 = w[self._cut:].reshape(self.n_hidden + 1, self.n_out)
        return W1, W2

    def predict(self, w, X):
        W1, W2 = self.unpack(w)
        Z = sigmoid(add_bias(X) @ W1)
        A = add_bias(Z) @ W2
        return A if self.linear_output else sigmoid(A)

    def error(self, w, X, T):
        Y = self.predict(w, X)
        return 0.5 * np.sum((Y - T) ** 2) / len(X)

    def error_grad(self, w, X, T):
        # forward pass
        W1, W2 = self.unpack(w)
        Xb = add_bias(X)
        Z = sigmoid(Xb @ W1)
        Zb = add_bias(Z)
        A = Zb @ W2
        Y = A if self.linear_output else sigmoid(A)
        E = 0.5 * np.sum((Y - T) ** 2) / len(X)

        # backward pass: output deltas, then hidden deltas
        D2 = (Y - T) / len(X)
        if not self.linear_output:
            D2 *= Y * (1.0 - Y)
            
        D1 = (D2 @ W2[:-1].T) * Z * (1.0 - Z)

        return E, np.concatenate([(Xb.T @ D1).ravel(), (Zb.T @ D2).ravel()])


def check_gradient(seed=0):
    """
    Finite difference check of error_grad.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0

    for linear in (False, True):
        net = Net(3, 4, 2, linear_output=linear)
        w = net.initial_weights(rng)
        X, T = rng.normal(size=(10, 3)), rng.uniform(size=(10, 2))
        _, g = net.error_grad(w, X, T)

        for i in rng.choice(net.n_weights, 15, replace=False):
            step = np.zeros_like(w)
            step[i] = 1e-6
            fd = (net.error(w + step, X, T) - net.error(w - step, X, T)) / 2e-6
            worst = max(worst, abs(fd - g[i]) / max(1e-12, abs(fd)))

    return worst


if __name__ == "__main__":
    print("worst relative gradient error: %.2e" % check_gradient())
