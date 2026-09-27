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


def scg(obj, w, monitor, rng, sigma=1e-4, lambda_init=1e-6):
    """
    Scaled conjugate gradient (Moller 1993, section 5).
    """

    # 1. initialisation
    N = len(w)
    lam, lam_bar = lambda_init, 0.0
    E, grad = obj.error_grad(w)
    r = -grad
    p = r.copy()
    success = True
    k = 1

    while True:
        p_norm2 = p @ p
        if p_norm2 < 1e-30:
            return k

        # 2. second order information
        if success:
            sigma_k = sigma / np.sqrt(p_norm2)
            _, grad_plus = obj.error_grad(w + sigma_k * p)
            s = (grad_plus + r) / sigma_k
            delta = p @ s

        # 3-4. scale delta, and make the Hessian positive definite if needed
        delta += (lam - lam_bar) * p_norm2
        if delta <= 0:
            lam_bar = 2 * (lam - delta / p_norm2)
            delta = -delta + lam * p_norm2
            lam = lam_bar

        # 5-6. step size and comparison parameter
        mu = p @ r
        alpha = mu / delta
        E_new = obj.error(w + alpha * p)
        comparison = 2 * delta * (E - E_new) / (mu * mu) if mu != 0 else -1.0

        if comparison >= 0:
            w += alpha * p
            E = E_new
            r_old = r
            _, grad = obj.error_grad(w)
            r = -grad
            lam_bar, success = 0.0, True

            if k % N == 0:
                p = r.copy()
            else:
                beta = (r @ r - r @ r_old) / mu
                p = r + beta * p

            if comparison >= 0.75:
                lam *= 0.25
        else:
            lam_bar, success = lam, False

        # 8. increase the scale parameter
        if comparison < 0.25:
            lam += delta * (1 - comparison) / p_norm2

        if monitor(w) or lam > 1e20 or not np.isfinite(lam):
            return k
        k += 1


def leapfrog(obj, w, monitor, rng, dt=0.5, delta=1.0, m=3, delta1=1e-3, epsilon=1e-8,
             trace=None):
    """
    LeapFrog dynamic minimisation, LFOP1(b) (Snyman 1982, 1983).
    """
    x = w
    _, g = obj.error_grad(x)
    a = -g
    v = 0.5 * a * dt
    x_prev, v_prev = x.copy(), v.copy()

    # i: restarts done since the particle last gained speed, j: how many are
    # allowed before the velocity is zeroed, s: consecutive uphill steps,
    # p: the factor by which the time step grows after a successful step.
    i, j, s, p = 0, 2, 0, 1.0
    restart_x = None
    k = 0

    while True:
        limited = False
        if restart_x is None:
            v_norm = np.linalg.norm(v)
            if v_norm * dt < delta:
                p += delta1
                dt *= p
            else:
                v *= delta / (dt * max(v_norm, 1e-30))
                limited = True
            
            # 5b. too many uphill steps
            if s >= m:
                dt *= 0.5
                x = 0.5 * (x + x_prev)
                v = 0.25 * (v + v_prev)
                s = 0

            x_next = x + v * dt
        else:
            # restart from the midpoint
            x_next, restart_x = restart_x, None

        _, g = obj.error_grad(x_next)
        if trace is not None:
            trace.append((obj.cu, dt, limited))
        a_next = -g
        v_next = v + a_next * dt

        # 7a. dt may grow again
        if a_next @ a > 0:
            s = 0
        else:
            s += 1
            p = 1.0

        k += 1
        if monitor(x_next) or np.linalg.norm(a_next) <= epsilon:
            return k

        # 8. still gaining speed
        if np.linalg.norm(v_next) > np.linalg.norm(v):
            i = 0
        else:
            # 8-9. interfere: damp and restart
            restart_x = 0.5 * (x_next + x)
            i += 1
            if i <= j:
                v_next = 0.25 * (v_next + v)
            else:
                v_next = np.zeros_like(v_next)
                j = 1

        x_prev, v_prev = x, v
        x, v, a = x_next, v_next, a_next


ALGORITHMS = {"sgd": sgd, "scg": scg, "leapfrog": leapfrog}

DEFAULTS = {
    "sgd": dict(learning_rate=0.1, momentum=0.9, batch_size=32),
    "scg": dict(sigma=1e-4, lambda_init=1e-6),
    "leapfrog": dict(dt=0.5, delta=1.0, m=3, delta1=1e-3),
}
