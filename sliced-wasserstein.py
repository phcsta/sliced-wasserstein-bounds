"""
Projected Wasserstein distance for a Gaussian AR(1) process.

We consider
    X_{t+1} = Q X_t + eta_{t+1},
    eta_t ~ N(0, C),

with deterministic initial condition X_0 = x0.

For a fixed direction v, the code computes the W_2 distance between
the law of <v, X_t> and its stationary Gaussian distribution, together
with the corresponding lower and upper bounds.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import solve_discrete_lyapunov
from scipy.special import gammaln
from matplotlib.ticker import MaxNLocator


def gaussian_moment_constant(r=2.0):
    """Return (E|Z|^r)^(1/r) for Z ~ N(0,1)."""
    if r < 1:
        raise ValueError("r must be at least one")

    log_moment = (
        0.5 * r * np.log(2.0)
        + gammaln((r + 1.0) / 2.0)
        - 0.5 * np.log(np.pi)
    )

    return np.exp(log_moment / r)


def projected_wasserstein(Q, C, v, x0, T=100):
    """
    Compute the projected W_2 distance and its bounds for t=0,...,T.

    Parameters
    ----------
    Q : (d,d) array
        Transition matrix.
    C : (d,d) array
        Covariance matrix of the Gaussian innovations.
    v : (d,) array
        Projection direction.
    x0 : (d,) array
        Deterministic initial condition.
    T : int
        Final time.

    Returns
    -------
    dict
        Arrays containing time, exact projected W_2 distance,
        lower bound, and upper bound.
    """
    Q = np.asarray(Q, dtype=float)
    C = np.asarray(C, dtype=float)
    v = np.asarray(v, dtype=float)
    x0 = np.asarray(x0, dtype=float)

    d = Q.shape[0]

    if Q.shape != (d, d):
        raise ValueError("Q must be square")
    if C.shape != (d, d):
        raise ValueError("C has incompatible dimension")
    if v.shape != (d,) or x0.shape != (d,):
        raise ValueError("v and x0 have incompatible dimensions")
    if np.max(np.abs(np.linalg.eigvals(Q))) >= 1:
        raise ValueError("Q must be Schur stable")

    C = 0.5 * (C + C.T)

    Sigma = solve_discrete_lyapunov(Q, C)
    Sigma = 0.5 * (Sigma + Sigma.T)

    sigma2_inf = v @ Sigma @ v
    if sigma2_inf <= 0:
        raise ValueError("The stationary projected variance must be positive")

    sigma_inf = np.sqrt(sigma2_inf)
    C2 = gaussian_moment_constant(2.0)

    time = np.arange(T + 1)
    w2 = np.zeros(T + 1)
    lower = np.zeros(T + 1)
    upper = np.zeros(T + 1)

    Qt = np.eye(d)
    Sigma_t = np.zeros((d, d))

    for t in time:
        mean_t = v @ Qt @ x0
        sigma2_t = v @ Sigma_t @ v
        sigma_t = np.sqrt(max(sigma2_t, 0.0))

        vt = Qt.T @ v
        covariance_remainder = vt @ Sigma @ vt

        lower[t] = abs(mean_t)
        upper[t] = (
            abs(mean_t)
            + C2 * covariance_remainder / sigma_inf
        )

        # Exact W_2 distance between one-dimensional Gaussian laws.
        w2[t] = np.hypot(mean_t, sigma_t - sigma_inf)

        Sigma_t = Q @ Sigma_t @ Q.T + C
        Qt = Q @ Qt

    return {
        "t": time,
        "w2": w2,
        "lower": lower,
        "upper": upper,
    }


def plot_wasserstein(data, v=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        data["t"],
        data["w2"],
        "o",
        ms=4,
        color="0.2",
        label=r"$W_2$",
    )
    ax.plot(
        data["t"],
        data["lower"],
        "--",
        lw=1.5,
        label="lower bound",
    )
    ax.plot(
        data["t"],
        data["upper"],
        "--",
        lw=1.5,
        label="upper bound",
    )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"projected $W_2$")
    
    if v is not None:
        ax.set_title(rf"Projection along $v={np.asarray(v)}$")

    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    return fig, ax


if __name__ == "__main__":

    Q = np.array([
        [49, -28, -10],
        [-28, 25, -38],
        [-10, -38, 16],
    ], dtype=float) / 90.0

    C = np.eye(3)
    x0 = np.ones(3)
    v = np.ones(3)

    data = projected_wasserstein(Q, C, v, x0, T=20)

    plot_wasserstein(data, v)
    plt.show()
