"""Projected Wasserstein-2 distance for a Gaussian autoregressive process.

    X_t = Q X_{t-1} + eta_t,   X_0 = x0,   eta_t ~ N(0, C) i.i.d.

C denotes the innovation covariance Sigma Xi Sigma^T in the manuscript.
"""

import numpy as np
from scipy.linalg import solve_discrete_lyapunov


def simulate_wasserstein(Q, C, v, x0, max_time=100):
    """Compute W2(v^T X_t, v^T X_inf) and its bounds for t = 0, ..., max_time.

    Q must be Schur stable and C positive semidefinite. The projection v
    is not normalized; its stationary variance must be positive.
    Return a dictionary of arrays with keys 't', 'lb', 'ub', and 'w2'.
    """
    Q = np.asarray(Q, dtype=float)
    C = np.asarray(C, dtype=float)
    v = np.asarray(v, dtype=float)
    x0 = np.asarray(x0, dtype=float)

    if Q.ndim != 2 or Q.shape[0] == 0 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a nonempty square matrix.")
    d = Q.shape[0]
    if C.shape != (d, d) or v.shape != (d,) or x0.shape != (d,):
        raise ValueError("C, v and x0 must have dimensions compatible with Q.")
    if not all(np.isfinite(a).all() for a in (Q, C, v, x0)):
        raise ValueError("All inputs must be finite.")
    if not np.any(v):
        raise ValueError("v must be nonzero.")
    if (
        not isinstance(max_time, (int, np.integer))
        or isinstance(max_time, (bool, np.bool_))
        or max_time < 0
    ):
        raise ValueError("max_time must be a nonnegative integer.")
    if np.max(np.abs(np.linalg.eigvals(Q))) >= 1.0:
        raise ValueError("Q must be Schur stable: spectral radius < 1.")
    if not np.allclose(C, C.T, rtol=1e-12, atol=0.0):
        raise ValueError("C must be symmetric.")
    C = (C + C.T) / 2.0
    eps = np.finfo(float).eps
    tol_C = 100 * eps * np.linalg.norm(C, ord=2)
    if np.linalg.eigvalsh(C).min() < -tol_C:
        raise ValueError("C must be positive semidefinite.")

    Sigma_inf = solve_discrete_lyapunov(Q, C)
    Sigma_inf = (Sigma_inf + Sigma_inf.T) / 2.0
    var_inf = float(v @ Sigma_inf @ v)
    if not np.isfinite(var_inf) or var_inf <= 0.0:
        raise ValueError("The stationary projected variance must be positive.")
    sigma_inf = np.sqrt(var_inf)
    tol_var = 100 * eps * var_inf

    result = {"t": np.arange(max_time + 1)}
    for name in ("lb", "ub", "w2"):
        result[name] = np.empty(max_time + 1)

    Q_t = np.eye(d)
    Sigma_t = np.zeros((d, d))

    for t in range(max_time + 1):
        mu_t = float(v @ Q_t @ x0)
        var_t = float(v @ Sigma_t @ v)

        # gamma_t = v^T (Sigma_inf - Sigma_t) v.
        u_t = Q_t.T @ v
        gamma_t = float(u_t @ Sigma_inf @ u_t)

        if var_t < -tol_var or gamma_t < -tol_var:
            raise FloatingPointError("Negative variance or covariance tail.")
        sigma_t = np.sqrt(max(var_t, 0.0))
        gamma_t = max(gamma_t, 0.0)

        # Rationalize sigma_inf - sigma_t to avoid cancellation at large t.
        delta_sigma = gamma_t / (sigma_inf + sigma_t)

        result["lb"][t] = abs(mu_t)
        result["ub"][t] = abs(mu_t) + gamma_t / sigma_inf
        result["w2"][t] = np.hypot(mu_t, delta_sigma)

        if t < max_time:
            Sigma_t = Sigma_t + Q_t @ C @ Q_t.T
            Sigma_t = (Sigma_t + Sigma_t.T) / 2.0
            Q_t = Q @ Q_t

    return result


def plot_results(result):
    """Plot the distance and bounds."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        result["t"], result["w2"], "o", color="0.2",
        markersize=4, label=r"Projected $\mathcal{W}_2$",
    )
    ax.plot(result["t"], result["lb"], "--", label="Lower bound")
    ax.plot(result["t"], result["ub"], "--", label="Upper bound")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(
        xlabel="Time step", ylabel="Distance",
        title="Fixed-direction Gaussian Wasserstein-2 distance",
    )
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Symmetric example from the manuscript.
    Q = np.array([
        [49.0, -28.0, -10.0],
        [-28.0, 25.0, -38.0],
        [-10.0, -38.0, 16.0],
    ]) / 90.0
    C = np.eye(3)
    x0 = np.array([2.0, -1.0, 1.5])
    v = np.ones(3)
    T = 20

    results = simulate_wasserstein(Q, C, v, x0, max_time=T)
    plot_results(results)
    plt.show()
