"""Projected W2 for a Gaussian AR(1) process, at integer times.
X[t+1] = Q @ X[t] + eta[t+1], with iid eta ~ N(0, C), X[0] = x0.
"""

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import gammaln


def get_bound_coeff(r: float = 2.0) -> float:
    """Return (E|Z|**r)**(1/r), where Z ~ N(0, 1)."""
    r = float(r)
    if not np.isfinite(r) or r < 1:
        raise ValueError("r must be finite and at least 1.")
    return float(np.exp(
        0.5 * np.log(2.0)
        + gammaln((r + 1.0) / 2.0) / r
        - np.log(np.pi) / (2.0 * r)
    ))


def simulate_wasserstein(Q, C, v, x0, max_time=100):
    """Compute projected W2 and bounds; C is the innovation covariance."""
    if any(np.iscomplexobj(a) for a in (Q, C, v, x0)):
        raise ValueError("Inputs must be real.")
    Q, C, v, x0 = [np.asarray(a, dtype=float)
                   for a in (Q, C, v, x0)]
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or Q.size == 0:
        raise ValueError("Q must be a nonempty square matrix.")
    d = Q.shape[0]
    if C.shape != (d, d) or v.shape != (d,) or x0.shape != (d,):
        raise ValueError("C, v and x0 must have compatible shapes.")
    if not all(np.all(np.isfinite(a)) for a in (Q, C, v, x0)):
        raise ValueError("Inputs must be finite real arrays.")
    if not np.any(v != 0.0):
        raise ValueError("v must be nonzero.")
    if (isinstance(max_time, (bool, np.bool_))
            or not isinstance(max_time, (int, np.integer))
            or max_time < 0):
        raise ValueError("max_time must be a nonnegative integer.")
    if np.max(np.abs(np.linalg.eigvals(Q))) >= 1.0:
        raise ValueError("Q must be Schur-stable.")

    eps = np.finfo(float).eps
    c_tol = 100.0 * d * eps * max(1.0, np.linalg.norm(C, 2))
    if np.linalg.norm(C - C.T, 2) > c_tol:
        raise ValueError("C must be symmetric.")
    C = 0.5 * (C + C.T)
    if np.linalg.eigvalsh(C)[0] < -c_tol:
        raise ValueError("C must be positive semidefinite.")

    Sigma_inf = solve_discrete_lyapunov(Q, C)
    Sigma_inf = 0.5 * (Sigma_inf + Sigma_inf.T)
    if not np.all(np.isfinite(Sigma_inf)):
        raise ValueError("The Lyapunov solution is not finite.")
    try:
        np.linalg.cholesky(Sigma_inf)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Sigma_inf must be positive definite.") from exc
    var_inf = float(v @ Sigma_inf @ v)
    if not np.isfinite(var_inf) or var_inf <= 0.0:
        raise ValueError("The projected stationary variance must be positive.")
    std_inf = np.sqrt(var_inf)
    coeff = get_bound_coeff(2.0)  # C_2 = 1.
    var_tol = (100.0 * d * eps * np.linalg.norm(Sigma_inf, 2)
               * float(v @ v))

    result = {"t": np.arange(max_time + 1)}
    result.update({name: np.empty(max_time + 1)
                   for name in ("lb", "ub", "w2")})
    A_t = np.eye(d)                  # Q**t (matrix power).
    P_t = np.zeros((d, d), dtype=float)

    for t in range(max_time + 1):
        mu_t = float(v @ A_t @ x0)
        var_t = float(v @ P_t @ v)
        w_t = A_t.T @ v              # Essential transpose.
        gamma_t = float(w_t @ Sigma_inf @ w_t)
        if var_t < -var_tol or gamma_t < -var_tol:
            raise FloatingPointError("Negative variance beyond roundoff.")
        if var_t > var_inf + var_tol:
            raise FloatingPointError("Finite-time variance exceeds its limit.")
        std_t = np.sqrt(np.clip(var_t, 0.0, var_inf))
        gamma_t = max(gamma_t, 0.0)

        result["lb"][t] = abs(mu_t)
        result["ub"][t] = abs(mu_t) + coeff * gamma_t / std_inf
        result["w2"][t] = np.hypot(mu_t, std_t - std_inf)

        if t < max_time:
            # Both updates use the current A_t = Q**t.
            P_t = P_t + A_t @ C @ A_t.T
            P_t = 0.5 * (P_t + P_t.T)
            A_t = Q @ A_t

    return result


def plot_results(result, v_label):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result["t"], result["w2"], "o", color="#333333",
            label=r"Projected $\mathcal{W}_2$ (Gaussian formula)")
    ax.plot(result["t"], result["lb"], "--", color="#1f77b4",
            label="Lower bound: mean contribution")
    ax.plot(result["t"], result["ub"], "--", color="#d62728",
            label="Upper bound: mean + covariance contribution")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Time step")
    ax.set_ylabel("Projected Wasserstein distance")
    ax.set_title(f"Fixed-direction bounds (v={v_label})")
    ax.grid(alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Q = np.array([
        [49, -28, -10],
        [-28, 25, -38],
        [-10, -38, 16],
    ], dtype=float) / 90.0
    C = np.eye(3)
    x0 = np.ones(3)
    v = np.ones(3)
    result = simulate_wasserstein(Q, C, v, x0, max_time=20)
    plot_results(result, str(v))
    plt.show()
