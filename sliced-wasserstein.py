import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gamma
from scipy.linalg import solve_discrete_lyapunov
from matplotlib.ticker import MaxNLocator

def get_bound_coeff(r: float = 2.0) -> float:
    return (np.sqrt(2) * gamma((r + 1) / 2)**(1/r)) / (np.pi**(1/(2*r)))

def simulate_wasserstein(Q, Sigma, v, x0, max_time=100):
    """
    Simulates the W2 distance evolution for an AR(1) process compared to 
    its stationary limit.
    """
    v = np.asarray(v, dtype=float)
    if np.allclose(v, 0):
        raise ValueError("Direction vector v cannot be zero.")

    # Stationary parameters
    Sigma_inf = solve_discrete_lyapunov(Q, Sigma)
    
    # Pre-compute stationary projection stats
    var_inf = v @ Sigma_inf @ v
    std_inf = np.sqrt(var_inf)
    coeff = get_bound_coeff(r=2.0) 

    # Initialization
    results = []
    
    # State variables for iteration (avoiding repeated matrix_power)
    Q_t = np.eye(len(Q))        # Represents Q^t
    Sigma_t = np.zeros_like(Q)  # Accumulator for finite-time covariance
    
    # We iterate t from 0 to max_time
    for t in range(max_time + 1):
        # 1. Compute projections for current time t
        mu_t = v @ Q_t @ x0
        
        # Variance of the process at time t (projected)
        var_t = v @ Sigma_t @ v
        std_t = np.sqrt(var_t)
        
        # 2. Compute Bounds and W2
        # Lower Bound: |<v, Q^t x0>|
        lb = np.abs(mu_t)
        
        # Upper Bound calculation
        v_prime = Q_t @ v
        term_geo = v_prime @ Sigma_inf @ v_prime
        ub = lb + (coeff / std_inf) * term_geo
        
        # Actual W2 (Gaussian case: Euclidean distance of parameters)
        w2 = np.sqrt(mu_t**2 + (std_t - std_inf)**2)
        
        results.append({
            't': t,
            'lb': lb,
            'ub': ub,
            'w2': w2
        })

        # 3. Update states for t+1
        # Sigma_{t+1} = Sigma_t + Q^t * Sigma * (Q^t)^T
        # Note: We update Sigma_t first using current Q_t, then update Q_t
        term_new = Q_t @ Sigma @ Q_t.T
        Sigma_t += term_new
        
        Q_t = Q @ Q_t

    return pd.DataFrame(results)

def plot_results(df, v_label):
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['t'], df['w2'], 'o', label=r'Actual $\mathcal{W}_2$', color='#333333')
    ax.plot(df['t'], df['lb'], '--', label='Lower Bound', color='#1f77b4', alpha=0.8)
    ax.plot(df['t'], df['ub'], '--', label='Upper Bound', color='#d62728', alpha=0.8)
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Time step")
    ax.set_ylabel("Distance")
    ax.set_title(f"Convergence Bounds (v={v_label})")
    ax.legend(frameon=True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Setup dynamics
    Q = (1/90) * np.array([
        [49, -28, -10],
        [-28, 25, -38],
        [-10, -38, 16]
    ])
    Sigma = np.eye(3)
    x0 = np.array([1, 1, 1])
    
    v = [1, 1, 1]
    T = 20

    df = simulate_wasserstein(Q, Sigma, v, x0, max_time=T)
    plot_results(df, str(v))
