# Sliced Wasserstein Distance Bounds Simulation

This repository implements a simulation framework for bounding the **Sliced Wasserstein Distance** in discrete-time linear dynamical systems (Autoregressive processes). The code validates theoretical bounds against the computed Wasserstein distance over time.

## Mathematical Context

The simulation follows the process $X_t = Q X_{t-1} + \xi_t$, where $Q$ is the contraction matrix and $\xi_t$ represents the noise term. Convergence bounds are projected along a direction vector $v$.

## Assumptions & Extensions

**Gaussianity (Default)**
The current implementation (`run_w2_simulation`) uses the analytical closed-form solution for the 2-Wasserstein distance:
$$W_2^2 = (\mu_t - \mu_\infty)^2 + (\sigma_t - \sigma_\infty)^2$$
This offers high computational efficiency but is strictly valid only for Gaussian noise. 

**Non-Gaussian Support**
For general distributions (e.g., Uniform, Laplacian), the framework supports extension via numerical methods:
1.  **Known $F^{-1}$:** Numerical integration of the quantile difference $|F_t^{-1} - {F_\infty^{-1}}|^r$.
2.  **Unknown $F^{-1}$:** Empirical quantile estimation using Monte Carlo particle simulations.

*Note: The core matrix evolution logic remains valid for any distribution; only the distance metric calculation requires adaptation.*

## Features

- **Lyapunov Solver**: Computes stationary covariance $\Sigma_\infty$ automatically.
- **Theoretical Bounds**: Calculates rigorous Upper and Lower convergence bounds.
- **Visualization**: Time-series plotting of distance evolution.
- **Modular Design**: Tested with ARMA(2,1), Symmetric, and Non-Diagonalizable systems.

## Usage

Ensure dependencies are installed:
```bash
pip install -r requirements.txt
