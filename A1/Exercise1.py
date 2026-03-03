import tidyfinance as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For reproducibility
np.random.seed(2026)

# Helper function for Question 2
def simulate_returns(periods, expected_returns, covariance_matrix):
    """
    Simulate asset returns from a multivariate normal distribution.

    periods: int, number of time periods to simulate
    expected_returns: array-like, true mean returns (mu)
    covariance_matrix: array-like, true covariance matrix (Sigma)
    """
    returns = np.random.multivariate_normal(expected_returns, covariance_matrix, size=periods)
    return returns

#Question 1 --------------------------------------
start_date = "2000-01-01"
end_date   = "2026-02-28"

# Download daily data
sp500 = tf.download_data(domain="stock_prices", symbols="^GSPC", start_date=start_date, end_date=end_date)
tbill = tf.download_data(domain="stock_prices", symbols="^IRX", start_date=start_date, end_date=end_date)

# Convert date column to datetime
sp500['date'] = pd.to_datetime(sp500['date'])
tbill['date'] = pd.to_datetime(tbill['date'])

# Set as index
sp500 = sp500.set_index('date')
tbill = tbill.set_index('date')
# Resample to monthly frequency and calculate returns
sp500_monthly = (
    sp500['close']
    .resample('ME')
    .last()
    .pct_change()
)

tbill['MonthlyRate'] = (1 + tbill['close'] / 100)**(1/12) - 1
tbill_monthly = (
    tbill['MonthlyRate']
    .resample('ME')
    .last()
)
# Combine into a single DataFrame
data = pd.DataFrame({
    "SP500_Return": sp500_monthly,
    "TBill_Return": tbill_monthly
}).dropna()

data["Excess_Return"] = data["SP500_Return"] - data["TBill_Return"]

mu_m = data["Excess_Return"].mean()
sigma2_m = data["Excess_Return"].var()

mu_annual = 12 * mu_m
sigma2_annual = 12 * sigma2_m

#print output
print("Monthly mean excess return:", mu_m)
print("Monthly variance:", sigma2_m)
print("Annualized mean excess return:", mu_annual)
print("Annualized variance:", sigma2_annual)

#Question 2 ----------------------------------------
N = 50
beta = np.random.uniform(0.5, 1.5, size=N)
sigma_eps = np.random.uniform(0.10/np.sqrt(12), 0.30/np.sqrt(12), size=N)

mu = beta * mu_m  # mu_m from Question 1
Sigma = sigma2_m * np.outer(beta, beta) + np.diag(sigma_eps**2)

T = len(data)  # number of months from your Question 1 dataset
simulated_returns = simulate_returns(T, mu, Sigma)

#Question 3 ----------------------------------------

def closed_form_frontier(mu, Sigma, R_grid=None, n_points=200):
    """
    Compute mean-variance efficient frontier and tangency portfolio.

    Parameters
    ----------
    mu : array-like, shape (N,)
        Expected returns (monthly, excess returns).
    Sigma : array-like, shape (N, N)
        Covariance matrix (monthly).
    R_grid : array-like or None
        Grid of target expected returns (monthly). If None, a grid spanning
        [min(mu) - 0.5*sd(mu), max(mu) + 0.5*sd(mu)] is created with n_points.
    n_points : int
        Number of points to create on R_grid when R_grid is None.

    Returns
    -------
    R_grid : np.ndarray, shape (nR,)
        Grid of target expected returns used.
    sigma : np.ndarray, shape (nR,)
        Portfolio risk (monthly std dev) on the frontier for each R_grid point.
    W : np.ndarray, shape (nR, N)
        Portfolio weights for each R on the frontier (rows sum to 1).
    w_t : np.ndarray, shape (N,)
        Tangency portfolio weights (sum to 1).
    R_t : float
        Tangency portfolio expected return (monthly).
    sigma_t : float
        Tangency portfolio risk (monthly std dev).
    """
    mu = np.asarray(mu).reshape(-1)
    Sigma = np.asarray(Sigma)
    N = mu.shape[0]
    ones = np.ones(N)

    # Solve Sigma x = rhs instead of inverting explicitly
    invSigma_ones = np.linalg.solve(Sigma, ones)
    invSigma_mu = np.linalg.solve(Sigma, mu)

    A = ones @ invSigma_ones
    B = ones @ invSigma_mu
    C = mu @ invSigma_mu
    D = A * C - B * B
    if D <= 0:
        raise np.linalg.LinAlgError("Non-positive D (check Sigma or numerical conditioning).")

    # Build R_grid if not provided
    if R_grid is None:
        mmin, mmax = mu.min(), mu.max()
        spread = max(1e-6, 0.5 * np.std(mu))
        R_grid = np.linspace(mmin - spread, mmax + spread, n_points)
    R_grid = np.asarray(R_grid).reshape(-1)
    nR = R_grid.size

    # compute coefficients alpha and beta for each R
    alpha = (C - B * R_grid) / D    # shape (nR,)
    beta = (A * R_grid - B) / D     # shape (nR,)

    # frontier weights: w(R) = alpha * Σ^{-1}1 + beta * Σ^{-1}μ
    W = alpha[:, None] * invSigma_ones[None, :] + beta[:, None] * invSigma_mu[None, :]

    # variance formula: var = (A R^2 - 2 B R + C) / D
    var_grid = (A * R_grid**2 - 2 * B * R_grid + C) / D
    # numerical safety: small negative rounding error -> clip
    var_grid = np.clip(var_grid, 0.0, None)
    sigma = np.sqrt(var_grid)

    # tangency portfolio (excess returns): proportional to Sigma^{-1} mu, scaled to sum=1
    w_t_unnorm = invSigma_mu
    denom = ones @ w_t_unnorm
    if np.isclose(denom, 0.0):
        raise np.linalg.LinAlgError("Degenerate tangency scaling (ones' Sigma^{-1} mu ~ 0).")
    w_t = w_t_unnorm / denom
    R_t = mu @ w_t
    sigma_t = np.sqrt(w_t @ (Sigma @ w_t))

    return R_grid, sigma, W, w_t, R_t, sigma_t

# Question 4 ----------------------------------------
# python
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(sigma_true, R_true, 'k-', lw=2)
ax.set_xlabel('Monthly risk (std. dev.)')
ax.set_ylabel('Monthly expected excess return')
ax.set_title('Question 4 — True efficient frontier (from μ and Σ)')
ax.grid(alpha=0.25)
plt.tight_layout()