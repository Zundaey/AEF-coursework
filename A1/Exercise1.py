import tidyfinance as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For reproducibility — must be set at the top of the script (Exercise 5 footnote)
np.random.seed(2026)

# simulate_returns helper (provided in Exercise 5, placed here for use throughout)
def simulate_returns(periods, expected_returns, covariance_matrix):
    """
    Simulate asset returns from a multivariate normal distribution.

    periods:           int, number of time periods to simulate
    expected_returns:  array-like, true mean returns (mu)
    covariance_matrix: array-like, true covariance matrix (Sigma)
    """
    returns = np.random.multivariate_normal(expected_returns, covariance_matrix, size=periods)
    return returns


# ===========================================================================
# Exercise 1 — Market data and excess returns
# ===========================================================================
start_date = "2000-01-01"
end_date   = "2026-02-28"

# Download daily price data for S&P 500 and 13-week T-Bill yield
sp500 = tf.download_data(domain="stock_prices", symbols="^GSPC", start_date=start_date, end_date=end_date)
tbill = tf.download_data(domain="stock_prices", symbols="^IRX",  start_date=start_date, end_date=end_date)

# Parse dates and set as index
sp500['date'] = pd.to_datetime(sp500['date'])
tbill['date'] = pd.to_datetime(tbill['date'])
sp500 = sp500.set_index('date')
tbill = tbill.set_index('date')

# Resample to month-end and compute monthly S&P 500 returns
sp500_monthly = sp500['close'].resample('ME').last().pct_change()

# Convert annualised T-Bill yield (%) to a monthly rate: r_f = (1 + IRX/100)^(1/12) - 1
tbill['MonthlyRate'] = (1 + tbill['close'] / 100) ** (1 / 12) - 1
tbill_monthly = tbill['MonthlyRate'].resample('ME').last()

# Merge and drop missing observations
data = pd.DataFrame({
    "SP500_Return": sp500_monthly,
    "TBill_Return": tbill_monthly
}).dropna()

# Compute monthly excess market returns
data["Excess_Return"] = data["SP500_Return"] - data["TBill_Return"]

# Estimate monthly mean and variance of excess returns
mu_m     = data["Excess_Return"].mean()
sigma2_m = data["Excess_Return"].var()

# Annualise: multiply means by 12, variances by 12
mu_annual     = 12 * mu_m
sigma2_annual = 12 * sigma2_m

print("=== Exercise 1 ===")
print(f"Monthly mean excess return:      {mu_m:.4f}")
print(f"Monthly variance:                {sigma2_m:.6f}")
print(f"Annualized mean excess return:   {mu_annual:.4f}")
print(f"Annualized variance:             {sigma2_annual:.6f}")


# ===========================================================================
# Exercise 2 — Construct 50 synthetic assets via single-factor model
# ===========================================================================
N = 50

# Draw betas and idiosyncratic volatilities from uniform distributions
beta      = np.random.uniform(0.5, 1.5, size=N)
sigma_eps = np.random.uniform(0.10 / np.sqrt(12), 0.30 / np.sqrt(12), size=N)

# True monthly parameters (treated as ground truth for the rest of the assignment)
mu    = beta * mu_m                                             # shape (N,)
Sigma = sigma2_m * np.outer(beta, beta) + np.diag(sigma_eps**2)  # shape (N, N)

# Simulate one sample (T months) to verify the setup
T = len(data)
simulated_returns = simulate_returns(T, mu, Sigma)

print("\n=== Exercise 2 ===")
print(f"N = {N} assets, T = {T} months")
print(f"beta range:      [{beta.min():.3f}, {beta.max():.3f}]")
print(f"sigma_eps range: [{sigma_eps.min():.4f}, {sigma_eps.max():.4f}]")


# ===========================================================================
# Exercise 3 — Mean-variance efficient frontier and tangency portfolio
# ===========================================================================
def closed_form_frontier(mu, Sigma, R_grid=None, n_points=200):
    """
    Compute the mean-variance efficient frontier and tangency portfolio
    analytically using the closed-form solution.

    Parameters
    ----------
    mu    : array-like, shape (N,)   — expected excess returns (monthly)
    Sigma : array-like, shape (N, N) — covariance matrix (monthly)
    R_grid: array-like or None       — target return grid; created automatically if None
    n_points: int                    — grid resolution when R_grid is None

    Returns
    -------
    R_grid   : np.ndarray (nR,)  — target return grid used
    sigma    : np.ndarray (nR,)  — frontier portfolio std dev
    W        : np.ndarray (nR,N) — frontier portfolio weights (rows sum to 1)
    w_t      : np.ndarray (N,)   — tangency portfolio weights
    R_t      : float             — tangency portfolio expected return
    sigma_t  : float             — tangency portfolio std dev
    """
    mu    = np.asarray(mu).reshape(-1)
    Sigma = np.asarray(Sigma)
    N_    = mu.shape[0]
    ones  = np.ones(N_)

    # Solve linear systems instead of inverting Sigma explicitly (numerically safer)
    invSigma_ones = np.linalg.solve(Sigma, ones)
    invSigma_mu   = np.linalg.solve(Sigma, mu)

    # Scalar frontier constants
    A = ones @ invSigma_ones   # 1' Σ^{-1} 1
    B = ones @ invSigma_mu     # 1' Σ^{-1} μ
    C = mu   @ invSigma_mu     # μ' Σ^{-1} μ
    D = A * C - B ** 2         # determinant-like term; must be > 0

    if D <= 0:
        raise np.linalg.LinAlgError("Non-positive D — check Sigma or numerical conditioning.")

    # Build return grid spanning the asset return range
    if R_grid is None:
        mmin, mmax = mu.min(), mu.max()
        spread = max(1e-6, 0.5 * np.std(mu))
        R_grid = np.linspace(mmin - spread, mmax + spread, n_points)
    R_grid = np.asarray(R_grid).reshape(-1)

    # Frontier weight coefficients for each target return
    alpha_coef = (C - B * R_grid) / D    # shape (nR,)
    beta_coef  = (A * R_grid - B) / D    # shape (nR,)

    # Frontier weights: w(R) = α Σ^{-1}1 + β Σ^{-1}μ
    W = alpha_coef[:, None] * invSigma_ones[None, :] + beta_coef[:, None] * invSigma_mu[None, :]

    # Frontier variance: var(R) = (A R^2 - 2B R + C) / D
    var_grid = (A * R_grid**2 - 2 * B * R_grid + C) / D
    var_grid = np.clip(var_grid, 0.0, None)   # clip tiny negative rounding errors
    sigma = np.sqrt(var_grid)

    # Tangency portfolio: proportional to Σ^{-1}μ, scaled to sum to 1
    denom = ones @ invSigma_mu
    if np.isclose(denom, 0.0):
        raise np.linalg.LinAlgError("Degenerate tangency: 1'Σ^{-1}μ ≈ 0.")
    w_t     = invSigma_mu / denom
    R_t     = mu @ w_t
    sigma_t = np.sqrt(w_t @ (Sigma @ w_t))

    return R_grid, sigma, W, w_t, R_t, sigma_t


# Compute the true efficient frontier using the true parameters from Exercise 2
R_true, sigma_true, W_true, w_t_true, R_t_true, sigma_t_true = closed_form_frontier(mu, Sigma)

print("\n=== Exercise 3 ===")
print(f"True tangency return (monthly):  {R_t_true:.4f}  (annualized: {R_t_true * 12:.4f})")
print(f"True tangency std dev (monthly): {sigma_t_true:.4f}  (annualized: {sigma_t_true * np.sqrt(12):.4f})")
print(f"True tangency Sharpe (monthly):  {R_t_true / sigma_t_true:.4f}")


# ===========================================================================
# Exercise 4 — Visualise the true efficient frontier
# ===========================================================================
fig, ax = plt.subplots(figsize=(7, 5))

# Plot frontier (annualised units for readability)
ax.plot(sigma_true * np.sqrt(12), R_true * 12, 'k-', lw=2, label='Efficient frontier')

# Mark the tangency portfolio
ax.scatter(
    sigma_t_true * np.sqrt(12), R_t_true * 12,
    color='red', zorder=5, s=100, label='Tangency portfolio'
)

ax.set_xlabel('Annualized std. dev.')
ax.set_ylabel('Annualized expected excess return')
ax.set_title('Exercise 4 — True efficient frontier (μ and Σ)')
ax.legend()
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('A1/q4_frontier.png', dpi=150)
plt.close()


# ===========================================================================
# Exercise 5 — Estimation error: how N/T affects the plug-in frontier
# ===========================================================================
# We draw many samples of size T from the true DGP and compute the plug-in
# frontier for each sample. As N/T increases, estimation error worsens.
T_values     = [60, 120, 252, 504]   # sample sizes to study (months)
n_sim_q5     = 100                   # number of Monte Carlo replications per T

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Exercise 5 — Plug-in frontier vs true frontier (100 simulations each)', fontsize=13)

for ax, T_sim in zip(axes.flat, T_values):
    # Overlay plug-in frontiers estimated from simulated samples.
    # Use R_true as the fixed return grid so all frontiers are comparable.
    for _ in range(n_sim_q5):
        sim_ret   = simulate_returns(T_sim, mu, Sigma)
        mu_hat    = sim_ret.mean(axis=0)
        Sigma_hat = np.cov(sim_ret.T)
        try:
            _, sigma_sim, _, _, _, _ = closed_form_frontier(mu_hat, Sigma_hat, R_grid=R_true)
            ax.plot(sigma_sim * np.sqrt(12), R_true * 12,
                    color='steelblue', alpha=0.15, lw=0.8)
        except Exception:
            pass  # skip degenerate samples

    # True frontier plotted on top as black reference line
    ax.plot(sigma_true * np.sqrt(12), R_true * 12, 'k-', lw=2, label='True frontier', zorder=5)

    ax.set_title(f'T = {T_sim} months  (N/T = {N / T_sim:.2f})')
    ax.set_xlabel('Annualized std. dev.')
    ax.set_ylabel('Annualized excess return')
    # Clip x-axis so outlier plug-in frontiers (small T) don't distort the scale
    ax.set_xlim(left=0, right=sigma_true.max() * np.sqrt(12) * 3)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig('A1/q5_estimation_error.png', dpi=150)
plt.close()

print("\n=== Exercise 5 ===")
print("Estimation error is driven by the ratio N/T.")
print("As T falls (or N grows), plug-in frontiers increasingly overstate the attainable Sharpe ratio.")


# ===========================================================================
# Exercise 6 — Out-of-sample performance: plug-in tangency vs true optimum
# ===========================================================================
# For each simulated sample we compute the plug-in tangency weights and
# evaluate their Sharpe ratio using the TRUE parameters (out-of-sample).
n_sim_q6   = 500
true_sharpe = R_t_true / sigma_t_true   # monthly Sharpe of the oracle tangency portfolio

results_q6 = {T: [] for T in T_values}

for T_sim in T_values:
    for _ in range(n_sim_q6):
        sim_ret   = simulate_returns(T_sim, mu, Sigma)
        mu_hat    = sim_ret.mean(axis=0)
        Sigma_hat = np.cov(sim_ret.T)
        try:
            _, _, _, w_hat, _, _ = closed_form_frontier(mu_hat, Sigma_hat)
            # Evaluate with TRUE parameters (out-of-sample)
            R_oos   = mu @ w_hat
            var_oos = w_hat @ Sigma @ w_hat
            results_q6[T_sim].append(R_oos / np.sqrt(var_oos))
        except Exception:
            pass

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(
    [results_q6[T] for T in T_values],
    tick_labels=[str(T) for T in T_values],
    patch_artist=True,
    boxprops=dict(facecolor='steelblue', alpha=0.6)
)
ax.axhline(true_sharpe, color='red', linestyle='--', lw=1.5,
           label=f'True tangency Sharpe = {true_sharpe:.3f}')
ax.set_xlabel('Sample size T (months)')
ax.set_ylabel('Monthly Sharpe ratio (out-of-sample, true params)')
ax.set_title('Exercise 6 — OOS Sharpe ratio of plug-in tangency portfolio')
ax.legend()
ax.grid(alpha=0.25, axis='y')
plt.tight_layout()
plt.savefig('A1/q6_oos_sharpe.png', dpi=150)
plt.close()

print("\n=== Exercise 6 ===")
for T_sim in T_values:
    vals = results_q6[T_sim]
    print(f"T={T_sim:>4}: median OOS Sharpe = {np.median(vals):.4f}  "
          f"(true = {true_sharpe:.4f}, ratio = {np.median(vals)/true_sharpe:.2f})")


# ===========================================================================
# Exercise 7 — Alternative portfolio strategies
# ===========================================================================

# --- Strategy implementations ---

def portfolio_equal_weight(N):
    """
    Naive diversification (1/N): equal weight for every asset.
    Intuition: ignores all parameter estimates entirely, so estimation
    error cannot compound — a natural lower bound for robustness.
    """
    return np.ones(N) / N


def portfolio_min_variance(Sigma):
    """
    Global minimum-variance (GMV) portfolio: minimise w'Σw s.t. Σw=1.
    Intuition: avoids estimating expected returns (the noisiest input)
    and focuses only on covariance structure, which is estimated more
    precisely from return data.
    """
    ones          = np.ones(Sigma.shape[0])
    invSigma_ones = np.linalg.solve(Sigma, ones)
    return invSigma_ones / (ones @ invSigma_ones)


def portfolio_volatility_parity(Sigma):
    """
    Volatility parity (inverse-volatility weighting): weights are
    inversely proportional to each asset's individual standard deviation.
    Intuition: equalises the standalone risk contribution of each asset
    without requiring an explicit covariance model inversion, making it
    more robust to estimation noise than GMV.
    """
    std = np.sqrt(np.diag(Sigma))
    w   = 1.0 / std
    return w / w.sum()


def portfolio_factor(beta_vec):
    """
    Factor portfolio: weights proportional to the asset's market beta.
    Intuition: in a single-factor world the only priced risk is market
    exposure (beta). Allocating proportionally to beta is therefore a
    theoretically motivated strategy that exploits the known factor
    structure and does not require estimating the full N×N covariance
    matrix.
    """
    return beta_vec / beta_vec.sum()


def sharpe_ratio_true(w):
    """Evaluate Sharpe ratio of portfolio w using the TRUE parameters."""
    R   = mu @ w
    vol = np.sqrt(w @ Sigma @ w)
    return R / vol


# --- Simulation ---
n_sim_q7 = 500

strategies = {
    'Equal weight (1/N)':  lambda mu_h, Sigma_h: portfolio_equal_weight(N),
    'Min. variance':       lambda mu_h, Sigma_h: portfolio_min_variance(Sigma_h),
    'Volatility parity':   lambda mu_h, Sigma_h: portfolio_volatility_parity(Sigma_h),
    'Factor portfolio':    lambda mu_h, Sigma_h: portfolio_factor(beta),          # uses true beta
    'Plug-in tangency':    lambda mu_h, Sigma_h: closed_form_frontier(mu_h, Sigma_h)[3],
}

results_q7 = {name: {T: [] for T in T_values} for name in strategies}

for T_sim in T_values:
    for _ in range(n_sim_q7):
        sim_ret   = simulate_returns(T_sim, mu, Sigma)
        mu_hat    = sim_ret.mean(axis=0)
        Sigma_hat = np.cov(sim_ret.T)
        for name, fn in strategies.items():
            try:
                w  = fn(mu_hat, Sigma_hat)
                sr = sharpe_ratio_true(w)
                results_q7[name][T_sim].append(sr)
            except Exception:
                pass

# --- Plot: median OOS Sharpe across T values ---
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for name, color in zip(strategies.keys(), colors):
    medians = [np.median(results_q7[name][T]) for T in T_values]
    ax.plot(T_values, medians, marker='o', label=name, color=color, lw=2)

ax.axhline(true_sharpe, color='k', linestyle='--', lw=1.5, label='True tangency (oracle)')
ax.set_xlabel('Sample size T (months)')
ax.set_ylabel('Median monthly Sharpe ratio (OOS, true params)')
ax.set_title('Exercise 7 — Strategy comparison across sample sizes')
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('A1/q7_strategy_comparison.png', dpi=150)
plt.close()

# --- Summary table ---
print("\n=== Exercise 7 — Median OOS Sharpe ratios ===")
header = f"{'Strategy':<25}" + "".join([f"{'T='+str(T):>10}" for T in T_values])
print(header)
print("-" * len(header))
for name in strategies:
    row = f"{name:<25}" + "".join([f"{np.median(results_q7[name][T]):>10.4f}" for T in T_values])
    print(row)
print(f"\n{'True tangency (oracle)':<25}" + "".join([f"{true_sharpe:>10.4f}" for _ in T_values]))
