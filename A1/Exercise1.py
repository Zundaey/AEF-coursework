# mandatory_assignment1_ex1.py

import tidyfinance as tf
import pandas as pd
import numpy as np

# Set random seed for reproducibility (used later in simulations)
np.random.seed(2026)

# 1. Download daily data from Yahoo!Finance
tickers = ["^GSPC", "^IRX"]

data = tf.download_data(
    domain="stock_prices",
    symbols=tickers,
    start_date="2000-01-01",
    end_date="2026-02-28"
)

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# 2. Compute monthly returns for S&P 500
sp500 = data[data['symbol'] == '^GSPC']['close'].resample('M').last()
sp500_returns = sp500.pct_change().dropna()

# 3. Convert 13-week T-bill annualized yield to monthly rate
t_bill = data[data['symbol'] == '^IRX']['close'].resample('M').last() / 100
t_bill_monthly_rate = (1 + t_bill)**(1/12) - 1

# 4. Compute monthly excess market return
excess_market_return = sp500_returns - t_bill_monthly_rate

# 5. Estimate mean and variance (monthly)
mu_m = excess_market_return.mean()
sigma2_m = excess_market_return.var()

# Annualized equivalents
mu_m_annual = mu_m * 12
sigma2_m_annual = sigma2_m * 12

# Combine results into a single DataFrame
monthly_data = pd.DataFrame({
    'SP500_Returns': sp500_returns,
    'TBill_13w_Monthly': t_bill_monthly_rate,
    'Excess_Market_Return': excess_market_return
})

# Display first rows and summary stats
print("First 5 rows of monthly data:\n", monthly_data.head())
print("\nMonthly mean excess return:", mu_m)
print("Monthly return variance:", sigma2_m)
print("Annualized mean excess return:", mu_m_annual)
print("Annualized return variance:", sigma2_m_annual)