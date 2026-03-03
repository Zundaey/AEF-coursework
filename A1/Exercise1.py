import tidyfinance as tf
import pandas as pd
import numpy as np

# For reproducibility
np.random.seed(2026)

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