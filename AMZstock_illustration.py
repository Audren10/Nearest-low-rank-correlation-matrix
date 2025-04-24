###############################################################################
## Illustration : using Black and Schole, we forecast the AMZ stock in a time##
## horizon h.                                                                ##
## In practice, using n assets, we can forecast the trajectory of the whole  ##
## portfolio using the correlation matrix between all assets                 ##
###############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the CSV file
df = pd.read_csv("stocks.csv", parse_dates=True, index_col=0)
amzn_prices = df['AMZN'].dropna()

# 🔧 Fix the datetime index using day/month/year format
amzn_prices.index = pd.to_datetime(amzn_prices.index, dayfirst=True)

# 2. Compute log-returns and estimate volatility
log_returns = np.log(amzn_prices / amzn_prices.shift(1)).dropna()
sigma = log_returns.std() * np.sqrt(252)  # annualized volatility
r = 0.03  # constant risk-free rate
S0 = amzn_prices.iloc[-1]  # last observed price

# 3. Simulation parameters
horizon_days = 5000  # roughly 6 months of trading days
dt = 1 / 252
n_simulations = 100

# 4. Monte Carlo Simulation
sim_matrix = np.zeros((horizon_days, n_simulations))
for i in range(n_simulations):
    Z = np.random.randn(horizon_days)
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    sim_path = [S0]
    for z in diffusion:
        sim_path.append(sim_path[-1] * np.exp(drift + z))
    sim_matrix[:, i] = sim_path[1:]  # remove the repeated S0

# 5. Average over the 100 simulations
mean_forecast = sim_matrix.mean(axis=1)

# 6. Generate future dates
last_date = amzn_prices.index[-1]
future_dates = pd.bdate_range(start=last_date, periods=horizon_days+1, freq='C')[1:]
mean_series = pd.Series(mean_forecast, index=future_dates)

# 7. Display the result
plt.figure(figsize=(12, 6))
plt.plot(amzn_prices, label='AMZN', color='blue')
plt.plot(mean_series, label='Forecasting', color='green', linestyle='--')
plt.axvline(last_date, color='gray', linestyle='--', label='Start of extrapolation')
plt.title("AMZN Stock Price Extrapolation")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

