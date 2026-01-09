import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

df = pd.read_csv(
    "ASI_close_prices_last_15_years_2025-06-05.csv",
    parse_dates=["date"]
)

df = df.sort_values("date").set_index("date")

# Log price & log returns
df["log_close"] = np.log(df["close"])
df["log_returns"] = df["log_close"].diff()

df = df.dropna()

HORIZON = 10

train = df.iloc[:-HORIZON]
valid = df.iloc[-HORIZON:]

y_train = train["log_returns"]
y_valid = valid["log_returns"]

model = ARIMA(
    y_train,
    order=(1, 0, 1),
    enforce_stationarity=True,
    enforce_invertibility=True
)

model_fit = model.fit()
print(model_fit.summary())

forecast_log_returns = model_fit.forecast(steps=HORIZON)
forecast_log_returns.index = valid.index

last_train_price = train["close"].iloc[-1]

# Convert log-returns → prices
forecast_prices = [last_train_price]

for r in forecast_log_returns:
    forecast_prices.append(forecast_prices[-1] * np.exp(r))

forecast_prices = forecast_prices[1:]
forecast_prices = pd.Series(forecast_prices, index=valid.index)

actual_prices = valid["close"]

mape = mean_absolute_percentage_error(actual_prices, forecast_prices)
print(f"Validation MAPE (10-day): {mape:.4%}")

in_sample_pred = model_fit.fittedvalues

train_prices_recon = [train["close"].iloc[0]]

for r in in_sample_pred[1:]:
    train_prices_recon.append(train_prices_recon[-1] * np.exp(r))

train_prices_recon = pd.Series(
    train_prices_recon,
    index=train.index
)

train_mape = mean_absolute_percentage_error(
    train["close"], train_prices_recon
)

print(f"Train MAPE: {train_mape:.4%}")
print(f"Validation MAPE: {mape:.4%}")

plt.figure(figsize=(12,5))
plt.plot(train.index[-100:], train["close"].iloc[-100:], label="Train (last 100)")
plt.plot(valid.index, actual_prices, label="Actual", marker="o")
plt.plot(valid.index, forecast_prices, label="Forecast", marker="o")
plt.legend()
plt.title("10-Day Price Forecast – ARIMA(1,0,1)")
plt.show()

final_model = ARIMA(
    df["log_returns"],
    order=(1, 0, 1),
    enforce_stationarity=True,
    enforce_invertibility=True
).fit()

with open("arima_log_returns_model.pkl", "wb") as f:
    pickle.dump(final_model, f)
