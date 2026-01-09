import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
import joblib


# Load data
df = pd.read_csv(
    "ASI_close_prices_last_15_years_2025-06-05.csv",
    parse_dates=["date"]
)

df = df.sort_values("date").set_index("date")

# Keep close price
df = df[["close"]].dropna()

df["log_close"] = np.log(df["close"])
df["log_return"] = df["log_close"].diff()

df = df.dropna()


train_size = int(len(df) * 0.8)

train = df.iloc[:train_size]
val   = df.iloc[train_size:]


model = ARIMA(
    train["log_return"],
    order=(1, 0, 0)
)

model_fit = model.fit()

print(model_fit.summary())

history = list(train["log_return"])
predictions = []

for t in range(len(val)):
    model = ARIMA(history, order=(1, 0, 0))
    model_fit = model.fit()
    
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    
    history.append(val["log_return"].iloc[t])

# Reconstruct price predictions
last_log_price = train["log_close"].iloc[-1]

log_price_preds = []
current_log_price = last_log_price

for r in predictions:
    current_log_price += r
    log_price_preds.append(current_log_price)

price_preds = np.exp(log_price_preds)


actual_prices = val["close"].values

mape = mean_absolute_percentage_error(actual_prices, price_preds)
print(f"Validation MAPE: {mape * 100:.2f}%")

# In-sample fit
train_fitted = model_fit.fittedvalues

train_mape = mean_absolute_percentage_error(
    np.exp(train["log_close"].iloc[1:]),
    np.exp(train["log_close"].iloc[0] + np.cumsum(train_fitted))
)

print(f"Train MAPE: {train_mape * 100:.2f}%")
print(f"Validation MAPE: {mape * 100:.2f}%")

residuals = model_fit.resid

plt.figure(figsize=(12,4))
plt.plot(residuals)
plt.title("Residuals (Should Look Like White Noise)")
plt.show()

final_model = ARIMA(
    df["log_return"],
    order=(1, 0, 0)
)

final_model_fit = final_model.fit()

forecast_returns = final_model_fit.forecast(steps=10)

last_log_price = df["log_close"].iloc[-1]

forecast_log_prices = last_log_price + np.cumsum(forecast_returns)
forecast_prices = np.exp(forecast_log_prices)

forecast_df = pd.DataFrame({
    "Day": range(1, 11),
    "Forecast_close": forecast_prices
})

print(forecast_df)

joblib.dump(final_model_fit, "asi_arima_log_returns_model.pkl")

model = joblib.load("asi_arima_log_returns_model.pkl")
