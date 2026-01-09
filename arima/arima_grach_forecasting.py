# ============================================================
# ARIMA + GARCH Forecasting for Stock Index Closing Prices
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. Load dataset
# ============================================================

DATA_PATH = "ASI_close_prices_last_15_years_2025-06-05.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").set_index("date")
df = df[["close"]].dropna()

# ============================================================
# 2. Feature engineering: log prices & log returns
# ============================================================

df["log_close"] = np.log(df["close"])
df["log_return"] = df["log_close"].diff()
df = df.dropna()

# ============================================================
# 3. Train / validation split (time-aware)
# ============================================================

train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
val   = df.iloc[train_size:]

# ============================================================
# 4. Train ARIMA(1,0,0) on log-returns (mean model)
# ============================================================

arima_model = ARIMA(train["log_return"], order=(1, 0, 0))
arima_fit = arima_model.fit()

print("\n===== ARIMA MODEL SUMMARY =====")
print(arima_fit.summary())

# ============================================================
# 5. Walk-forward validation (overfitting check)
# ============================================================

history = list(train["log_return"])
predicted_returns = []

for t in range(len(val)):
    model = ARIMA(history, order=(1, 0, 0))
    model_fit = model.fit()
    
    forecast = model_fit.forecast()[0]
    predicted_returns.append(forecast)
    
    history.append(val["log_return"].iloc[t])

# ============================================================
# 6. Convert predicted returns to prices
# ============================================================

last_log_price = train["log_close"].iloc[-1]

log_price_preds = []
current_log_price = last_log_price

for r in predicted_returns:
    current_log_price += r
    log_price_preds.append(current_log_price)

price_preds = np.exp(log_price_preds)

# ============================================================
# 7. MAPE evaluation
# ============================================================

actual_prices = val["close"].values
val_mape = mean_absolute_percentage_error(actual_prices, price_preds)

print(f"\nValidation MAPE: {val_mape * 100:.2f}%")

# ============================================================
# 8. Residual diagnostics (mean model)
# ============================================================

residuals = arima_fit.resid

plt.figure(figsize=(12, 4))
plt.plot(residuals)
plt.title("ARIMA Residuals (Mean Model)")
plt.show()

# ============================================================
# 9. Fit GARCH(1,1) on ARIMA residuals (volatility model)
# ============================================================

garch = arch_model(
    residuals * 100,   # scale improves numerical stability
    vol="Garch",
    p=1,
    q=1,
    mean="Zero"
)

garch_fit = garch.fit(disp="off")

print("\n===== GARCH MODEL SUMMARY =====")
print(garch_fit.summary())

# ============================================================
# 10. Train final models on FULL dataset
# ============================================================

final_arima = ARIMA(df["log_return"], order=(1, 0, 0))
final_arima_fit = final_arima.fit()

final_residuals = final_arima_fit.resid

final_garch = arch_model(
    final_residuals * 100,
    vol="Garch",
    p=1,
    q=1,
    mean="Zero"
)

final_garch_fit = final_garch.fit(disp="off")

# ============================================================
# 11. Forecast next 10 trading days
# ============================================================

N_FORECAST = 10

# Mean (returns) forecast
forecast_returns = final_arima_fit.forecast(steps=N_FORECAST)

# Volatility forecast
garch_forecast = final_garch_fit.forecast(horizon=N_FORECAST)
forecast_vol = np.sqrt(garch_forecast.variance.values[-1]) / 100

# Reconstruct prices
last_log_price = df["log_close"].iloc[-1]
forecast_log_prices = last_log_price + np.cumsum(forecast_returns)
forecast_prices = np.exp(forecast_log_prices)

forecast_df = pd.DataFrame({
    "Day": range(1, N_FORECAST + 1),
    "Forecast_close": forecast_prices,
    "Forecast_Volatility": forecast_vol
})

print("\n===== 10-DAY FORECAST =====")
print(forecast_df)

# ============================================================
# 12. Save trained models
# ============================================================

joblib.dump(final_arima_fit, "./arima/arima_mean_model.pkl")
joblib.dump(final_garch_fit, "./arima/garch_volatility_model.pkl")

print("\nModels saved:")
print(" - ./arima/arima_mean_model.pkl")
print(" - ./arima/garch_volatility_model.pkl")

# ============================================================
# END OF SCRIPT
# ============================================================
