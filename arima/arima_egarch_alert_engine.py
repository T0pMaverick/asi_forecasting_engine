# ============================================================
# ARIMA + EGARCH (Student-t) Risk Alert Engine for ASI
# (Directional Risk + Rolling Recalibration + Regime Gate)
# ============================================================

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from scipy.stats import t
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "ASI_close_prices_last_15_years_2025-06-05.csv"

FORECAST_HORIZON = 10
N_SIM = 20000

ROLLING_YEARS = 4          # rolling EGARCH window (3–5 recommended)
TRADING_DAYS = 252
ROLLING_WINDOW = ROLLING_YEARS * TRADING_DAYS

VOL_REGIME_PERCENTILE = 0.70  # regime gate threshold

# ============================================================
# 1. Load data
# ============================================================

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").set_index("date")
df = df[["close"]].dropna()

# ============================================================
# 2. Log returns
# ============================================================

df["log_close"] = np.log(df["close"])
df["log_return"] = df["log_close"].diff()
df = df.dropna()

# ============================================================
# 3. ARIMA(1,0,0) — FIXED MEAN MODEL
# ============================================================

arima = ARIMA(df["log_return"], order=(1, 0, 0))
arima_fit = arima.fit()

residuals = arima_fit.resid

print("\n===== ARIMA MEAN MODEL =====")
print(arima_fit.summary())

# ============================================================
# 4. ROLLING EGARCH (Student-t) — VOLATILITY MODEL
# ============================================================

rolling_data = residuals.iloc[-ROLLING_WINDOW:] * 100

egarch = arch_model(
    rolling_data,
    vol="EGARCH",
    p=1,
    q=1,
    mean="Zero",
    dist="t"
)

egarch_fit = egarch.fit(disp="off")

print("\n===== ROLLING EGARCH (Student-t) =====")
print(egarch_fit.summary())

params = egarch_fit.params
nu = params["nu"]

# ============================================================
# 5. VOLATILITY REGIME GATE
# ============================================================

cond_vol = egarch_fit.conditional_volatility
current_vol = cond_vol.iloc[-1]
vol_threshold = cond_vol.quantile(VOL_REGIME_PERCENTILE)

HIGH_VOL_REGIME = current_vol >= vol_threshold

print("\n===== VOLATILITY REGIME =====")
print(f"Current Volatility : {current_vol:.4f}")
print(f"{int(VOL_REGIME_PERCENTILE*100)}th Percentile : {vol_threshold:.4f}")
print("High Vol Regime    :", HIGH_VOL_REGIME)

# ============================================================
# 6. MONTE CARLO SIMULATION (10-DAY RETURNS)
# ============================================================

simulated_returns = []

last_vol = current_vol

for _ in range(N_SIM):
    vol_t = last_vol
    path_return = 0.0

    for _ in range(FORECAST_HORIZON):
        shock = t.rvs(df=nu)

        vol_t = np.exp(
            params["omega"]
            + params["alpha[1]"] * (abs(shock) - np.sqrt(2 / np.pi))
            + params["beta[1]"] * np.log(vol_t)
        )

        ret = shock * vol_t / 100
        path_return += ret

    simulated_returns.append(path_return)

simulated_returns = np.array(simulated_returns)

# ============================================================
# 7. THRESHOLD PROBABILITIES
# ============================================================

thresholds = {
    "-20%": np.log(0.80),
    "-10%": np.log(0.90),
    "-5%":  np.log(0.95),
    "+5%":  np.log(1.05),
    "+10%": np.log(1.10),
    "+20%": np.log(1.20),
}

alert_probs = {}

for label, thresh in thresholds.items():
    if thresh < 0:
        alert_probs[label] = np.mean(simulated_returns <= thresh)
    else:
        alert_probs[label] = np.mean(simulated_returns >= thresh)

alert_df = pd.DataFrame.from_dict(
    alert_probs, orient="index", columns=["Probability"]
)

# ============================================================
# 8. DIRECTIONAL ALERT LOGIC (LOSS-AWARE)
# ============================================================

def alert_level(label, p, high_vol):
    # Downside risk (more sensitive)
    if label in ["-5%", "-10%", "-20%"]:
        if label == "-5%" and p >= 0.10:
            return "WATCH 🟡"
        if label == "-10%" and p >= 0.10 and high_vol:
            return "WARNING 🟠"
        if label == "-20%" and p >= 0.05 and high_vol:
            return "CRITICAL 🔴"
        return "NO ALERT 🟢"

    # Upside risk (less sensitive)
    else:
        if p >= 0.20:
            return "WATCH 🟡"
        return "NO ALERT 🟢"


alert_df["Alert"] = [
    alert_level(idx, row["Probability"], HIGH_VOL_REGIME)
    for idx, row in alert_df.iterrows()
]

print("\n===== 10-DAY RISK ALERTS (BUSINESS-READY) =====")
print(alert_df)

# ============================================================
# 9. BACKTEST SUMMARY (SANITY CHECK)
# ============================================================

LOOKAHEAD = 10
TEST_WINDOW = 500

events = []

for i in range(len(df) - LOOKAHEAD - TEST_WINDOW, len(df) - LOOKAHEAD):
    start_price = df["close"].iloc[i]
    future_max = df["close"].iloc[i+1:i+LOOKAHEAD+1].max()
    future_min = df["close"].iloc[i+1:i+LOOKAHEAD+1].min()

    max_return = np.log(future_max / start_price)
    min_return = np.log(future_min / start_price)

    events.append({
        "+10%": int(max_return >= np.log(1.10)),
        "-10%": int(min_return <= np.log(0.90)),
    })

events_df = pd.DataFrame(events)

print("\n===== BACKTEST SUMMARY (LAST 500 WINDOWS) =====")
print(events_df.mean().rename("Event Frequency"))

# ============================================================
# END
# ============================================================
