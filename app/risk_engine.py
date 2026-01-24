# app/risk_engine.py

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from scipy.stats import t

TRADING_DAYS = 252
ROLLING_YEARS = 4
H = 10
N_SIM = 20000
VOL_PCTL = 0.70

def run_risk_engine(price_df: pd.DataFrame):
    df = price_df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    df["log_close"] = np.log(df["close"])
    df["log_return"] = df["log_close"].diff()
    df = df.dropna()

    # ARIMA mean (residual extraction only)
    arima = ARIMA(df["log_return"], order=(1, 0, 0))
    arima_fit = arima.fit()
    residuals = arima_fit.resid

    # Rolling EGARCH
    window = ROLLING_YEARS * TRADING_DAYS
    rolling_resid = residuals.iloc[-window:] * 100

    egarch = arch_model(
        rolling_resid,
        vol="EGARCH",
        p=1, q=1,
        mean="Zero",
        dist="t"
    )
    egarch_fit = egarch.fit(disp="off")

    cond_vol = egarch_fit.conditional_volatility
    current_vol = cond_vol.iloc[-1]
    high_vol = bool(current_vol >= cond_vol.quantile(VOL_PCTL))

    params = egarch_fit.params
    nu = params["nu"]

    # Monte Carlo
    sims = []
    for _ in range(N_SIM):
        vol = current_vol
        cum_ret = 0.0
        for _ in range(H):
            shock = t.rvs(df=nu)
            vol = np.exp(
                params["omega"]
                + params["alpha[1]"] * (abs(shock) - np.sqrt(2 / np.pi))
                + params["beta[1]"] * np.log(vol)
            )
            cum_ret += shock * vol / 100
        sims.append(cum_ret)

    sims = np.array(sims)

    def prob(th):
        return float(np.mean(sims <= th)) if th < 0 else float(np.mean(sims >= th))

    return {
        "vol_regime": high_vol,
        "prob_minus_5": prob(np.log(0.95)),
        "prob_minus_10": prob(np.log(0.90)),
        "prob_minus_20": prob(np.log(0.80)),
        "prob_plus_5": prob(np.log(1.05)),
        "prob_plus_10": prob(np.log(1.10)),
        "prob_plus_20": prob(np.log(1.20)),
    }
