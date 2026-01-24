import pandas as pd
import requests
import numpy as np

# ============================================================
# CONFIG
# ============================================================

FASTAPI_URL = "http://localhost:8000/alerts/run-as-of"
LOOKAHEAD_DAYS = 10

PROD_DATA_PATH = "prod testing.csv"
OUTPUT_PATH = "production_backtest_results.csv"

EPOCHS = 100
THRESH_RANGE = np.arange(0.05, 0.31, 0.03)

# ============================================================
# 1. Load production-period price data
# ============================================================

df = pd.read_csv(PROD_DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

assert "close" in df.columns, "CSV must contain 'close' column"

# ============================================================
# 2. Helper: actual future outcomes
# ============================================================

def future_events(index):
    start_price = df.loc[index, "close"]

    future = df.iloc[index + 1 : index + 1 + LOOKAHEAD_DAYS]
    if len(future) < LOOKAHEAD_DAYS:
        return None

    return {
        "actual_plus_5": int(future["close"].max() >= start_price * 1.05),
        "actual_plus_10": int(future["close"].max() >= start_price * 1.10),
        "actual_plus_20": int(future["close"].max() >= start_price * 1.20),
        "actual_minus_5": int(future["close"].min() <= start_price * 0.95),
        "actual_minus_10": int(future["close"].min() <= start_price * 0.90),
        "actual_minus_20": int(future["close"].min() <= start_price * 0.80),
    }

# ============================================================
# 3. Generate alerts from probabilities
# ============================================================

def generate_alerts(alert_data, t):
    vol = alert_data["vol_regime"]

    return {
        "alert_minus_5": "WATCH"
            if alert_data["prob_minus_5"] >= t["minus_5"] else "NONE",

        "alert_minus_10": "WARNING"
            if alert_data["prob_minus_10"] >= t["minus_10"] and vol else "NONE",

        "alert_minus_20": "CRITICAL"
            if alert_data["prob_minus_20"] >= t["minus_20"] and vol else "NONE",

        "alert_plus_5": "WATCH"
            if alert_data["prob_plus_5"] >= t["plus_5"] else "NONE",

        "alert_plus_10": "WARNING"
            if alert_data["prob_plus_10"] >= t["plus_10"] and vol else "NONE",

        "alert_plus_20": "CRITICAL"
            if alert_data["prob_plus_20"] >= t["plus_20"] and vol else "NONE",
    }

# ============================================================
# 4. Investor-safe scoring function
# ============================================================

def score_backtest(bt):
    score = 0

    for side in ["minus_5", "minus_10", "plus_5", "plus_10"]:
        alert_col = f"alert_{side}"
        actual_col = f"actual_{side}"

        triggered = bt[bt[alert_col] != "NONE"]
        if len(triggered) == 0:
            continue

        score += triggered[actual_col].mean()

    return score

# ============================================================
# 5. Threshold tuning (epoch method)
# ============================================================

best_score = -1
best_thresholds = None

for epoch in range(EPOCHS):
    thresholds = {
        "minus_5": np.random.choice(THRESH_RANGE),
        "minus_10": np.random.choice(THRESH_RANGE),
        "minus_20": np.random.choice(THRESH_RANGE[:4]),
        "plus_5": np.random.choice(THRESH_RANGE),
        "plus_10": np.random.choice(THRESH_RANGE),
        "plus_20": np.random.choice(THRESH_RANGE[:4]),
    }
    print(f"Epoch : {epoch}\tThresholds : {thresholds}")

    temp_rows = []

    for i in range(len(df) - LOOKAHEAD_DAYS):
        test_date = df.loc[i, "date"]

        try:
            r = requests.post(
                FASTAPI_URL,
                json={"as_of_date": str(test_date)},
                timeout=30
            )
            r.raise_for_status()
            alert_data = r.json()
        except:
            continue

        if "prob_minus_5" not in alert_data:
            continue

        actual = future_events(i)
        if actual is None:
            continue

        alerts = generate_alerts(alert_data, thresholds)

        temp_rows.append({**alerts, **actual})

    if not temp_rows:
        continue

    temp_df = pd.DataFrame(temp_rows)
    score = score_backtest(temp_df)

    if score > best_score:
        best_score = score
        best_thresholds = thresholds
        print(f"\n\nEpoch {epoch} → New best score: {best_score:.3f}")
        print(f"\nEpoch {epoch} → Best Thresholds So Far: {best_thresholds}")
# ============================================================
# 6. Final backtest using best thresholds
# ============================================================

print("\nBest thresholds found:")
for k, v in best_thresholds.items():
    print(f"{k}: {v}")

results = []

for i in range(len(df) - LOOKAHEAD_DAYS):
    test_date = df.loc[i, "date"]

    try:
        alert_data = requests.post(
            FASTAPI_URL,
            json={"as_of_date": str(test_date)},
            timeout=30
        ).json()
    except:
        continue

    if "prob_minus_5" not in alert_data:
        continue

    actual = future_events(i)
    if actual is None:
        continue

    alerts = generate_alerts(alert_data, best_thresholds)

    results.append({
        "date": test_date,
        "close": df.loc[i, "close"],

        # probabilities
        "prob_minus_5": alert_data["prob_minus_5"],
        "prob_minus_10": alert_data["prob_minus_10"],
        "prob_minus_20": alert_data["prob_minus_20"],
        "prob_plus_5": alert_data["prob_plus_5"],
        "prob_plus_10": alert_data["prob_plus_10"],
        "prob_plus_20": alert_data["prob_plus_20"],

        # alerts
        **alerts,
        "vol_regime": alert_data["vol_regime"],

        # actuals
        **actual
    })

backtest_df = pd.DataFrame(results)
backtest_df.to_csv(OUTPUT_PATH, index=False)

# ============================================================
# 7. Investor summary
# ============================================================

print("\n===== INVESTOR SUMMARY =====")

for side, label in [
    ("minus_5", "−5% Drop"),
    ("minus_10", "−10% Drop"),
    ("plus_5", "+5% Rise"),
    ("plus_10", "+10% Rise"),
]:
    alerts = backtest_df[backtest_df[f"alert_{side}"] != "NONE"]
    hit_rate = alerts[f"actual_{side}"].mean() if len(alerts) > 0 else 0

    print(f"When alerting {label}, correctness was {hit_rate:.2%}")

print(f"\nBacktest saved to: {OUTPUT_PATH}")
