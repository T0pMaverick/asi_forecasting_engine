import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "production_backtest_results.csv"

# ============================================================
# 1. Load backtest results
# ============================================================

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

print(f"Loaded {len(df)} production backtest records")

# ============================================================
# 2. Helper: evaluate alert quality
# ============================================================

def evaluate_alert(df, alert_col, actual_col, alert_name):
    """
    Simple confusion-matrix style evaluation
    """

    total_days = len(df)

    alerts_issued = df[df[alert_col] != "NONE"]
    no_alerts = df[df[alert_col] == "NONE"]

    true_positive = ((df[alert_col] != "NONE") & (df[actual_col] == 1)).sum()
    false_positive = ((df[alert_col] != "NONE") & (df[actual_col] == 0)).sum()
    false_negative = ((df[alert_col] == "NONE") & (df[actual_col] == 1)).sum()
    true_negative = ((df[alert_col] == "NONE") & (df[actual_col] == 0)).sum()

    hit_rate = true_positive / max(len(alerts_issued), 1)
    miss_rate = false_negative / max((df[actual_col] == 1).sum(), 1)

    return {
        "Alert Type": alert_name,
        "Total Days": total_days,
        "Alerts Issued": len(alerts_issued),
        "Actual Events": int((df[actual_col] == 1).sum()),
        "Correct Alerts (Hits)": true_positive,
        "False Alarms": false_positive,
        "Missed Events": false_negative,
        "Hit Rate (When Alerted)": round(hit_rate, 2),
        "Miss Rate (When Event Happened)": round(miss_rate, 2),
    }

# ============================================================
# 3. Evaluate key alert types
# ============================================================

results = []

results.append(evaluate_alert(df, "alert_minus_5", "actual_minus_5", "−5% Drop"))
results.append(evaluate_alert(df, "alert_minus_10", "actual_minus_10", "−10% Drop"))
results.append(evaluate_alert(df, "alert_plus_5", "actual_plus_5", "+5% Rise"))
results.append(evaluate_alert(df, "alert_plus_10", "actual_plus_10", "+10% Rise"))

summary_df = pd.DataFrame(results)

print("\n===== INVESTOR SUMMARY TABLE =====")
print(summary_df)

# ============================================================
# 4. Simple bar chart (hit rate)
# ============================================================

plt.figure()
plt.bar(summary_df["Alert Type"], summary_df["Hit Rate (When Alerted)"])
plt.title("How Often Alerts Were Correct")
plt.ylabel("Hit Rate")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.show()

# ============================================================
# 5. Simple explanation output
# ============================================================

print("\n===== PLAIN ENGLISH INTERPRETATION =====")

for _, row in summary_df.iterrows():
    print(
        f"When the model warned about {row['Alert Type']}, "
        f"it was correct {int(row['Hit Rate (When Alerted)']*100)}% of the time. "
        f"It missed {int(row['Miss Rate (When Event Happened)']*100)}% of actual events."
    )
