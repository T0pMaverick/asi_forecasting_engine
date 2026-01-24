import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "production_backtest_results.csv"

# ============================================================
# 1. Load results
# ============================================================

df = pd.read_csv(DATA_PATH)

# Ensure no missing values in alert columns
alert_cols = [
    "alert_minus_5", "alert_minus_10", "alert_minus_20",
    "alert_plus_5", "alert_plus_10", "alert_plus_20"
]

df[alert_cols] = df[alert_cols].fillna("NONE")

# ============================================================
# 2. ALERTS vs ACTUAL EVENTS (BAR CHART)
# ============================================================

events = {
    "-5% Drop": ("alert_minus_5", "actual_minus_5", "WATCH"),
    "-10% Drop": ("alert_minus_10", "actual_minus_10", "WARNING"),
    "+5% Rise": ("alert_plus_5", "actual_plus_5", "WATCH"),
    "+10% Rise": ("alert_plus_10", "actual_plus_10", "WARNING"),
}

alert_counts = []
actual_counts = []

labels = []

for name, (alert_col, actual_col, level) in events.items():
    alert_counts.append((df[alert_col] == level).sum())
    actual_counts.append(df[actual_col].sum())
    labels.append(name)

x = range(len(labels))

plt.figure(figsize=(10, 5))
plt.bar(x, alert_counts, width=0.4, label="Model Alerts")
plt.bar([i + 0.4 for i in x], actual_counts, width=0.4, label="Actual Events")

plt.xticks([i + 0.2 for i in x], labels)
plt.ylabel("Count")
plt.title("Model Alerts vs Actual Market Events")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 3. ALERT ACCURACY (HIT RATE)
# ============================================================

hit_rates = []

for name, (alert_col, actual_col, level) in events.items():
    alerts = df[df[alert_col] == level]
    if len(alerts) == 0:
        hit_rates.append(0)
    else:
        hit_rates.append(alerts[actual_col].mean())

plt.figure(figsize=(8, 5))
plt.bar(labels, hit_rates)
plt.ylim(0, 1)
plt.ylabel("Hit Rate")
plt.title("How Often Alerts Were Correct")
plt.axhline(0.5, linestyle="--", color="gray", alpha=0.5)
plt.tight_layout()
plt.show()

# ============================================================
# 4. PROBABILITY VS REALITY (TRUST CHART)
# ============================================================

prob_pairs = [
    ("prob_minus_5", "actual_minus_5", "-5% Drop"),
    ("prob_minus_10", "actual_minus_10", "-10% Drop"),
    ("prob_plus_5", "actual_plus_5", "+5% Rise"),
    ("prob_plus_10", "actual_plus_10", "+10% Rise"),
]

labels = []
prob_event = []
prob_no_event = []

for prob_col, actual_col, label in prob_pairs:
    labels.append(label)
    prob_event.append(df[df[actual_col] == 1][prob_col].mean())
    prob_no_event.append(df[df[actual_col] == 0][prob_col].mean())

x = range(len(labels))

plt.figure(figsize=(10, 5))
plt.bar(x, prob_event, width=0.4, label="When Event Happened")
plt.bar([i + 0.4 for i in x], prob_no_event, width=0.4, label="When Event Did Not Happen")

plt.xticks([i + 0.2 for i in x], labels)
plt.ylabel("Average Predicted Probability")
plt.title("Model Confidence vs Reality")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 5. SIMPLE INVESTOR SUMMARY
# ============================================================

print("\n===== PLAIN ENGLISH INTERPRETATION =====")
for i, label in enumerate(labels):
    print(
        f"When the model warned about {label}, "
        f"it was correct {hit_rates[i]*100:.1f}% of the time."
    )
