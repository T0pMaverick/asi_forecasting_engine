# ASI Market Turn Detection - Comprehensive Analysis & Model Training Report

**Date:** December 5, 2025  
**Analysis Period:** February 23, 2012 to December 5, 2025 (3,264 trading days)

---

## Executive Summary

This report presents a comprehensive analysis of market turn detection approaches for the ASI (All Share Index), testing multiple threshold and timeframe combinations to identify the optimal strategy for predicting market downturns.

### Key Findings

1. **Best Combination:** -5% drawdown in 10 days
   - Score: 81.7/100
   - Positive rate: 9.83% (321 cases out of 3,265)
   - Confirmation signals: 54% trend break, 53% negative momentum, 59% RSI below 50

2. **Best Model:** LightGBM
   - AUC-ROC: 0.9818
   - Precision: 0.8824
   - Recall: 0.7500
   - F1 Score: 0.8108

3. **Recommended Multi-Tier System:**
   - ⚠️ Early Warning: -3% in 7 days
   - 🚨 Confirmed Turn: -5% in 10 days
   - 🔴 Major Downturn: -7% in 10 days

---

## Analysis Methodology

### Combinations Tested

- **Thresholds:** -3%, -5%, -7%
- **Timeframes:** 5, 7, 10, 15 days
- **Total Combinations:** 12

### Confirmation Signals Analyzed

1. **Trend Break:** Price below SMA50
2. **Negative Momentum:** 20-day return < 0
3. **RSI Below 50:** Relative Strength Index declining
4. **MACD Bearish:** MACD below signal line

---

## Results: Top 5 Combinations

| Rank | Threshold | Timeframe | Score | Positive Rate | Trend Break | Momentum | RSI < 50 | MACD Bearish |
|------|-----------|-----------|-------|---------------|-------------|----------|----------|--------------|
| 1 | -5% | 10 days | 81.7 | 9.83% | 54.2% | 53.3% | 58.6% | 57.3% |
| 2 | -3% | 7 days | 80.9 | 14.98% | 50.1% | 51.7% | 53.8% | 56.4% |
| 3 | -5% | 7 days | 79.2 | 6.10% | 57.3% | 55.8% | 63.8% | 60.3% |
| 4 | -3% | 10 days | 77.7 | 22.24% | 47.8% | 49.9% | 51.1% | 53.6% |
| 5 | -5% | 5 days | 75.1 | 3.68% | 57.5% | 59.2% | 66.7% | 65.0% |

---

## Model Training Results

### Dataset Split

- **Training Set:** 1,839 samples (180 positive, 1,659 negative)
- **Validation Set:** 613 samples (60 positive, 553 negative)
- **Test Set:** 613 samples (60 positive, 553 negative)
- **Features:** 26 technical indicators

### Model Performance Comparison

| Model | AUC-ROC | Precision | Recall | F1 Score | Accuracy |
|-------|---------|-----------|--------|----------|----------|
| **LightGBM** | **0.9818** | **0.8824** | **0.7500** | **0.8108** | **0.9657** |
| Gradient Boosting | 0.9673 | 0.9500 | 0.6333 | 0.7600 | 0.9608 |
| Random Forest | 0.9589 | 0.7500 | 0.7000 | 0.7241 | 0.9478 |
| Logistic Regression | 0.8662 | 0.3672 | 0.7833 | 0.5000 | 0.8467 |

### Best Model: LightGBM

**Test Set Performance:**
- **True Positives:** 45 (correctly predicted downturns)
- **False Positives:** 6 (false alarms)
- **True Negatives:** 547 (correctly predicted no downturn)
- **False Negatives:** 15 (missed downturns)

**Key Strengths:**
- Excellent AUC score (0.9818) - very good at distinguishing between classes
- High precision (0.8824) - low false positive rate
- Good recall (0.7500) - catches 75% of actual downturns
- High accuracy (0.9657) - overall correct predictions

---

## Recommendations

### 1. Primary Detection Strategy

**Use: -5% drawdown in 10 days**

**Rationale:**
- Balanced positive rate (9.83%) - not too frequent, not too rare
- Good confirmation signals (54-59% of cases show trend/momentum/RSI signals)
- 10-day timeframe provides actionable lead time
- Best overall score in comprehensive analysis

### 2. Multi-Tier Warning System

Implement a three-tier system for different risk levels:

1. **Early Warning (-3% in 7 days)**
   - Purpose: Early signal for cautious investors
   - Action: Reduce exposure, tighten stops
   - Frequency: ~15% of trading days

2. **Confirmed Turn (-5% in 10 days)** ⭐ **PRIMARY**
   - Purpose: Main signal for defensive positioning
   - Action: Reduce positions, increase cash, hedge
   - Frequency: ~10% of trading days

3. **Major Downturn (-7% in 10 days)**
   - Purpose: Signal for major corrections/bear markets
   - Action: Significant position reduction, defensive assets
   - Frequency: ~5% of trading days

### 3. Confirmation Signals

When primary signal triggers, check for confirmations:

- ✅ **Price below SMA50** (54% of cases)
- ✅ **Negative 20-day momentum** (53% of cases)
- ✅ **RSI below 50** (59% of cases)
- ✅ **MACD bearish** (57% of cases)

**Action Rule:** If 2+ confirmations present, increase confidence in signal.

### 4. Model Usage

**Best Model:** LightGBM (saved in `models/lightgbm.pkl`)

**Usage:**
- Monitor daily: Run model on latest ASI data
- Probability threshold: Use 0.5 (or adjust based on risk tolerance)
- Combine with confirmation signals for higher confidence

---

## Why -5% in 10 Days is Better Than -5% in 5 Days

### Original Approach (-5% in 5 days)
- ❌ Too aggressive (only 3.68% positive rate)
- ❌ Less actionable lead time
- ❌ More false positives relative to signal frequency
- ❌ Higher confirmation rates but fewer overall signals

### Recommended Approach (-5% in 10 days)
- ✅ More balanced (9.83% positive rate)
- ✅ Better lead time for action
- ✅ Good balance of precision and recall
- ✅ Strong confirmation signals (54-59%)
- ✅ More training data (321 vs 120 cases)

---

## Files Generated

1. **Models:**
   - `models/lightgbm.pkl` - Best model (LightGBM)
   - `models/random_forest.pkl` - Random Forest model
   - `models/gradient_boosting.pkl` - Gradient Boosting model
   - `models/logistic_regression.pkl` - Logistic Regression model
   - `models/model_summary.json` - Model comparison summary

2. **Analysis Results:**
   - `comprehensive_analysis_results.json` - Full analysis results
   - `combination_rankings.csv` - All combinations ranked
   - `combination_comparison.png` - Visualization of combinations
   - `model_training_results.png` - Model performance plots

3. **Scripts:**
   - `comprehensive_downturn_analysis.py` - Analysis script
   - `train_downturn_predictor.py` - Training script
   - `asi_predictor.py` - Data fetching script

---

## Next Steps

1. **Deploy Model:** Use LightGBM model for daily predictions
2. **Monitor Performance:** Track predictions vs actual outcomes
3. **Refine Thresholds:** Adjust probability threshold based on results
4. **Add Features:** Consider adding volume, sector data, or external factors
5. **Backtest Strategy:** Test trading strategy based on predictions

---

## Conclusion

The comprehensive analysis shows that **-5% drawdown in 10 days** is the optimal approach for detecting market turns in the ASI, providing:
- Good balance between signal frequency and reliability
- Strong confirmation signals
- Actionable lead time
- Excellent model performance (AUC: 0.9818)

The trained LightGBM model is ready for deployment and can provide 10-day advance warning of potential -5% market downturns with high accuracy.

---

**Report Generated:** December 5, 2025  
**Analysis Period:** 2012-02-23 to 2025-12-05  
**Total Trading Days Analyzed:** 3,264

