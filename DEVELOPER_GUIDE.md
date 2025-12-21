# ASI Market Downturn Prediction System - Developer Guide

**Version:** 1.0  
**Last Updated:** December 5, 2025  
**Target:** Predict -5% ASI drawdowns 10 days in advance

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [Data Pipeline](#data-pipeline)
6. [Model Training](#model-training)
7. [Prediction Workflow](#prediction-workflow)
8. [Deployment Guide](#deployment-guide)
9. [API Usage](#api-usage)
10. [Maintenance & Updates](#maintenance--updates)
11. [Troubleshooting](#troubleshooting)

---

## System Overview

### Purpose
The ASI Market Downturn Prediction System predicts when the All Share Index (ASI) will experience a -5% drawdown within the next 10 days, providing early warning signals for risk management.

### Key Features
- **Early Warning:** 10-day advance prediction of -5% drawdowns
- **High Accuracy:** LightGBM model with 98.18% AUC-ROC
- **Multi-Model Support:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Comprehensive Analysis:** Depth and duration analysis of drawdowns
- **Production Ready:** Trained models ready for deployment

### Model Performance
- **Best Model:** LightGBM
- **AUC-ROC:** 0.9818
- **Precision:** 0.8824
- **Recall:** 0.7500
- **F1 Score:** 0.8108
- **Accuracy:** 96.57%

---

## Architecture

### System Components

```
┌─────────────────┐
│  Data Fetcher   │ → Fetches ASI close prices from TradingView
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Engine  │ → Computes 26 technical indicators
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Trainer  │ → Trains multiple ML models
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Predictor     │ → Generates predictions for new data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Results Export │ → CSV/JSON output for analysis
└─────────────────┘
```

### Data Flow

1. **Data Collection:** ASI close prices from TradingView API
2. **Feature Engineering:** Compute 26 technical indicators
3. **Label Creation:** Identify -5% drawdowns in 10-day windows
4. **Model Training:** Train and validate multiple models
5. **Prediction:** Generate predictions for new dates
6. **Output:** Save results to CSV/JSON

---

## Project Structure

```
ASI_prediction/
├── asi_predictor.py                    # Data fetcher (ASI close prices)
├── comprehensive_downturn_analysis.py   # Threshold/timeframe analysis
├── train_downturn_predictor.py         # Model training script
├── analyze_drawdown_depth_duration.py  # Depth/duration analysis
├── predict_november_2025.py           # Example prediction script
│
├── models/                              # Trained models
│   ├── lightgbm.pkl                    # Best model (LightGBM)
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── logistic_regression.pkl
│   └── model_summary.json              # Model performance summary
│
├── data/                                # Data files
│   └── ASI_close_prices_last_15_years_2025-12-05.csv
│
├── outputs/                             # Generated outputs
│   ├── test_results.csv                # Test set predictions
│   ├── november_2025_predictions.csv   # November predictions
│   ├── drawdown_details.csv            # Historical drawdowns
│   └── combination_rankings.csv       # Threshold analysis
│
└── docs/                                # Documentation
    ├── SUMMARY_REPORT.md
    ├── DRAWDOWN_DEPTH_DURATION_REPORT.md
    └── DEVELOPER_GUIDE.md (this file)
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- pip package manager
- TradingView account (optional, for data fetching)

### Step 1: Clone/Setup Project

```bash
cd ASI_prediction
```

### Step 2: Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost lightgbm  # Optional but recommended
pip install tvDatafeed  # For TradingView data fetching
```

### Step 3: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, xgboost, lightgbm; print('All packages installed')"
```

### Step 4: Environment Setup

Create a `.env` file (optional, for database connections):

```env
NEON_DB_URL=your_database_url_here
```

---

## Data Pipeline

### 1. Fetch ASI Data

**Script:** `asi_predictor.py`

**Usage:**
```bash
python asi_predictor.py
```

**What it does:**
- Fetches last 15 years of ASI close prices from TradingView
- Saves to CSV: `ASI_close_prices_last_15_years_YYYY-MM-DD.csv`
- Handles multiple symbol formats automatically

**Output:**
```
ASI_close_prices_last_15_years_2025-12-05.csv
Columns: date, close
```

### 2. Update Data Regularly

For production, set up a cron job or scheduled task:

```bash
# Daily update (runs at 6 PM)
0 18 * * * cd /path/to/ASI_prediction && python asi_predictor.py
```

---

## Model Training

### Step 1: Run Comprehensive Analysis (Optional)

Analyze different threshold/timeframe combinations:

```bash
python comprehensive_downturn_analysis.py
```

**Output:**
- `combination_rankings.csv` - All combinations ranked
- `comprehensive_analysis_results.json` - Full analysis
- `combination_comparison.png` - Visualizations

### Step 2: Train Models

**Script:** `train_downturn_predictor.py`

**Usage:**
```bash
python train_downturn_predictor.py
```

**What it does:**
1. Loads ASI data
2. Computes 26 technical indicators
3. Creates labels (-5% drawdown in 10 days)
4. Splits data: 60% train, 20% validation, 20% test
5. Trains multiple models:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost (if available)
   - LightGBM (if available)
6. Evaluates and compares models
7. Saves best model to `models/lightgbm.pkl`
8. Exports test results to `test_results.csv`

**Output Files:**
- `models/lightgbm.pkl` - Best trained model
- `models/model_summary.json` - Performance summary
- `test_results.csv` - Test set predictions
- `model_training_results.png` - Visualizations

### Step 3: Verify Training

Check model performance:

```bash
python -c "import json; print(json.load(open('models/model_summary.json')))"
```

---

## Prediction Workflow

### Method 1: Predict for Specific Dates

**Script:** `predict_november_2025.py` (example)

**Usage:**
```python
from train_downturn_predictor import DownturnPredictorTrainer
import pickle
import pandas as pd

# Load trained model
with open('models/lightgbm.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']

# Load and prepare data
trainer = DownturnPredictorTrainer('ASI_close_prices_last_15_years_2025-12-05.csv')
trainer.compute_features()

# Filter to dates you want to predict
target_dates = trainer.df[trainer.df['date'] >= '2025-11-01']
X = target_dates[feature_names]

# Make predictions
if scaler:
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
else:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

# Create results
results = pd.DataFrame({
    'date': target_dates['date'],
    'close': target_dates['close'],
    'predicted': predictions,
    'probability': probabilities
})
```

### Method 2: Daily Production Predictions

Create `predict_daily.py`:

```python
#!/usr/bin/env python3
"""Daily prediction script for production"""

import pandas as pd
import pickle
from datetime import datetime, timedelta
from train_downturn_predictor import DownturnPredictorTrainer

def predict_today():
    """Generate prediction for today"""
    
    # Load latest data
    trainer = DownturnPredictorTrainer('ASI_close_prices_last_15_years_2025-12-05.csv')
    trainer.compute_features()
    
    # Get today's data
    today = datetime.now().date()
    today_data = trainer.df[trainer.df['date'].dt.date == today]
    
    if len(today_data) == 0:
        print(f"No data available for {today}")
        return None
    
    # Load model
    with open('models/lightgbm.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
    
    # Prepare features
    X = today_data[feature_names]
    
    if X.isna().any().any():
        print("Missing features for today")
        return None
    
    # Predict
    if scaler:
        X_scaled = scaler.transform(X)
        probability = model.predict_proba(X_scaled)[0, 1]
        prediction = model.predict(X_scaled)[0]
    else:
        probability = model.predict_proba(X)[0, 1]
        prediction = model.predict(X)[0]
    
    return {
        'date': today,
        'close': today_data['close'].iloc[0],
        'prediction': bool(prediction),
        'probability': float(probability),
        'signal': 'HIGH' if probability >= 0.7 else 'MEDIUM' if probability >= 0.5 else 'LOW'
    }

if __name__ == "__main__":
    result = predict_today()
    if result:
        print(f"Date: {result['date']}")
        print(f"ASI Close: {result['close']:.2f}")
        print(f"Prediction: {'DRAWDOWN EXPECTED' if result['prediction'] else 'NO DRAWDOWN'}")
        print(f"Probability: {result['probability']:.2%}")
        print(f"Signal Strength: {result['signal']}")
```

---

## Deployment Guide

### Option 1: Standalone Script Deployment

**Steps:**

1. **Prepare deployment package:**
```bash
# Create deployment directory
mkdir asi_prediction_deploy
cp -r models/ asi_prediction_deploy/
cp train_downturn_predictor.py asi_prediction_deploy/
cp predict_daily.py asi_prediction_deploy/
cp requirements.txt asi_prediction_deploy/
```

2. **Create requirements.txt:**
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
lightgbm>=3.3.0
xgboost>=1.7.0
tvDatafeed>=1.5.0
```

3. **Deploy:**
```bash
# On production server
cd asi_prediction_deploy
pip install -r requirements.txt
python predict_daily.py
```

### Option 2: API Deployment (Flask/FastAPI)

**Create `api_server.py`:**

```python
from flask import Flask, jsonify, request
import pickle
import pandas as pd
from train_downturn_predictor import DownturnPredictorTrainer

app = Flask(__name__)

# Load model at startup
with open('models/lightgbm.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']

@app.route('/predict', methods=['POST'])
def predict():
    """Predict for given date"""
    data = request.json
    date = pd.to_datetime(data['date'])
    
    # Load and compute features
    trainer = DownturnPredictorTrainer('ASI_close_prices_last_15_years_2025-12-05.csv')
    trainer.compute_features()
    
    # Get data for date
    date_data = trainer.df[trainer.df['date'] == date]
    if len(date_data) == 0:
        return jsonify({'error': 'Date not found'}), 404
    
    # Prepare features
    X = date_data[feature_names]
    
    # Predict
    if scaler:
        X_scaled = scaler.transform(X)
        probability = model.predict_proba(X_scaled)[0, 1]
        prediction = model.predict(X_scaled)[0]
    else:
        probability = model.predict_proba(X)[0, 1]
        prediction = model.predict(X)[0]
    
    return jsonify({
        'date': str(date),
        'prediction': bool(prediction),
        'probability': float(probability),
        'signal': 'HIGH' if probability >= 0.7 else 'MEDIUM' if probability >= 0.5 else 'LOW'
    })

@app.route('/predict/latest', methods=['GET'])
def predict_latest():
    """Predict for latest available date"""
    trainer = DownturnPredictorTrainer('ASI_close_prices_last_15_years_2025-12-05.csv')
    trainer.compute_features()
    
    latest = trainer.df.iloc[-1]
    X = latest[feature_names].values.reshape(1, -1)
    
    if scaler:
        X_scaled = scaler.transform(X)
        probability = model.predict_proba(X_scaled)[0, 1]
        prediction = model.predict(X_scaled)[0]
    else:
        probability = model.predict_proba(X)[0, 1]
        prediction = model.predict(X)[0]
    
    return jsonify({
        'date': str(latest['date']),
        'close': float(latest['close']),
        'prediction': bool(prediction),
        'probability': float(probability),
        'signal': 'HIGH' if probability >= 0.7 else 'MEDIUM' if probability >= 0.5 else 'LOW'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Run API:**
```bash
python api_server.py
```

**Test API:**
```bash
# Predict latest
curl http://localhost:5000/predict/latest

# Predict specific date
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2025-11-17"}'
```

### Option 3: Docker Deployment

**Create `Dockerfile`:**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "api_server.py"]
```

**Build and run:**
```bash
docker build -t asi-predictor .
docker run -p 5000:5000 asi-predictor
```

---

## API Usage

### REST API Endpoints

#### 1. Predict Latest Date
```http
GET /predict/latest
```

**Response:**
```json
{
  "date": "2025-12-05",
  "close": 21497.08,
  "prediction": true,
  "probability": 0.9412,
  "signal": "HIGH"
}
```

#### 2. Predict Specific Date
```http
POST /predict
Content-Type: application/json

{
  "date": "2025-11-17"
}
```

**Response:**
```json
{
  "date": "2025-11-17",
  "prediction": true,
  "probability": 0.9412,
  "signal": "HIGH"
}
```

### Python Client Example

```python
import requests

# Predict latest
response = requests.get('http://localhost:5000/predict/latest')
result = response.json()

if result['signal'] == 'HIGH':
    print(f"⚠️ HIGH RISK: {result['probability']:.1%} chance of -5% drawdown")
elif result['signal'] == 'MEDIUM':
    print(f"⚡ MEDIUM RISK: {result['probability']:.1%} chance")
else:
    print(f"✓ LOW RISK: {result['probability']:.1%} chance")
```

---

## Maintenance & Updates

### Regular Tasks

#### 1. Update Data (Daily)
```bash
python asi_predictor.py
```

#### 2. Retrain Model (Monthly/Quarterly)
```bash
# Retrain with latest data
python train_downturn_predictor.py

# Compare new model performance
python -c "import json; print(json.load(open('models/model_summary.json')))"
```

#### 3. Monitor Model Performance
```bash
# Check recent predictions vs actual
python -c "
import pandas as pd
df = pd.read_csv('test_results.csv')
print('Accuracy:', df['correct'].mean())
print('Recent predictions:', df.tail(10))
"
```

### Model Retraining Schedule

**Recommended:**
- **Monthly:** Retrain with latest 3 months of data
- **Quarterly:** Full retrain with all available data
- **After major market events:** Immediate retrain

**Retraining Script:**
```bash
# Backup current model
cp models/lightgbm.pkl models/lightgbm_backup_$(date +%Y%m%d).pkl

# Retrain
python train_downturn_predictor.py

# Compare performance
python compare_models.py  # Create this script to compare old vs new
```

### Data Quality Checks

```python
# Check data completeness
import pandas as pd

df = pd.read_csv('ASI_close_prices_last_15_years_2025-12-05.csv')
print(f"Total records: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Missing values: {df.isna().sum()}")
print(f"Latest date: {df['date'].max()}")
```

---

## Troubleshooting

### Common Issues

#### 1. Missing Features Error
**Error:** `KeyError: 'feature_name'`

**Solution:**
```python
# Ensure all features are computed
trainer = DownturnPredictorTrainer('data.csv')
trainer.compute_features()  # Must call this first
```

#### 2. Model Not Found
**Error:** `FileNotFoundError: models/lightgbm.pkl`

**Solution:**
```bash
# Train models first
python train_downturn_predictor.py
```

#### 3. TradingView API Limits
**Error:** `Data access may be limited`

**Solution:**
- Use TradingView login credentials
- Or use database fallback (if configured)
- Or manually update CSV file

#### 4. NaN Values in Features
**Error:** `NaN values in feature matrix`

**Solution:**
```python
# Check which features have NaN
print(df[feature_names].isna().sum())

# Ensure sufficient historical data (need 200+ days for SMA200)
# Filter out rows with NaN
df = df.dropna(subset=feature_names)
```

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

For large datasets:

```python
# Use chunking for large files
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process(chunk)
```

---

## Feature Engineering Details

### 26 Features Used

**Momentum Indicators:**
- `momentum_10`, `momentum_20`, `momentum_50`
- `return_1d`, `return_3d`, `return_5d`, `return_7d`, `return_10d`

**Trend Indicators:**
- `rsi_14`, `rsi_7`
- `macd`, `macd_hist`
- `price_vs_sma20`, `price_vs_sma50`, `price_vs_sma200`
- `sma20_vs_sma50`, `sma50_vs_sma200`

**Volatility:**
- `volatility_20`, `volatility_60`

**Drawdowns:**
- `local_dd_20`, `local_dd_60`, `local_dd_120`

**Confirmation Signals:**
- `below_sma50`, `negative_momentum_20`, `rsi_below_50`, `macd_bearish`

### Feature Importance (Top 10)

Based on LightGBM model:
1. `momentum_20` - 20-day momentum
2. `local_dd_60` - 60-day local drawdown
3. `rsi_14` - 14-day RSI
4. `price_vs_sma50` - Price vs SMA50
5. `volatility_20` - 20-day volatility
6. `macd_hist` - MACD histogram
7. `return_10d` - 10-day return
8. `momentum_10` - 10-day momentum
9. `local_dd_20` - 20-day local drawdown
10. `sma20_vs_sma50` - SMA20 vs SMA50

---

## Best Practices

### 1. Version Control
- Commit model files separately
- Use model versioning: `lightgbm_v1.0.pkl`
- Track model performance in `model_summary.json`

### 2. Testing
- Always test on validation set before production
- Compare new model vs old model performance
- Monitor prediction accuracy over time

### 3. Monitoring
- Track prediction accuracy weekly
- Monitor false positive/negative rates
- Alert on model performance degradation

### 4. Documentation
- Document any model changes
- Keep changelog of model versions
- Document feature engineering changes

---

## Quick Reference

### Key Files
- **Data:** `ASI_close_prices_last_15_years_YYYY-MM-DD.csv`
- **Model:** `models/lightgbm.pkl`
- **Training:** `train_downturn_predictor.py`
- **Prediction:** `predict_november_2025.py` (example)

### Key Commands
```bash
# Fetch data
python asi_predictor.py

# Train models
python train_downturn_predictor.py

# Predict for dates
python predict_november_2025.py

# Run API server
python api_server.py
```

### Key Thresholds
- **Prediction Target:** -5% drawdown in 10 days
- **Probability Threshold:** 0.5 (default)
- **High Confidence:** ≥ 0.7
- **Medium Confidence:** 0.5 - 0.7
- **Low Confidence:** < 0.5

---

## Support & Resources

### Documentation Files
- `SUMMARY_REPORT.md` - Analysis summary
- `DRAWDOWN_DEPTH_DURATION_REPORT.md` - Drawdown statistics
- `DEVELOPER_GUIDE.md` - This file

### Model Files
- `models/model_summary.json` - Model performance
- `models/lightgbm.pkl` - Trained model
- `test_results.csv` - Test set results

### Contact
For issues or questions, refer to the codebase documentation or create an issue in the repository.

---

**Last Updated:** December 5, 2025  
**System Version:** 1.0  
**Model Version:** LightGBM (AUC: 0.9818)

