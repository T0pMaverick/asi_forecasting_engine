#!/usr/bin/env python3
"""
Simple XGBoost Model Performance Tester
Tests XGBoost models against real ASI data (June-December 2025)
Creates investor-ready CSV comparison report
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_rsi(series, period=14):
    """Calculate RSI indicator"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)

def engineer_features(df):
    """Engineer features for XGBoost model"""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    
    # Moving averages
    for period in [10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
    
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # RSI
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['rsi_7'] = calculate_rsi(df['close'], 7)
    
    # Volatility
    for period in [10, 20, 60]:
        df[f'volatility_{period}'] = df['returns'].rolling(period).std() * np.sqrt(252)
    
    # Returns and momentum
    for period in [1, 3, 5, 7, 10, 20, 50]:
        df[f'return_{period}d'] = df['close'].pct_change(period)
        df[f'momentum_{period}'] = df['close'].pct_change(period)
    
    # Price vs moving averages
    for period in [20, 50, 200]:
        df[f'price_vs_sma{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
    
    df['sma20_vs_sma50'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
    df['sma50_vs_sma200'] = (df['sma_50'] - df['sma_200']) / df['sma_200']
    
    # Local drawdowns
    for w in [20, 60, 120]:
        peak_w = df['close'].rolling(w).max()
        df[f'local_dd_{w}'] = (df['close'] - peak_w) / peak_w
    
    # Binary indicators
    df['below_sma50'] = (df['close'] < df['sma_50']).astype(int)
    df['negative_momentum_20'] = (df['momentum_20'] < 0).astype(int)
    df['rsi_below_50'] = (df['rsi_14'] < 50).astype(int)
    df['macd_bearish'] = (df['macd'] < df['macd_signal']).astype(int)
    
    return df

def check_actual_downturn(df, start_idx):
    """Check if actual -5% downturn occurred in next 10 days"""
    if start_idx >= len(df) - 10:
        return False, None, None
    
    start_price = df.loc[start_idx, 'close']
    
    for i in range(start_idx + 1, min(start_idx + 11, len(df))):
        current_price = df.loc[i, 'close']
        drop = (current_price - start_price) / start_price
        
        if drop <= -0.05:  # -5% threshold
            days_to_drop = i - start_idx
            max_drop_in_period = drop
            
            # Find maximum drop in remaining days
            for j in range(i, min(start_idx + 11, len(df))):
                future_drop = (df.loc[j, 'close'] - start_price) / start_price
                if future_drop < max_drop_in_period:
                    max_drop_in_period = future_drop
            
            return True, days_to_drop, max_drop_in_period
    
    return False, None, None

def load_xgboost_model(model_path):
    """Load XGBoost model"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

def test_xgboost_models():
    """Main testing function"""
    print("=" * 60)
    print("XGBOOST MODEL PERFORMANCE TESTING")
    print("Testing Period: June 6, 2025 - December 5, 2025")
    print("=" * 60)
    
    # Load real data
    print("\n1. Loading real market data...")
    real_data = pd.read_csv('.\\risk_analysis\\prod testing.csv')
    real_data['date'] = pd.to_datetime(real_data['date'], format='%m/%d/%Y')
    real_data = real_data.sort_values('date').reset_index(drop=True)
    real_data['close'] = pd.to_numeric(real_data['close'])
    
    print(f"✓ Loaded {len(real_data)} records")
    print(f"✓ Date range: {real_data['date'].min().strftime('%Y-%m-%d')} to {real_data['date'].max().strftime('%Y-%m-%d')}")
    
    # Load XGBoost model
    print("\n2. Loading XGBoost model...")
    model_files = ['xgboost.pkl', 'models/xgboost.pkl', 'comprehensive_models/classification_xgboost.pkl']
    model_data = None
    
    for model_file in model_files:
        if os.path.exists(model_file):
            model_data = load_xgboost_model(model_file)
            if model_data:
                print(f"✓ Loaded model: {model_file}")
                break
    
    if not model_data:
        print("❌ Could not find XGBoost model. Please ensure model file exists.")
        return
    
    # Extract model components
    model = model_data['model']
    scaler = model_data.get('scaler', None)
    feature_names = model_data['feature_names']
    
    print(f"✓ Model loaded with {len(feature_names)} features")
    
    # Engineer features
    print("\n3. Engineering features...")
    df_features = engineer_features(real_data)
    print("✓ Features engineered")
    
    # Test predictions
    print("\n4. Testing predictions...")
    results = []
    
    for i in range(len(df_features) - 10):  # Leave 10 days for outcome checking
        try:
            row = df_features.iloc[i]
            test_date = row['date']
            asi_price = row['close']
            
            # Prepare features
            feature_values = []
            for feature in feature_names:
                value = row.get(feature, 0)
                if pd.isna(value):
                    value = 0
                feature_values.append(value)
            
            X = np.array([feature_values])
            
            # Apply scaling if available
            if scaler:
                X = scaler.transform(X)
            
            # Make prediction
            prediction = bool(model.predict(X)[0])
            probability = float(model.predict_proba(X)[0, 1])
            
            # Determine risk level
            if prediction:
                if probability >= 0.8:
                    risk_level = "HIGH"
                elif probability >= 0.6:
                    risk_level = "MODERATE"
                else:
                    risk_level = "LOW"
            else:
                risk_level = "STABLE"
            
            # Check actual outcome
            actual_downturn, days_to_drop, max_drop = check_actual_downturn(df_features, i)
            
            # Performance metrics
            correct_prediction = (prediction == actual_downturn)
            
            # Business impact (for $1M portfolio)
            if prediction:
                predicted_loss = 1000000 * 0.05  # Assume 5% loss if downturn predicted
            else:
                predicted_loss = 0
            
            actual_loss = 1000000 * abs(max_drop) if max_drop else 0
            
            results.append({
                'Date': test_date.strftime('%Y-%m-%d'),
                'ASI_Price': round(asi_price, 2),
                'Predicted_Downturn': prediction,
                'Downturn_Probability': round(probability, 4),
                'Risk_Level': risk_level,
                'Actual_Downturn': actual_downturn,
                'Actual_Days_to_Drop': days_to_drop,
                'Actual_Max_Drop_Pct': round(max_drop * 100, 2) if max_drop else None,
                'Prediction_Correct': correct_prediction,
                'Predicted_Loss_1M_Portfolio': round(predicted_loss, 0),
                'Actual_Loss_1M_Portfolio': round(actual_loss, 0)
            })
            
        except Exception as e:
            print(f"Error processing {test_date}: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    print("\n5. Calculating performance metrics...")
    
    total_predictions = len(results_df)
    correct_predictions = results_df['Prediction_Correct'].sum()
    accuracy = correct_predictions / total_predictions
    
    # Confusion matrix
    tp = len(results_df[(results_df['Predicted_Downturn'] == True) & (results_df['Actual_Downturn'] == True)])
    tn = len(results_df[(results_df['Predicted_Downturn'] == False) & (results_df['Actual_Downturn'] == False)])
    fp = len(results_df[(results_df['Predicted_Downturn'] == True) & (results_df['Actual_Downturn'] == False)])
    fn = len(results_df[(results_df['Predicted_Downturn'] == False) & (results_df['Actual_Downturn'] == True)])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    actual_downturns = results_df['Actual_Downturn'].sum()
    predicted_downturns = results_df['Predicted_Downturn'].sum()
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    results_file = f'XGBoost_Performance_Report_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    
    # Create summary report
    summary_data = {
        'Metric': [
            'Testing Period',
            'Total Predictions',
            'Overall Accuracy',
            'Precision (No False Alarms)',
            'Recall (Catches Real Downturns)',
            'F1 Score',
            'True Positives',
            'True Negatives', 
            'False Positives (False Alarms)',
            'False Negatives (Missed)',
            'Actual Downturns in Period',
            'Predicted Downturns',
            'Average Predicted Loss ($1M)',
            'Average Actual Loss ($1M)'
        ],
        'Value': [
            f"{results_df['Date'].min()} to {results_df['Date'].max()}",
            total_predictions,
            f"{accuracy:.1%}",
            f"{precision:.1%}",
            f"{recall:.1%}",
            f"{f1_score:.3f}",
            tp,
            tn,
            fp,
            fn,
            actual_downturns,
            predicted_downturns,
            f"${results_df['Predicted_Loss_1M_Portfolio'].mean():,.0f}",
            f"${results_df['Actual_Loss_1M_Portfolio'].mean():,.0f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = f'XGBoost_Summary_{timestamp}.csv'
    summary_df.to_csv(summary_file, index=False)
    
    # Print results
    print("\n" + "=" * 60)
    print("INVESTOR PRESENTATION RESULTS")
    print("=" * 60)
    print(f"📊 TESTING SUMMARY:")
    print(f"   Testing Period: {results_df['Date'].min()} to {results_df['Date'].max()}")
    print(f"   Total Predictions: {total_predictions:,}")
    print(f"   Overall Accuracy: {accuracy:.1%}")
    print(f"   Precision: {precision:.1%} (How often predictions were right)")
    print(f"   Recall: {recall:.1%} (How many real downturns were caught)")
    print(f"   F1 Score: {f1_score:.3f}")
    
    print(f"\n📈 DETAILED BREAKDOWN:")
    print(f"   ✅ True Positives: {tp} (Correctly predicted downturns)")
    print(f"   ✅ True Negatives: {tn} (Correctly predicted stability)")  
    print(f"   ❌ False Positives: {fp} (False alarms)")
    print(f"   ❌ False Negatives: {fn} (Missed downturns)")
    
    print(f"\n📊 MARKET ANALYSIS:")
    print(f"   Actual Downturns in Period: {actual_downturns}")
    print(f"   Model Predicted Downturns: {predicted_downturns}")
    
    print(f"\n💰 PORTFOLIO IMPACT (Based on $1M Portfolio):")
    print(f"   Average Predicted Loss: ${results_df['Predicted_Loss_1M_Portfolio'].mean():,.0f}")
    print(f"   Average Actual Loss: ${results_df['Actual_Loss_1M_Portfolio'].mean():,.0f}")
    
    print("\n" + "=" * 60)
    print("📁 INVESTOR-READY FILES CREATED:")
    print(f"   📊 Detailed Report: {results_file}")
    print(f"   📋 Executive Summary: {summary_file}")
    print("=" * 60)
    print("✅ Ready to show to investors!")
    
    return results_file, summary_file

if __name__ == "__main__":
    test_xgboost_models()