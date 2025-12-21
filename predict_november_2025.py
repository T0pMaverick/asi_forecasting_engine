#!/usr/bin/env python3
"""
Predict -5% drawdown signals for November 2025
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def compute_features(df):
    """Compute technical indicators (same as training script)"""
    # Moving Averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # RSI
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)
    
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['rsi_7'] = calculate_rsi(df['close'], 7)
    
    # Volatility
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
    df['volatility_60'] = df['returns'].rolling(60).std() * np.sqrt(252)
    
    # Momentum
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_20'] = df['close'].pct_change(20)
    df['momentum_50'] = df['close'].pct_change(50)
    
    # Trend indicators
    df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    df['price_vs_sma200'] = (df['close'] - df['sma_200']) / df['sma_200']
    df['sma20_vs_sma50'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
    df['sma50_vs_sma200'] = (df['sma_50'] - df['sma_200']) / df['sma_200']
    
    # Local drawdowns
    for w in [20, 60, 120]:
        peak_w = df['close'].rolling(w).max()
        df[f'local_dd_{w}'] = (df['close'] - peak_w) / peak_w
    
    # Returns
    df['return_1d'] = df['close'].pct_change(1)
    df['return_3d'] = df['close'].pct_change(3)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_7d'] = df['close'].pct_change(7)
    df['return_10d'] = df['close'].pct_change(10)
    
    # Confirmation signals
    df['below_sma50'] = (df['close'] < df['sma_50']).astype(int)
    df['negative_momentum_20'] = (df['momentum_20'] < 0).astype(int)
    df['rsi_below_50'] = (df['rsi_14'] < 50).astype(int)
    df['macd_bearish'] = (df['macd'] < df['macd_signal']).astype(int)
    
    return df

def create_labels_for_november(df, threshold=-0.05, prediction_days=10):
    """Create actual labels for November dates"""
    df['will_have_drawdown'] = False
    df['max_drawdown_in_period'] = 0.0
    
    for i in range(len(df) - prediction_days):
        current_price = df.loc[i, 'close']
        peak_price = current_price
        max_drawdown = 0.0
        
        lookahead_window = df.iloc[i+1:i+1+prediction_days]
        
        for j, row in lookahead_window.iterrows():
            price = row['close']
            if price > peak_price:
                peak_price = price
            drawdown = (price - peak_price) / peak_price
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        
        if max_drawdown <= threshold:
            df.loc[i, 'will_have_drawdown'] = True
        df.loc[i, 'max_drawdown_in_period'] = max_drawdown
    
    return df

def main():
    print("="*80)
    print("NOVEMBER 2025 PREDICTIONS")
    print("="*80)
    
    # Load ASI data
    print("\nLoading ASI data...")
    df = pd.read_csv('ASI_close_prices_last_15_years_2025-12-05.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['close'] = pd.to_numeric(df['close'])
    
    # Filter to November 2025
    november_df = df[(df['date'] >= '2025-11-01') & (df['date'] <= '2025-11-30')].copy()
    
    if len(november_df) == 0:
        print("No November 2025 data found!")
        return
    
    print(f"Found {len(november_df)} trading days in November 2025")
    print(f"Date range: {november_df['date'].min()} to {november_df['date'].max()}")
    
    # We need full historical data to compute features properly
    # So use the full dataframe for feature computation
    print("\nComputing features...")
    df = compute_features(df)
    
    # Extract November dates with features
    november_with_features = df[(df['date'] >= '2025-11-01') & (df['date'] <= '2025-11-30')].copy()
    
    # Feature columns (same as training)
    feature_cols = [
        'rsi_14', 'rsi_7', 'macd', 'macd_hist', 
        'momentum_10', 'momentum_20', 'momentum_50',
        'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
        'sma20_vs_sma50', 'sma50_vs_sma200',
        'volatility_20', 'volatility_60',
        'local_dd_20', 'local_dd_60', 'local_dd_120',
        'return_1d', 'return_3d', 'return_5d', 'return_7d', 'return_10d',
        'below_sma50', 'negative_momentum_20', 'rsi_below_50', 'macd_bearish'
    ]
    
    # Select features
    available_features = [col for col in feature_cols if col in november_with_features.columns]
    X_november = november_with_features[available_features].copy()
    
    # Remove rows with NaN
    valid_mask = ~X_november.isna().any(axis=1)
    X_november = X_november[valid_mask]
    november_dates = november_with_features.loc[valid_mask, 'date']
    november_close = november_with_features.loc[valid_mask, 'close']
    
    print(f"Valid samples for prediction: {len(X_november)}")
    
    if len(X_november) == 0:
        print("No valid samples after removing NaN values!")
        return
    
    # Load trained model
    print("\nLoading trained model...")
    with open('models/lightgbm.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
    
    print(f"Model: {model_data['model_name']}")
    print(f"Features: {len(feature_names)}")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_november)
    y_pred_proba = model.predict_proba(X_november)[:, 1]
    
    # Create actual labels for comparison
    print("\nComputing actual outcomes...")
    df_labeled = create_labels_for_november(df.copy(), threshold=-0.05, prediction_days=10)
    november_actual = df_labeled[(df_labeled['date'] >= '2025-11-01') & 
                                  (df_labeled['date'] <= '2025-11-30')].copy()
    november_actual = november_actual.loc[valid_mask]
    y_actual = november_actual['will_have_drawdown'].values
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'date': november_dates.values,
        'close': november_close.values,
        'actual': y_actual,
        'predicted': y_pred,
        'predicted_probability': y_pred_proba
    })
    
    # Add prediction correctness
    results_df['correct'] = (results_df['actual'] == results_df['predicted']).astype(int)
    
    # Add prediction type
    results_df['prediction_type'] = results_df.apply(
        lambda row: 'TP' if (row['actual'] == True and row['predicted'] == True) else
                   'TN' if (row['actual'] == False and row['predicted'] == False) else
                   'FP' if (row['actual'] == False and row['predicted'] == True) else
                   'FN', axis=1
    )
    
    # Add signal strength
    results_df['signal_strength'] = results_df['predicted_probability'].apply(
        lambda x: 'HIGH' if x >= 0.7 else 'MEDIUM' if x >= 0.5 else 'LOW'
    )
    
    # Sort by date
    results_df = results_df.sort_values('date').reset_index(drop=True)
    
    # Save results
    output_file = 'november_2025_predictions.csv'
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("PREDICTION RESULTS FOR NOVEMBER 2025")
    print("="*80)
    print(f"\nTotal predictions: {len(results_df)}")
    print(f"Date range: {results_df['date'].min()} to {results_df['date'].max()}")
    print(f"\nPredictions:")
    print(f"  Positive signals: {results_df['predicted'].sum()}")
    print(f"  Actual positives: {results_df['actual'].sum()}")
    print(f"  Correct predictions: {results_df['correct'].sum()} ({results_df['correct'].mean()*100:.2f}%)")
    
    print(f"\nSignal Strength Distribution:")
    print(results_df['signal_strength'].value_counts())
    
    print(f"\nHigh Confidence Signals (>= 0.7):")
    high_conf = results_df[results_df['predicted_probability'] >= 0.7]
    if len(high_conf) > 0:
        print(high_conf[['date', 'close', 'predicted_probability', 'predicted', 'actual']].to_string(index=False))
    else:
        print("  None")
    
    print(f"\nMedium Confidence Signals (0.5-0.7):")
    med_conf = results_df[(results_df['predicted_probability'] >= 0.5) & 
                          (results_df['predicted_probability'] < 0.7)]
    if len(med_conf) > 0:
        print(med_conf[['date', 'close', 'predicted_probability', 'predicted', 'actual']].to_string(index=False))
    else:
        print("  None")
    
    print(f"\nAll November Predictions:")
    print(results_df[['date', 'close', 'predicted_probability', 'predicted', 'actual', 'signal_strength']].to_string(index=False))
    
    print(f"\nResults saved to {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()

