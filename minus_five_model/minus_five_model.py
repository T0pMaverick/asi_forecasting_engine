#!/usr/bin/env python3
"""
ASI Comprehensive Forecasting System with Walk-Forward Validation
Features:
- 6 forecasting models (XGBoost, LightGBM, ARIMA, SARIMA, SARIMAX)
- Walk-forward validation
- 10-day price forecasting
- Downturn detection and analysis
- Severity and duration prediction
Author: Dushan
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    mean_absolute_percentage_error
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Advanced Models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

# Time Series Models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available. Install with: pip install statsmodels")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

class ComprehensiveASIForecastingSystem:
    """
    Complete ASI forecasting system with walk-forward validation
    """
    
    def __init__(self, training_csv, testing_csv):
        """Initialize the forecasting system"""
        print("="*80)
        print("ASI COMPREHENSIVE FORECASTING SYSTEM")
        print("="*80)
        print("Features:")
        print("✓ 6 Forecasting Models (XGBoost, LightGBM, ARIMA, SARIMA, SARIMAX, )")
        print("✓ Walk-Forward Validation")
        print("✓ 10-Day Price Forecasting")
        print("✓ Downturn Detection (-5% threshold)")
        print("✓ Severity & Duration Analysis")
        print("✓ Production Testing & CSV Export")
        print("="*80)
        
        # Load training data
        self.df = pd.read_csv(training_csv)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.df['close'] = pd.to_numeric(self.df['close'])
        
        # Load testing data
        self.test_df = pd.read_csv(testing_csv)
        self.test_df['date'] = pd.to_datetime(self.test_df['date'])
        self.test_df = self.test_df.sort_values('date').reset_index(drop=True)
        self.test_df['close'] = pd.to_numeric(self.test_df['close'])
        
        print(f"\nTraining data: {len(self.df)} records from {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"Testing data: {len(self.test_df)} records from {self.test_df['date'].min()} to {self.test_df['date'].max()}")
        
        # Initialize model containers
        self.models = {}
        self.model_performance = {}
        self.walk_forward_results = {}
        self.forecast_horizon = 10
        
    def create_enhanced_features(self):
        """Create comprehensive technical indicators for forecasting"""
        print("\nCreating enhanced features for forecasting...")
        
        # Basic price features
        self.df['returns'] = self.df['close'].pct_change()
        self.df['log_returns'] = np.log(self.df['close']).diff()
        self.df['price_change'] = self.df['close'].diff()
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10, 20]:
            self.df[f'close_lag_{lag}'] = self.df['close'].shift(lag)
            self.df[f'returns_lag_{lag}'] = self.df['returns'].shift(lag)
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            self.df[f'sma_{period}'] = self.df['close'].rolling(period).mean()
            self.df[f'ema_{period}'] = self.df['close'].ewm(span=period, adjust=False).mean()
        
        # Technical Indicators
        self.df['ema_12'] = self.df['close'].ewm(span=12, adjust=False).mean()
        self.df['ema_26'] = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['macd'] = self.df['ema_12'] - self.df['ema_26']
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
        self.df['macd_hist'] = self.df['macd'] - self.df['macd_signal']
        
        # RSI
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = -delta.clip(upper=0).rolling(period).mean()
            rs = gain / (loss + 1e-9)
            return 100 - 100 / (1 + rs)
        
        self.df['rsi_14'] = calculate_rsi(self.df['close'], 14)
        self.df['rsi_7'] = calculate_rsi(self.df['close'], 7)
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = self.df[f'sma_{period}']
            std = self.df['close'].rolling(period).std()
            self.df[f'bb_upper_{period}'] = sma + 2 * std
            self.df[f'bb_lower_{period}'] = sma - 2 * std
            self.df[f'bb_position_{period}'] = (self.df['close'] - self.df[f'bb_lower_{period}']) / (self.df[f'bb_upper_{period}'] - self.df[f'bb_lower_{period}'])
        
        # Volatility
        for period in [5, 10, 20, 60]:
            self.df[f'volatility_{period}'] = self.df['returns'].rolling(period).std() * np.sqrt(252)
        
        # Volatility ratios (after all volatilities are calculated)
        for period in [5, 10, 20, 60]:
            if period != 20:  # Don't create ratio for volatility_20/volatility_20
                self.df[f'volatility_ratio_{period}'] = self.df[f'volatility_{period}'] / (self.df['volatility_20'] + 1e-9)
        
        # Momentum indicators
        for period in [5, 10, 20, 50]:
            self.df[f'momentum_{period}'] = self.df['close'].pct_change(period)
            self.df[f'roc_{period}'] = ((self.df['close'] - self.df['close'].shift(period)) / self.df['close'].shift(period)) * 100
        
        # Price position relative to moving averages
        for period in [20, 50, 200]:
            self.df[f'price_vs_sma{period}'] = (self.df['close'] - self.df[f'sma_{period}']) / (self.df[f'sma_{period}'] + 1e-9)
        
        # Moving average crossovers
        self.df['sma20_vs_sma50'] = (self.df['sma_20'] - self.df['sma_50']) / (self.df['sma_50'] + 1e-9)
        self.df['sma50_vs_sma200'] = (self.df['sma_50'] - self.df['sma_200']) / (self.df['sma_200'] + 1e-9)
        
        # Advanced features
        self.df['high_low_ratio'] = self.df['close'] / (self.df['close'].rolling(20).min() + 1e-9)
        self.df['support_resistance'] = self.df['close'] / (self.df['close'].rolling(252).max() + 1e-9)
        
        # Trend features
        self.df['trend_5'] = np.where(self.df['sma_5'] > self.df['sma_5'].shift(1), 1, 0)
        self.df['trend_20'] = np.where(self.df['sma_20'] > self.df['sma_20'].shift(1), 1, 0)
        
        # Calendar features
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        
        print(f"Created {len(self.df.columns)} total features")
        
    def prepare_features_for_models(self):
        """Select and prepare features for modeling"""
        
        # Define feature columns (excluding date, close, and target-related columns)
        feature_cols = [
            # Price-based features
            'returns', 'log_returns', 'price_change',
            
            # Lagged features
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5', 'close_lag_10',
            'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_5',
            
            # Technical indicators
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'macd', 'macd_hist', 'rsi_14', 'rsi_7',
            
            # Bollinger Bands
            'bb_position_20', 'bb_position_50',
            
            # Volatility
            'volatility_5', 'volatility_10', 'volatility_20', 'volatility_ratio_5', 'volatility_ratio_10',
            
            # Momentum
            'momentum_5', 'momentum_10', 'momentum_20', 'roc_5', 'roc_10', 'roc_20',
            
            # Relative position
            'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
            'sma20_vs_sma50', 'sma50_vs_sma200',
            
            # Advanced features
            'high_low_ratio', 'support_resistance', 'trend_5', 'trend_20',
            
            # Calendar features
            'day_of_week', 'month', 'quarter'
        ]
        
        # Keep only available features
        self.feature_names = [col for col in feature_cols if col in self.df.columns]
        print(f"Selected {len(self.feature_names)} features for modeling")
        
        return self.feature_names
    

    
    def train_xgboost_model(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model for multi-step forecasting"""
        if not XGBOOST_AVAILABLE:
            return None
        
        print("Training XGBoost model...")
        
        # Multi-output XGBoost (train separate model for each day)
        models = {}
        train_metrics_list = []
        val_metrics_list = []
        
        for day in range(self.forecast_horizon):
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            if y_train.ndim > 1:
                # Fit model
                model.fit(X_train, y_train[:, day])
                
                # Get predictions for both train and validation
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                # Calculate metrics
                train_metrics = self.calculate_model_metrics(y_train[:, day], train_pred)
                val_metrics = self.calculate_model_metrics(y_val[:, day], val_pred)
                
                train_metrics_list.append(train_metrics)
                val_metrics_list.append(val_metrics)
                
            else:
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                train_metrics = self.calculate_model_metrics(y_train, train_pred)
                val_metrics = self.calculate_model_metrics(y_val, val_pred)
                
                train_metrics_list.append(train_metrics)
                val_metrics_list.append(val_metrics)
            
            models[f'day_{day+1}'] = {
                'model': model, 
                'train_metrics': train_metrics_list[-1],
                'val_metrics': val_metrics_list[-1]
            }
        
        # Aggregate metrics
        avg_train_mae = np.mean([m['mae'] for m in train_metrics_list])
        avg_val_mae = np.mean([m['mae'] for m in val_metrics_list])
        avg_train_mape = np.mean([m['mape'] for m in train_metrics_list])
        avg_val_mape = np.mean([m['mape'] for m in val_metrics_list])
        
        # Overfitting check
        overfitting_score = avg_val_mae / avg_train_mae if avg_train_mae > 0 else float('inf')
        is_overfitting = overfitting_score > 2.0  # If validation error is 2x training error
        
        return {
            'models': models, 
            'type': 'multi_model', 
            'val_mae': avg_val_mae,
            'train_mae': avg_train_mae,
            'val_mape': avg_val_mape,
            'train_mape': avg_train_mape,
            'overfitting_score': overfitting_score,
            'is_overfitting': is_overfitting
        }
    
    def train_lightgbm_model(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        if not LIGHTGBM_AVAILABLE:
            return None
        
        print("Training LightGBM model...")
        
        models = {}
        train_metrics_list = []
        val_metrics_list = []
        
        for day in range(self.forecast_horizon):
            model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            if y_train.ndim > 1:
                model.fit(X_train, y_train[:, day])
                
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                train_metrics = self.calculate_model_metrics(y_train[:, day], train_pred)
                val_metrics = self.calculate_model_metrics(y_val[:, day], val_pred)
                
                train_metrics_list.append(train_metrics)
                val_metrics_list.append(val_metrics)
                
            else:
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                train_metrics = self.calculate_model_metrics(y_train, train_pred)
                val_metrics = self.calculate_model_metrics(y_val, val_pred)
                
                train_metrics_list.append(train_metrics)
                val_metrics_list.append(val_metrics)
            
            models[f'day_{day+1}'] = {
                'model': model, 
                'train_metrics': train_metrics_list[-1],
                'val_metrics': val_metrics_list[-1]
            }
        
        # Aggregate metrics
        avg_train_mae = np.mean([m['mae'] for m in train_metrics_list])
        avg_val_mae = np.mean([m['mae'] for m in val_metrics_list])
        avg_train_mape = np.mean([m['mape'] for m in train_metrics_list])
        avg_val_mape = np.mean([m['mape'] for m in val_metrics_list])
        
        # Overfitting check
        overfitting_score = avg_val_mae / avg_train_mae if avg_train_mae > 0 else float('inf')
        is_overfitting = overfitting_score > 2.0
        
        return {
            'models': models, 
            'type': 'multi_model', 
            'val_mae': avg_val_mae,
            'train_mae': avg_train_mae,
            'val_mape': avg_val_mape,
            'train_mape': avg_train_mape,
            'overfitting_score': overfitting_score,
            'is_overfitting': is_overfitting
        }
    
    def train_arima_model(self, train_prices, val_prices):
        """Train ARIMA model"""
        if not STATSMODELS_AVAILABLE:
            return None
        
        print("Training ARIMA model...")
        
        try:
            # Fit ARIMA model
            model = ARIMA(train_prices, order=(2, 1, 2))
            fitted_model = model.fit()
            
            # Forecast for validation
            forecast = fitted_model.forecast(steps=len(val_prices))
            
            # Calculate metrics
            val_metrics = self.calculate_model_metrics(val_prices, forecast)
            
            # For train metrics, use in-sample fitted values
            train_fitted = fitted_model.fittedvalues
            train_metrics = self.calculate_model_metrics(train_prices[len(train_prices)-len(train_fitted):], train_fitted)
            
            # Overfitting check
            overfitting_score = val_metrics['mae'] / train_metrics['mae'] if train_metrics['mae'] > 0 else float('inf')
            is_overfitting = overfitting_score > 2.0
            
            return {
                'model': fitted_model, 
                'val_mae': val_metrics['mae'],
                'train_mae': train_metrics['mae'],
                'val_mape': val_metrics['mape'],
                'train_mape': train_metrics['mape'],
                'overfitting_score': overfitting_score,
                'is_overfitting': is_overfitting,
                'type': 'arima'
            }
        except Exception as e:
            print(f"ARIMA training failed: {e}")
            return None
    
    def train_sarima_model(self, train_prices, val_prices):
        """Train SARIMA model"""
        if not STATSMODELS_AVAILABLE:
            return None
        
        print("Training SARIMA model...")
        
        try:
            # SARIMA with seasonal period of 52 (weekly seasonality)
            model = SARIMAX(train_prices, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            fitted_model = model.fit(disp=False)
            
            forecast = fitted_model.forecast(steps=len(val_prices))
            
            # Calculate metrics
            val_metrics = self.calculate_model_metrics(val_prices, forecast)
            
            # For train metrics, use in-sample fitted values
            train_fitted = fitted_model.fittedvalues
            train_metrics = self.calculate_model_metrics(train_prices[len(train_prices)-len(train_fitted):], train_fitted)
            
            # Overfitting check
            overfitting_score = val_metrics['mae'] / train_metrics['mae'] if train_metrics['mae'] > 0 else float('inf')
            is_overfitting = overfitting_score > 2.0
            
            return {
                'model': fitted_model, 
                'val_mae': val_metrics['mae'],
                'train_mae': train_metrics['mae'],
                'val_mape': val_metrics['mape'],
                'train_mape': train_metrics['mape'],
                'overfitting_score': overfitting_score,
                'is_overfitting': is_overfitting,
                'type': 'sarima'
            }
        except Exception as e:
            print(f"SARIMA training failed: {e}")
            return None
    
    def train_sarimax_model(self, train_prices, val_prices, exog_train=None, exog_val=None):
        """Train SARIMAX model with exogenous variables"""
        if not STATSMODELS_AVAILABLE:
            return None
        
        print("Training SARIMAX model...")
        
        try:
            model = SARIMAX(train_prices, exog=exog_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            fitted_model = model.fit(disp=False)
            
            forecast = fitted_model.forecast(steps=len(val_prices), exog=exog_val)
            
            # Calculate metrics
            val_metrics = self.calculate_model_metrics(val_prices, forecast)
            
            # For train metrics, use in-sample fitted values
            train_fitted = fitted_model.fittedvalues
            train_metrics = self.calculate_model_metrics(train_prices[len(train_prices)-len(train_fitted):], train_fitted)
            
            # Overfitting check
            overfitting_score = val_metrics['mae'] / train_metrics['mae'] if train_metrics['mae'] > 0 else float('inf')
            is_overfitting = overfitting_score > 2.0
            
            return {
                'model': fitted_model, 
                'val_mae': val_metrics['mae'],
                'train_mae': train_metrics['mae'],
                'val_mape': val_metrics['mape'],
                'train_mape': train_metrics['mape'],
                'overfitting_score': overfitting_score,
                'is_overfitting': is_overfitting,
                'type': 'sarimax'
            }
        except Exception as e:
            print(f"SARIMAX training failed: {e}")
            return None
    
    def calculate_model_metrics(self, y_true, y_pred):
        """Calculate comprehensive model metrics including MAPE and overfitting check"""
        # Handle multi-dimensional targets
        if y_true.ndim > 1:
            mae_scores = []
            mape_scores = []
            rmse_scores = []
            
            for i in range(y_true.shape[1]):
                mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
                mape = mean_absolute_percentage_error(y_true[:, i], y_pred[:, i]) * 100
                rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
                
                mae_scores.append(mae)
                mape_scores.append(mape)
                rmse_scores.append(rmse)
            
            return {
                'mae': np.mean(mae_scores),
                'mape': np.mean(mape_scores),
                'rmse': np.mean(rmse_scores),
                'mae_std': np.std(mae_scores),
                'mape_std': np.std(mape_scores)
            }
        else:
            mae = mean_absolute_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            return {
                'mae': mae,
                'mape': mape,
                'rmse': rmse,
                'mae_std': 0,
                'mape_std': 0
            }
    
    def prepare_targets_for_training(self, prices, start_idx, forecast_horizon=10):
        """Prepare multi-step targets for training"""
        targets = []
        for i in range(start_idx, len(prices) - forecast_horizon):
            # Next 10 days prices
            future_prices = prices[i+1:i+forecast_horizon+1].values
            if len(future_prices) == forecast_horizon:
                targets.append(future_prices)
        
        return np.array(targets)
    
    def walk_forward_validation(self, min_train_years=5):
        """Implement walk-forward validation"""
        print("\n" + "="*60)
        print("WALK-FORWARD VALIDATION")
        print("="*60)
        
        # Determine validation splits
        total_years = (self.df['date'].max() - self.df['date'].min()).days / 365.25
        print(f"Total data span: {total_years:.1f} years")
        
        if total_years < min_train_years + 1:
            print(f"Insufficient data for walk-forward validation (need {min_train_years + 1}+ years)")
            return {}
        
        # Create time-based splits
        start_year = self.df['date'].min().year + min_train_years
        end_year = self.df['date'].max().year
        
        results = {}
        
        for test_year in range(start_year, end_year + 1):
            print(f"\nValidation fold: Training up to {test_year-1}, Testing {test_year}")
            
            # Split data
            train_mask = self.df['date'].dt.year < test_year
            test_mask = self.df['date'].dt.year == test_year
            
            if test_mask.sum() < 30:  # Need at least 30 test samples
                continue
            
            train_data = self.df[train_mask].copy()
            test_data = self.df[test_mask].copy()
            
            print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")
            
            # Prepare features and targets
            feature_cols = self.prepare_features_for_models()
            
            # Remove NaN rows
            train_data = train_data.dropna(subset=feature_cols + ['close'])
            test_data = test_data.dropna(subset=feature_cols + ['close'])
            
            if len(train_data) < 500:  # Need minimum training data
                continue
            
            # Features
            X_train = train_data[feature_cols].values
            X_test = test_data[feature_cols].values
            
            # Targets (next 10-day prices)
            y_train = self.prepare_targets_for_training(
                train_data['close'], 
                start_idx=0, 
                forecast_horizon=self.forecast_horizon
            )
            
            if len(y_train) == 0:
                continue
            
            # Align features with targets
            X_train = X_train[:len(y_train)]
            
            # Split training into train/val
            split_idx = int(0.8 * len(X_train))
            X_train_split = X_train[:split_idx]
            y_train_split = y_train[:split_idx]
            X_val_split = X_train[split_idx:]
            y_val_split = y_train[split_idx:]
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_split)
            X_val_scaled = scaler.transform(X_val_split)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            fold_models = {}
            
            # 1. XGBoost
            xgb_model = self.train_xgboost_model(X_train_scaled, y_train_split, X_val_scaled, y_val_split)
            if xgb_model:
                fold_models['XGBoost'] = xgb_model
            
            # 2. LightGBM
            lgb_model = self.train_lightgbm_model(X_train_scaled, y_train_split, X_val_scaled, y_val_split)
            if lgb_model:
                fold_models['LightGBM'] = lgb_model
            
            # 3. ARIMA (price-based)
            arima_model = self.train_arima_model(
                train_data['close'].values[-500:],  # Use recent data
                test_data['close'].values[:min(30, len(test_data))]  # Forecast limited period
            )
            if arima_model:
                fold_models['ARIMA'] = arima_model
            
            # 4. SARIMA
            sarima_model = self.train_sarima_model(
                train_data['close'].values[-500:],
                test_data['close'].values[:min(30, len(test_data))]
            )
            if sarima_model:
                fold_models['SARIMA'] = sarima_model
            
            # 5. SARIMAX (with basic exog variables)
            try:
                exog_train = train_data[['returns', 'volatility_20']].fillna(0).values[-500:]
                exog_test = test_data[['returns', 'volatility_20']].fillna(0).values[:min(30, len(test_data))]
                sarimax_model = self.train_sarimax_model(
                    train_data['close'].values[-500:],
                    test_data['close'].values[:min(30, len(test_data))],
                    exog_train, exog_test
                )
                if sarimax_model:
                    fold_models['SARIMAX'] = sarimax_model
            except:
                pass
            
            # Store results
            results[test_year] = {
                'models': fold_models,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'scaler': scaler
            }
            
            print(f"Trained {len(fold_models)} models for {test_year}")
        
        self.walk_forward_results = results
        print(f"\nWalk-forward validation completed. {len(results)} folds processed.")
        
        return results
    
    def evaluate_models_walk_forward(self):
        """Evaluate all models using walk-forward results"""
        print("\n" + "="*60)
        print("MODEL EVALUATION (Walk-Forward)")
        print("="*60)
        
        if not self.walk_forward_results:
            print("No walk-forward results available")
            return {}
        
        # Aggregate performance across folds
        model_performance = {}
        
        for fold, results in self.walk_forward_results.items():
            for model_name, model_data in results['models'].items():
                if model_name not in model_performance:
                    model_performance[model_name] = {
                        'fold_maes': [],
                        'fold_mapes': [],
                        'fold_overfitting_scores': [],
                        'avg_mae': 0,
                        'avg_mape': 0,
                        'avg_overfitting_score': 0,
                        'std_mae': 0,
                        'best_fold': None,
                        'is_overfitting_count': 0
                    }
                
                if 'val_mae' in model_data:
                    model_performance[model_name]['fold_maes'].append(model_data['val_mae'])
                    
                    # Add MAPE if available
                    if 'val_mape' in model_data:
                        model_performance[model_name]['fold_mapes'].append(model_data['val_mape'])
                    
                    # Add overfitting score if available
                    if 'overfitting_score' in model_data:
                        model_performance[model_name]['fold_overfitting_scores'].append(model_data['overfitting_score'])
                        
                        if model_data.get('is_overfitting', False):
                            model_performance[model_name]['is_overfitting_count'] += 1
        
        # Calculate aggregate metrics
        for model_name in model_performance:
            maes = model_performance[model_name]['fold_maes']
            mapes = model_performance[model_name]['fold_mapes']
            overfitting_scores = model_performance[model_name]['fold_overfitting_scores']
            
            if maes:
                model_performance[model_name]['avg_mae'] = np.mean(maes)
                model_performance[model_name]['std_mae'] = np.std(maes)
                model_performance[model_name]['best_fold'] = min(maes)
                
            if mapes:
                model_performance[model_name]['avg_mape'] = np.mean(mapes)
                
            if overfitting_scores:
                model_performance[model_name]['avg_overfitting_score'] = np.mean(overfitting_scores)
        
        # Display results
        print("\nModel Performance Summary:")
        print("-" * 80)
        for model_name, perf in model_performance.items():
            if perf['fold_maes']:
                print(f"{model_name}:")
                print(f"  Average MAE: {perf['avg_mae']:.2f}")
                print(f"  Average MAPE: {perf.get('avg_mape', 'N/A'):.2f}%" if isinstance(perf.get('avg_mape', 0), (int, float)) else f"  Average MAPE: N/A")
                print(f"  Std MAE: {perf['std_mae']:.2f}")
                print(f"  Best MAE: {perf['best_fold']:.2f}")
                print(f"  Overfitting Score: {perf.get('avg_overfitting_score', 'N/A'):.2f}" if perf.get('avg_overfitting_score', 0) != 0 else f"  Overfitting Score: N/A")
                print(f"  Overfitting Count: {perf['is_overfitting_count']}/{len(perf['fold_maes'])}")
                print(f"  Folds: {len(perf['fold_maes'])}")
                print()
        
        # Select best model (lowest MAE with reasonable overfitting)
        valid_models = [name for name, perf in model_performance.items() if perf['fold_maes']]
        if valid_models:
            # Filter models with reasonable overfitting scores (< 3.0 average)
            reasonable_models = [
                name for name in valid_models 
                if model_performance[name].get('avg_overfitting_score', 0) < 3.0
            ]
            
            # If no reasonable models, use all valid models
            if not reasonable_models:
                reasonable_models = valid_models
                print("⚠️  Warning: All models show high overfitting scores")
            
            best_model = min(reasonable_models, key=lambda x: model_performance[x]['avg_mae'])
            print(f"🏆 Best Model: {best_model} (MAE: {model_performance[best_model]['avg_mae']:.2f}, MAPE: {model_performance[best_model].get('avg_mape', 'N/A'):.2f}%)")
            
            # Check for overfitting warning
            if model_performance[best_model].get('avg_overfitting_score', 0) > 2.0:
                print(f"⚠️  Warning: Best model shows signs of overfitting (score: {model_performance[best_model]['avg_overfitting_score']:.2f})")
            
            self.best_model_name = best_model
        else:
            print("No valid models found")
            self.best_model_name = None
        
        self.model_performance = model_performance
        
        return model_performance
    
    def train_final_model(self):
        """Train final model on all available data"""
        print(f"\n" + "="*60)
        print(f"TRAINING FINAL MODEL ({getattr(self, 'best_model_name', 'Unknown')})")
        print("="*60)
        
        if not hasattr(self, 'best_model_name') or self.best_model_name is None:
            print("No best model selected. Using XGBoost as default.")
            self.best_model_name = 'XGBoost'
        
        # Prepare full dataset
        feature_cols = self.prepare_features_for_models()
        clean_df = self.df.dropna(subset=feature_cols + ['close'])
        
        # Features and targets
        X_full = clean_df[feature_cols].values
        y_full = self.prepare_targets_for_training(
            clean_df['close'], 
            start_idx=0, 
            forecast_horizon=self.forecast_horizon
        )
        
        # Align
        X_full = X_full[:len(y_full)]
        
        # Scale
        self.final_scaler = RobustScaler()
        X_full_scaled = self.final_scaler.fit_transform(X_full)
        
        # Split for validation
        split_idx = int(0.9 * len(X_full_scaled))
        X_train = X_full_scaled[:split_idx]
        y_train = y_full[:split_idx]
        X_val = X_full_scaled[split_idx:]
        y_val = y_full[split_idx:]
        
        # Train best model
        if self.best_model_name == 'XGBoost':
            self.final_model = self.train_xgboost_model(X_train, y_train, X_val, y_val)
        elif self.best_model_name == 'LightGBM':
            self.final_model = self.train_lightgbm_model(X_train, y_train, X_val, y_val)
        elif self.best_model_name == 'ARIMA':
            self.final_model = self.train_arima_model(clean_df['close'].values[-1000:], clean_df['close'].values[-100:])
        elif self.best_model_name == 'SARIMA':
            self.final_model = self.train_sarima_model(clean_df['close'].values[-1000:], clean_df['close'].values[-100:])
        elif self.best_model_name == 'SARIMAX':
            exog_train = clean_df[['returns', 'volatility_20']].fillna(0).values[-1000:]
            exog_val = clean_df[['returns', 'volatility_20']].fillna(0).values[-100:]
            self.final_model = self.train_sarimax_model(
                clean_df['close'].values[-1000:], clean_df['close'].values[-100:], exog_train, exog_val
            )
        else:
            # Default to XGBoost
            self.final_model = self.train_xgboost_model(X_train, y_train, X_val, y_val)
        
        print(f"✓ Final {self.best_model_name} model trained on {len(X_full)} samples")
        return self.final_model
    
    def predict_next_10_days(self, reference_price, features=None):
        """Predict next 10 days prices"""
        if not hasattr(self, 'final_model') or self.final_model is None:
            print("No final model available")
            return None
        
        model_type = self.final_model['type']
        
        try:
            if model_type == 'multi_model':
                # XGBoost/LightGBM approach
                if features is None:
                    # Use last known features
                    feature_cols = self.prepare_features_for_models()
                    features = self.df[feature_cols].iloc[-1:].values
                
                features_scaled = self.final_scaler.transform(features)
                
                predictions = []
                for day in range(self.forecast_horizon):
                    model = self.final_model['models'][f'day_{day+1}']['model']
                    pred = model.predict(features_scaled)[0]
                    predictions.append(pred)
                
                return np.array(predictions)
            
            elif model_type == 'arima' or model_type == 'sarima':
                # Time series models
                model = self.final_model['model']
                forecast = model.forecast(steps=self.forecast_horizon)
                return np.array(forecast)
            
            elif model_type == 'sarimax':
                # SARIMAX model
                model = self.final_model['model']
                # Use simple exog values for forecast
                exog_forecast = np.zeros((self.forecast_horizon, 2))  # returns, volatility
                forecast = model.forecast(steps=self.forecast_horizon, exog=exog_forecast)
                return np.array(forecast)
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None
    
    def visualize_testing_results(self, results_df):
        """Create comprehensive visualizations of testing results"""
        print("\nCreating visualization plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # Create subplots
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Price forecasts vs actual timeline
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(results_df['date'], results_df['yesterday_price'], 'b-', alpha=0.8, linewidth=2, label='Actual Prices')
        
        # Highlight downturns
        downturn_dates = results_df[results_df['downturn_detected'] == True]['date']
        downturn_prices = results_df[results_df['downturn_detected'] == True]['yesterday_price']
        ax1.scatter(downturn_dates, downturn_prices, color='red', s=50, alpha=0.7, label='Predicted Downturns')
        
        ax1.set_title('ASI Price Timeline with Predicted Downturns', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ASI Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Downturn detection distribution
        ax2 = fig.add_subplot(gs[0, 2])
        downturn_counts = results_df['downturn_detected'].value_counts()
        colors = ['lightgreen', 'lightcoral']
        ax2.pie(downturn_counts.values, labels=['No Downturn', 'Downturn Detected'], 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Downturn Detection Distribution', fontsize=12, fontweight='bold')
        
        # 3. Severity analysis
        ax3 = fig.add_subplot(gs[0, 3])
        if results_df['downturn_detected'].sum() > 0:
            severity_data = results_df[results_df['downturn_detected']]['predicted_severity_beyond_5pct'] * 100
            ax3.hist(severity_data, bins=15, alpha=0.7, color='orange', edgecolor='black')
            ax3.axvline(severity_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {severity_data.mean():.1f}%')
            ax3.set_title('Additional Severity Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Additional Severity (%)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Days to trough analysis
        ax4 = fig.add_subplot(gs[1, :2])
        if results_df['downturn_detected'].sum() > 0:
            trough_data = results_df[results_df['downturn_detected']]['predicted_days_to_trough']
            ax4.hist(trough_data, bins=10, alpha=0.7, color='purple', edgecolor='black')
            ax4.axvline(trough_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {trough_data.mean():.1f} days')
            ax4.set_title('Days to Trough Distribution', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Days to Trough')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Model performance comparison (if available)
        ax5 = fig.add_subplot(gs[1, 2:])
        if hasattr(self, 'model_performance') and self.model_performance:
            model_names = []
            mae_scores = []
            mape_scores = []
            
            for model_name, perf in self.model_performance.items():
                if perf.get('fold_maes'):
                    model_names.append(model_name)
                    mae_scores.append(perf['avg_mae'])
                    mape_scores.append(perf.get('avg_mape', 0))
            
            if model_names:
                x = np.arange(len(model_names))
                width = 0.35
                
                bars1 = ax5.bar(x - width/2, mae_scores, width, label='MAE', alpha=0.8, color='skyblue')
                ax5_twin = ax5.twinx()
                bars2 = ax5_twin.bar(x + width/2, mape_scores, width, label='MAPE (%)', alpha=0.8, color='lightcoral')
                
                ax5.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
                ax5.set_xlabel('Models')
                ax5.set_ylabel('MAE', color='blue')
                ax5_twin.set_ylabel('MAPE (%)', color='red')
                ax5.set_xticks(x)
                ax5.set_xticklabels(model_names, rotation=45)
                
                # Add value labels on bars
                for bar, score in zip(bars1, mae_scores):
                    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_scores)*0.01, 
                            f'{score:.0f}', ha='center', va='bottom', fontsize=9)
                
                for bar, score in zip(bars2, mape_scores):
                    if score > 0:
                        ax5_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mape_scores)*0.01, 
                                f'{score:.1f}%', ha='center', va='bottom', fontsize=9)
                
                ax5.grid(True, alpha=0.3)
        
        # 6. Forecasting accuracy over time
        ax6 = fig.add_subplot(gs[2, :])
        if 'forecast_day_1' in results_df.columns:
            # Calculate forecast errors for different horizons
            forecast_errors = {}
            for day in range(1, min(6, self.forecast_horizon + 1)):  # Show first 5 days
                if f'forecast_day_{day}' in results_df.columns:
                    forecast_col = f'forecast_day_{day}'
                    # Calculate error as percentage
                    errors = abs(results_df[forecast_col] - results_df['yesterday_price']) / results_df['yesterday_price'] * 100
                    forecast_errors[f'Day {day}'] = errors.dropna()
            
            if forecast_errors:
                box_data = [forecast_errors[key] for key in forecast_errors.keys()]
                ax6.boxplot(box_data, labels=list(forecast_errors.keys()))
                ax6.set_title('Forecast Error Distribution by Horizon', fontsize=12, fontweight='bold')
                ax6.set_ylabel('Forecast Error (%)')
                ax6.grid(True, alpha=0.3)
        
        # 7. Summary statistics
        ax7 = fig.add_subplot(gs[3, :])
        summary_text = "ASI Forecasting System - Testing Results Summary\n\n"
        summary_text += f"Total Predictions: {len(results_df)}\n"
        summary_text += f"Downturns Detected: {results_df['downturn_detected'].sum()}\n"
        summary_text += f"Detection Rate: {results_df['downturn_detected'].mean()*100:.1f}%\n\n"
        
        if results_df['downturn_detected'].sum() > 0:
            downturn_subset = results_df[results_df['downturn_detected']]
            summary_text += f"Average Additional Severity: {downturn_subset['predicted_severity_beyond_5pct'].mean()*100:.2f}%\n"
            summary_text += f"Average Days to Trough: {downturn_subset['predicted_days_to_trough'].mean():.1f}\n"
            recovery_rate = downturn_subset['predicted_will_recover'].mean()*100
            summary_text += f"Predicted Recovery Rate: {recovery_rate:.1f}%\n\n"
        
        if hasattr(self, 'best_model_name'):
            summary_text += f"Best Model: {self.best_model_name}\n"
        
        if hasattr(self, 'historical_stats') and self.historical_stats:
            summary_text += f"\nHistorical Context:\n"
            summary_text += f"Historical Avg Additional Severity: {self.historical_stats['avg_additional_severity']*100:.1f}%\n"
            summary_text += f"Historical Avg Days to Trough: {self.historical_stats['avg_days_to_trough']:.1f}\n"
            summary_text += f"Historical Recovery Rate: {self.historical_stats['recovery_rate']*100:.1f}%\n"
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)
        ax7.axis('off')
        
        plt.suptitle(f'ASI Forecasting System - Comprehensive Analysis Results', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save plot
        plt.savefig('./asi_forecasting_analysis_results.png', dpi=150, bbox_inches='tight')
        print("✓ Comprehensive analysis plot saved to 'asi_forecasting_analysis_results.png'")
        
        plt.show()
    
    def test_individual_models_on_production(self):
        """Test each model individually on production data"""
        print("\n" + "="*80)
        print("INDIVIDUAL MODEL TESTING ON PRODUCTION DATA")
        print("="*80)
        
        if not hasattr(self, 'walk_forward_results') or not self.walk_forward_results:
            print("No trained models available for individual testing")
            return None
        
        # Get the latest fold models
        latest_fold = max(self.walk_forward_results.keys())
        latest_models = self.walk_forward_results[latest_fold]['models']
        latest_scaler = self.walk_forward_results[latest_fold]['scaler']
        
        print(f"Using models from fold {latest_fold}")
        print(f"Available models: {list(latest_models.keys())}")
        
        individual_results = {}
        
        for model_name, model_data in latest_models.items():
            print(f"\nTesting {model_name} model...")
            
            model_predictions = []
            
            for i in range(len(self.test_df)):
                test_date = self.test_df.iloc[i]['date']
                yesterday_price = self.test_df.iloc[i]['close']
                
                try:
                    # Get prediction based on model type
                    if model_data['type'] == 'multi_model':
                        # XGBoost/LightGBM
                        feature_cols = self.prepare_features_for_models()
                        features = self.df[feature_cols].iloc[-1:].values
                        features_scaled = latest_scaler.transform(features)
                        
                        predictions = []
                        for day in range(self.forecast_horizon):
                            model = model_data['models'][f'day_{day+1}']['model']
                            pred = model.predict(features_scaled)[0]
                            predictions.append(pred)
                        
                        forecasted_prices = np.array(predictions)
                        
                    elif model_data['type'] in ['arima', 'sarima', 'sarimax']:
                        # Time series models
                        model = model_data['model']
                        
                        if model_data['type'] == 'sarimax':
                            # Use simple exog values for forecast
                            exog_forecast = np.zeros((self.forecast_horizon, 2))
                            forecasted_prices = model.forecast(steps=self.forecast_horizon, exog=exog_forecast)
                        else:
                            forecasted_prices = model.forecast(steps=self.forecast_horizon)
                        
                        forecasted_prices = np.array(forecasted_prices)
                    
                    # Analyze downturn
                    analysis = self.detect_downturns_and_analyze(yesterday_price, forecasted_prices)
                    
                    model_predictions.append({
                        'date': test_date,
                        'yesterday_price': yesterday_price,
                        'downturn_detected': analysis['downturn_detected'],
                        'downturn_day': analysis['downturn_day'],
                        'severity_beyond_5pct': analysis['severity_beyond_5pct'],
                        'days_to_trough': analysis['days_to_trough'],
                        'forecasted_prices': forecasted_prices
                    })
                    
                except Exception as e:
                    print(f"  Error predicting {test_date}: {e}")
                    model_predictions.append({
                        'date': test_date,
                        'yesterday_price': yesterday_price,
                        'downturn_detected': False,
                        'error': str(e)
                    })
            
            # Store results for this model
            individual_results[model_name] = model_predictions
            
            # Calculate summary statistics
            successful_predictions = [p for p in model_predictions if 'error' not in p]
            downturn_count = sum(1 for p in successful_predictions if p['downturn_detected'])
            
            print(f"  ✓ {model_name} Results:")
            print(f"    Successful predictions: {len(successful_predictions)}/{len(model_predictions)}")
            print(f"    Downturns detected: {downturn_count}")
            print(f"    Detection rate: {downturn_count/len(successful_predictions)*100:.1f}%")
            
            if downturn_count > 0:
                avg_severity = np.mean([p['severity_beyond_5pct'] for p in successful_predictions if p['downturn_detected']]) * 100
                avg_days = np.mean([p['days_to_trough'] for p in successful_predictions if p['downturn_detected']])
                print(f"    Average additional severity: {avg_severity:.2f}%")
                print(f"    Average days to trough: {avg_days:.1f}")
        
        # Save individual results
        individual_results_summary = {}
        for model_name, predictions in individual_results.items():
            successful = [p for p in predictions if 'error' not in p]
            downturn_count = sum(1 for p in successful if p['downturn_detected'])
            
            individual_results_summary[model_name] = {
                'total_predictions': len(predictions),
                'successful_predictions': len(successful),
                'downturns_detected': downturn_count,
                'detection_rate': downturn_count/len(successful) if successful else 0,
                'avg_severity': np.mean([p['severity_beyond_5pct'] for p in successful if p['downturn_detected']]) if downturn_count > 0 else 0,
                'avg_days_to_trough': np.mean([p['days_to_trough'] for p in successful if p['downturn_detected']]) if downturn_count > 0 else 0
            }
        
        # Save to JSON
        with open('./individual_model_results.json', 'w') as f:
            json.dump({
                'summary': individual_results_summary,
                'detailed_predictions': {k: [{key: val for key, val in pred.items() if key != 'forecasted_prices'} 
                                           for pred in v] for k, v in individual_results.items()}
            }, f, indent=2, default=str)
        
        print(f"\n✓ Individual model testing completed!")
        print(f"✓ Summary saved to 'individual_model_results.json'")
        
        return individual_results_summary
        
    
    def detect_downturns_and_analyze(self, yesterday_price, forecasted_prices):
        """Detect downturns and analyze severity/duration"""
        if forecasted_prices is None:
            return {
                'downturn_detected': False,
                'downturn_day': None,
                'severity_beyond_5pct': 0,
                'days_to_trough': 0,
                'will_recover': False,
                'recovery_days': None
            }
        
        # Check for -5% downturn
        downturn_detected = False
        downturn_day = None
        
        for day, price in enumerate(forecasted_prices):
            drop_pct = (price - yesterday_price) / yesterday_price
            if drop_pct <= -0.05:
                downturn_detected = True
                downturn_day = day + 1  # 1-indexed
                break
        
        analysis = {
            'downturn_detected': downturn_detected,
            'downturn_day': downturn_day,
            'severity_beyond_5pct': 0,
            'days_to_trough': 0,
            'will_recover': False,
            'recovery_days': None
        }
        
        if downturn_detected:
            # Find trough (minimum price)
            trough_idx = np.argmin(forecasted_prices)
            trough_price = forecasted_prices[trough_idx]
            
            # Calculate additional severity beyond -5%
            total_drop = (trough_price - yesterday_price) / yesterday_price
            analysis['severity_beyond_5pct'] = max(0, abs(total_drop) - 0.05)  # Beyond initial -5%
            analysis['days_to_trough'] = trough_idx + 1
            
            # Check for recovery (return to 100% of original price)
            recovery_threshold = yesterday_price
            for day in range(trough_idx + 1, len(forecasted_prices)):
                if forecasted_prices[day] >= recovery_threshold:
                    analysis['will_recover'] = True
                    analysis['recovery_days'] = day + 1
                    break
        
        return analysis
    
    def calculate_historical_downturn_stats(self):
        """Calculate historical downturn statistics from training data"""
        print("\nCalculating historical downturn statistics...")
        
        downturns = []
        threshold = -0.05
        lookforward = 30
        
        for i in range(len(self.df) - lookforward):
            current_price = self.df.iloc[i]['close']
            
            # Check for downturn in next 30 days
            future_prices = self.df.iloc[i+1:i+lookforward+1]['close'].values
            
            downturn_occurred = False
            for j, future_price in enumerate(future_prices):
                drop = (future_price - current_price) / current_price
                if drop <= threshold:
                    downturn_occurred = True
                    
                    # Find trough
                    remaining_prices = future_prices[j:]
                    if len(remaining_prices) > 0:
                        trough_idx = j + np.argmin(remaining_prices)
                        trough_price = future_prices[trough_idx]
                        
                        # Calculate metrics
                        total_drop = (trough_price - current_price) / current_price
                        additional_severity = abs(total_drop) - 0.05
                        days_to_trough = trough_idx + 1
                        
                        # Check recovery
                        recovered = False
                        recovery_days = None
                        for k in range(trough_idx + 1, len(future_prices)):
                            if future_prices[k] >= current_price:
                                recovered = True
                                recovery_days = k + 1
                                break
                        
                        downturns.append({
                            'additional_severity': additional_severity,
                            'days_to_trough': days_to_trough,
                            'recovered': recovered,
                            'recovery_days': recovery_days
                        })
                    break
        
        if downturns:
            df_downturns = pd.DataFrame(downturns)
            
            # Calculate statistics
            self.historical_stats = {
                'avg_additional_severity': df_downturns['additional_severity'].mean(),
                'avg_days_to_trough': df_downturns['days_to_trough'].mean(),
                'recovery_rate': df_downturns['recovered'].mean(),
                'avg_recovery_days': df_downturns[df_downturns['recovered']]['recovery_days'].mean() if df_downturns['recovered'].any() else None
            }
            
            print(f"Historical downturn analysis based on {len(downturns)} events:")
            print(f"  Average additional severity: {self.historical_stats['avg_additional_severity']:.3f} ({self.historical_stats['avg_additional_severity']*100:.1f}%)")
            print(f"  Average days to trough: {self.historical_stats['avg_days_to_trough']:.1f}")
            print(f"  Recovery rate: {self.historical_stats['recovery_rate']*100:.1f}%")
            if self.historical_stats['avg_recovery_days']:
                print(f"  Average recovery days: {self.historical_stats['avg_recovery_days']:.1f}")
        else:
            self.historical_stats = None
            print("No historical downturns found for analysis")
        
        return self.historical_stats
    
    def test_on_production_data(self):
        """Test the model on production data and generate comprehensive results"""
        print("\n" + "="*80)
        print("PRODUCTION TESTING")
        print("="*80)
        
        if not hasattr(self, 'final_model') or self.final_model is None:
            print("No final model available. Train model first.")
            return None
        
        results = []
        
        # Calculate historical stats
        historical_stats = self.calculate_historical_downturn_stats()
        
        for i in range(len(self.test_df)):
            test_date = self.test_df.iloc[i]['date']
            yesterday_price = self.test_df.iloc[i]['close']
            
            print(f"Forecasting for {test_date.strftime('%Y-%m-%d')} (yesterday price: {yesterday_price:.2f})")
            
            # Predict next 10 days
            forecasted_prices = self.predict_next_10_days(yesterday_price)
            
            if forecasted_prices is not None:
                # Downturn analysis
                analysis = self.detect_downturns_and_analyze(yesterday_price, forecasted_prices)
                
                # Create result row
                result = {
                    'date': test_date,
                    'yesterday_price': yesterday_price,
                    'downturn_detected': analysis['downturn_detected'],
                    'downturn_day': analysis['downturn_day'],
                    'predicted_severity_beyond_5pct': analysis['severity_beyond_5pct'],
                    'predicted_days_to_trough': analysis['days_to_trough'],
                    'predicted_will_recover': analysis['will_recover'],
                    'predicted_recovery_days': analysis['recovery_days']
                }
                
                # Add individual day forecasts
                for day in range(self.forecast_horizon):
                    result[f'forecast_day_{day+1}'] = forecasted_prices[day]
                    result[f'day_{day+1}_drop_pct'] = (forecasted_prices[day] - yesterday_price) / yesterday_price * 100
                
                # Add historical context
                if historical_stats:
                    result['historical_avg_additional_severity_pct'] = historical_stats['avg_additional_severity'] * 100
                    result['historical_avg_days_to_trough'] = historical_stats['avg_days_to_trough']
                    result['historical_recovery_rate_pct'] = historical_stats['recovery_rate'] * 100
                    if historical_stats['avg_recovery_days']:
                        result['historical_avg_recovery_days'] = historical_stats['avg_recovery_days']
                
            else:
                # Failed prediction
                result = {
                    'date': test_date,
                    'yesterday_price': yesterday_price,
                    'downturn_detected': False,
                    'error': 'Prediction failed'
                }
                
                # Add empty forecast columns
                for day in range(self.forecast_horizon):
                    result[f'forecast_day_{day+1}'] = None
                    result[f'day_{day+1}_drop_pct'] = None
            
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save to CSV
        output_file = './asi_production_test_results.csv'
        results_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Production testing completed!")
        print(f"✓ Results saved to: {output_file}")
        print(f"✓ Total predictions: {len(results_df)}")
        print(f"✓ Downturns detected: {results_df['downturn_detected'].sum()}")
        
        # Summary statistics
        if results_df['downturn_detected'].sum() > 0:
            downturn_subset = results_df[results_df['downturn_detected']]
            print("\nDownturn Analysis Summary:")
            print(f"  Average predicted additional severity: {downturn_subset['predicted_severity_beyond_5pct'].mean()*100:.2f}%")
            print(f"  Average predicted days to trough: {downturn_subset['predicted_days_to_trough'].mean():.1f}")
            print(f"  Recovery prediction rate: {downturn_subset['predicted_will_recover'].mean()*100:.1f}%")
        
        # Create visualizations
        self.visualize_testing_results(results_df)
        
        # Test individual models
        self.test_individual_models_on_production()
        
        return results_df
    
    def create_comprehensive_summary(self):
        """Create comprehensive analysis summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*80)
        
        summary = {
            'system_info': {
                'training_data_range': f"{self.df['date'].min()} to {self.df['date'].max()}",
                'testing_data_range': f"{self.test_df['date'].min()} to {self.test_df['date'].max()}",
                'training_samples': len(self.df),
                'testing_samples': len(self.test_df),
                'features_used': len(self.feature_names) if hasattr(self, 'feature_names') else 0,
                'forecast_horizon': self.forecast_horizon
            },
            'model_performance': getattr(self, 'model_performance', {}),
            'best_model': getattr(self, 'best_model_name', 'Unknown'),
            'historical_statistics': getattr(self, 'historical_stats', {}),
            'walk_forward_folds': len(getattr(self, 'walk_forward_results', {}))
        }
        
        # Save summary
        with open('./asi_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("✓ Comprehensive summary saved to 'asi_analysis_summary.json'")
        
        # Display key insights
        print("\nKey Insights:")
        print(f"• Best performing model: {summary['best_model']}")
        print(f"• Walk-forward validation folds: {summary['walk_forward_folds']}")
        print(f"• Features engineered: {summary['system_info']['features_used']}")
        
        if summary['historical_statistics']:
            stats = summary['historical_statistics']
            print(f"• Historical average additional drop: {stats.get('avg_additional_severity', 0)*100:.1f}%")
            print(f"• Historical average days to trough: {stats.get('avg_days_to_trough', 0):.1f}")
            print(f"• Historical recovery rate: {stats.get('recovery_rate', 0)*100:.1f}%")
        
        return summary
    
    def save_models(self):
        """Save trained models (with memory optimization)"""
        print("\nSaving models...")
        
        os.makedirs('./asi_minus_five_models', exist_ok=True)
        
        # Save final model
        if hasattr(self, 'final_model'):
            with open('./asi_minus_five_models/final_model.pkl', 'wb') as f:
                pickle.dump({
                    'model': self.final_model,
                    'scaler': getattr(self, 'final_scaler', None),
                    'feature_names': getattr(self, 'feature_names', []),
                    'model_name': getattr(self, 'best_model_name', 'Unknown'),
                    'forecast_horizon': self.forecast_horizon,
                    'historical_stats': getattr(self, 'historical_stats', {})
                }, f)
        
        # Save model performance summary (instead of full walk_forward_results to avoid memory error)
        if hasattr(self, 'model_performance'):
            with open('./asi_minus_five_models/model_performance_summary.json', 'w') as f:
                json.dump(self.model_performance, f, indent=2, default=str)
        
        # Save walk-forward summary (without large model objects)
        if hasattr(self, 'walk_forward_results'):
            summary_results = {}
            for fold, results in self.walk_forward_results.items():
                summary_results[fold] = {
                    'train_size': results['train_size'],
                    'test_size': results['test_size'],
                    'models_trained': list(results['models'].keys()),
                    'model_performance': {
                        name: {
                            'val_mae': model_data.get('val_mae'),
                            'val_mape': model_data.get('val_mape'),
                            'train_mae': model_data.get('train_mae'),
                            'train_mape': model_data.get('train_mape'),
                            'overfitting_score': model_data.get('overfitting_score'),
                            'is_overfitting': model_data.get('is_overfitting'),
                            'type': model_data.get('type')
                        }
                        for name, model_data in results['models'].items()
                    }
                }
            
            with open('./asi_minus_five_models/walk_forward_summary.json', 'w') as f:
                json.dump(summary_results, f, indent=2, default=str)
        
        print("✓ Models saved to './asi_minus_five_models/'")
        print("✓ Memory optimized - saved summaries instead of full model objects")

def main():
    """Main execution function"""
    print("Starting Comprehensive ASI Forecasting System...")
    
    # Initialize system
    system = ComprehensiveASIForecastingSystem(
        training_csv='ASI_close_prices_last_15_years_2025-06-05.csv',
        testing_csv='prod testing.csv'
    )
    
    # Step 1: Create enhanced features
    system.create_enhanced_features()
    
    # Step 2: Walk-forward validation
    system.walk_forward_validation()
    
    # Step 3: Evaluate models
    system.evaluate_models_walk_forward()
    
    # Step 4: Train final model on all data
    system.train_final_model()
    
    # Step 5: Test on production data
    system.test_on_production_data()
    
    # Step 6: Create comprehensive summary
    system.create_comprehensive_summary()
    
    # Step 7: Save models
    system.save_models()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ASI FORECASTING SYSTEM COMPLETE!")
    print("="*80)
    print("✓ 5 forecasting models tested (XGBoost, LightGBM, ARIMA, SARIMA, SARIMAX)")
    print("✓  model removed for improved performance")
    print("✓ Walk-forward validation completed")
    print("✓ MAPE evaluation and overfitting checks added")
    print("✓ Best model selected and trained")
    print("✓ Production testing completed")
    print("✓ Individual model testing completed")
    print("✓ Comprehensive visualizations created")
    print("✓ Results exported with memory optimization")
    print("\nOutputs Generated:")
    print("• asi_production_test_results.csv - Main results file")
    print("• asi_forecasting_analysis_results.png - Comprehensive visualizations")
    print("• individual_model_results.json - Individual model performance")
    print("• asi_analysis_summary.json - System summary")
    print("• ./asi_minus_five_models/ - Trained models directory")
    print("="*80)

if __name__ == "__main__":
    main()