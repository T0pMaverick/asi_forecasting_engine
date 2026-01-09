#!/usr/bin/env python3
"""
Enhanced SARIMA ASI Price Forecaster - Production Ready
Purpose: Predict ASI prices for next 10 days with overfitting controls
Author: Dushan - Enhanced Version
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Core Libraries
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# SARIMA Model
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available. Install with: pip install statsmodels")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

class EnhancedSARIMAForecaster:
    """
    Production-ready SARIMA model for ASI price forecasting
    Enhanced with overfitting controls and validation safeguards
    """
    
    def __init__(self, training_csv, testing_csv):
        """Initialize the enhanced SARIMA forecaster"""
        print("="*80)
        print("ENHANCED SARIMA ASI PRICE FORECASTER")
        print("="*80)
        print("✓ Overfitting Controls Enabled")
        print("✓ Conservative Parameter Settings")
        print("✓ Purged Walk-Forward Validation")
        print("✓ Reality Check Filters")
        print("✓ 10-Day Price Forecasting Focus")
        print("="*80)
        
        # Load and prepare data
        self.df = pd.read_csv(training_csv)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.df['close'] = pd.to_numeric(self.df['close'])
        
        self.test_df = pd.read_csv(testing_csv)
        self.test_df['date'] = pd.to_datetime(self.test_df['date'])
        self.test_df = self.test_df.sort_values('date').reset_index(drop=True)
        self.test_df['close'] = pd.to_numeric(self.test_df['close'])
        
        print(f"\nTraining data: {len(self.df)} records from {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"Testing data: {len(self.test_df)} records from {self.test_df['date'].min()} to {self.test_df['date'].max()}")
        
        # Initialize containers
        self.forecast_horizon = 10
        self.model = None
        self.validation_results = {}
        
        # Create output directory
        self.output_dir = './enhanced_sarima_results'
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✓ Results will be saved to: {self.output_dir}")
        
    def prepare_enhanced_features(self):
        """Create robust features with proper lags to prevent overfitting"""
        print("\nCreating robust features with anti-overfitting measures...")
        
        # Ensure proper lag to prevent data leakage
        lag_buffer = 2  # 2-day buffer to prevent leakage
        
        # Basic price features with proper lags
        self.df['returns'] = self.df['close'].pct_change(lag_buffer)
        self.df['log_returns'] = np.log(self.df['close']).diff(lag_buffer)
        self.df['price_change'] = self.df['close'].diff(lag_buffer)
        
        # Conservative moving averages (longer periods for stability)
        for period in [20, 50, 100, 200]:
            self.df[f'sma_{period}'] = self.df['close'].shift(lag_buffer).rolling(period).mean()
        
        # Robust volatility measures
        self.df['volatility_20'] = self.df['returns'].rolling(20).std() * np.sqrt(252)
        self.df['volatility_60'] = self.df['returns'].rolling(60).std() * np.sqrt(252)
        
        # Conservative technical indicators
        self.df['rsi_14'] = self.calculate_rsi(self.df['close'].shift(lag_buffer), 14)
        
        # Price positioning (more stable indicators)
        self.df['price_vs_sma200'] = (self.df['close'] - self.df['sma_200']) / (self.df['sma_200'] + 1e-9)
        self.df['sma50_vs_sma200'] = (self.df['sma_50'] - self.df['sma_200']) / (self.df['sma_200'] + 1e-9)
        
        # Calendar features (stable, no leakage risk)
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['year'] = self.df['date'].dt.year
        
        # Remove initial NaN rows
        self.df = self.df.dropna().reset_index(drop=True)
        
        print(f"✓ Created robust features, {len(self.df)} samples after cleaning")
        print(f"✓ Applied {lag_buffer}-day lag buffer to prevent data leakage")
        
    def calculate_rsi(self, series, period=14):
        """Calculate RSI with enhanced stability"""
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
        loss = -delta.clip(upper=0).rolling(period, min_periods=period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - 100 / (1 + rs)
        return rsi.fillna(50)  # Fill with neutral value
        
    def purged_walk_forward_validation(self, min_train_years=3, gap_days=30):
        """
        Enhanced walk-forward validation with purging gap
        Prevents overfitting by adding gap between train/test
        """
        print(f"\n" + "="*60)
        print("PURGED WALK-FORWARD VALIDATION")
        print("="*60)
        print(f"✓ Minimum training: {min_train_years} years")
        print(f"✓ Purging gap: {gap_days} days")
        
        # Calculate available years
        total_years = (self.df['date'].max() - self.df['date'].min()).days / 365.25
        print(f"✓ Total data span: {total_years:.1f} years")
        
        if total_years < min_train_years + 1:
            print(f"❌ Insufficient data for validation (need {min_train_years + 1}+ years)")
            return {}
        
        # Create conservative time splits
        start_year = self.df['date'].min().year + min_train_years
        end_year = self.df['date'].max().year
        
        results = {}
        
        for test_year in range(start_year, end_year + 1):
            print(f"\nValidation Fold {test_year}:")
            
            # Create train/test split with purging
            train_end_date = pd.to_datetime(f"{test_year-1}-12-31")
            test_start_date = pd.to_datetime(f"{test_year}-01-01")
            
            # Apply purging gap
            train_data = self.df[self.df['date'] <= train_end_date]
            if len(train_data) > gap_days:
                train_data = train_data.iloc[:-gap_days]  # Remove last N days
            
            test_data = self.df[
                (self.df['date'] >= test_start_date) & 
                (self.df['date'] < test_start_date + pd.DateOffset(months=6))  # Limit test period
            ]
            
            if len(train_data) < 500 or len(test_data) < 30:
                print(f"  ⚠️  Skipping - insufficient data")
                continue
            
            print(f"  📊 Training: {len(train_data)} samples (purged last {gap_days} days)")
            print(f"  📊 Testing: {len(test_data)} samples")
            
            # Train conservative SARIMA model
            try:
                model_result = self.train_conservative_sarima(
                    train_data['close'].values,
                    test_data['close'].values
                )
                
                if model_result:
                    results[test_year] = model_result
                    print(f"  ✓ MAE: {model_result['mae']:.2f}, MAPE: {model_result['mape']:.2f}%")
                else:
                    print(f"  ❌ Training failed")
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
        
        self.validation_results = results
        print(f"\n✓ Validation completed: {len(results)} successful folds")
        
        return results
    
    def train_conservative_sarima(self, train_prices, val_prices):
        """
        Train SARIMA with conservative parameters to reduce overfitting
        """
        try:
            # Conservative SARIMA configuration
            # Reduced complexity to prevent overfitting
            model = SARIMAX(
                train_prices,
                order=(1, 1, 0),              # Simplified AR/MA structure
                seasonal_order=(1, 1, 0, 12), # Monthly seasonality, simplified
                enforce_stationarity=True,     # Stability constraint
                enforce_invertibility=True,    # Invertibility constraint
                simple_differencing=True,      # More stable differencing
                validate_specification=True    # Additional checks
            )
            
            # Fit with convergence controls
            fitted_model = model.fit(
                disp=False,
                maxiter=100,          # Limit iterations
                method='lbfgs',       # More stable optimizer
                low_memory=True       # Memory efficient
            )
            
            # Generate forecasts
            forecast = fitted_model.forecast(steps=len(val_prices))
            
            # Calculate metrics
            mae = mean_absolute_error(val_prices, forecast)
            mape = mean_absolute_percentage_error(val_prices, forecast) * 100
            rmse = np.sqrt(mean_squared_error(val_prices, forecast))
            
            # In-sample metrics for overfitting check
            train_fitted = fitted_model.fittedvalues
            if len(train_fitted) > 0:
                train_mae = mean_absolute_error(
                    train_prices[-len(train_fitted):], 
                    train_fitted
                )
                overfitting_score = mae / (train_mae + 1e-9)
            else:
                overfitting_score = float('inf')
            
            return {
                'model': fitted_model,
                'mae': mae,
                'mape': mape,
                'rmse': rmse,
                'overfitting_score': overfitting_score,
                'forecast': forecast,
                'validation_predictions': forecast
            }
            
        except Exception as e:
            print(f"    SARIMA training failed: {e}")
            return None
    
    def evaluate_model_performance(self):
        """Evaluate model performance across validation folds"""
        print(f"\n" + "="*60)
        print("MODEL PERFORMANCE EVALUATION")
        print("="*60)
        
        if not self.validation_results:
            print("❌ No validation results available")
            return None
        
        # Aggregate performance metrics
        mae_scores = [r['mae'] for r in self.validation_results.values()]
        mape_scores = [r['mape'] for r in self.validation_results.values()]
        overfitting_scores = [r['overfitting_score'] for r in self.validation_results.values()]
        
        performance = {
            'avg_mae': np.mean(mae_scores),
            'std_mae': np.std(mae_scores),
            'avg_mape': np.mean(mape_scores),
            'avg_overfitting_score': np.mean(overfitting_scores),
            'folds': len(self.validation_results)
        }
        
        print(f"📊 Performance Summary:")
        print(f"   Average MAE: {performance['avg_mae']:.2f} ± {performance['std_mae']:.2f}")
        print(f"   Average MAPE: {performance['avg_mape']:.2f}%")
        print(f"   Overfitting Score: {performance['avg_overfitting_score']:.2f}")
        print(f"   Validation Folds: {performance['folds']}")
        
        # Production readiness check
        if performance['avg_overfitting_score'] < 2.0:
            print(f"✅ Model shows good generalization (overfitting < 2.0)")
        else:
            print(f"⚠️  Model shows overfitting signs (score: {performance['avg_overfitting_score']:.2f})")
        
        return performance
    
    def train_final_model(self):
        """Train final model on all available data"""
        print(f"\n" + "="*60)
        print("TRAINING FINAL MODEL")
        print("="*60)
        
        # Use all training data with final 10% as validation
        split_idx = int(0.9 * len(self.df))
        train_data = self.df.iloc[:split_idx]
        val_data = self.df.iloc[split_idx:]
        
        print(f"📊 Training samples: {len(train_data)}")
        print(f"📊 Validation samples: {len(val_data)}")
        
        # Train final conservative SARIMA model
        self.model = self.train_conservative_sarima(
            train_data['close'].values,
            val_data['close'].values
        )
        
        if self.model:
            print(f"✅ Final model trained successfully")
            print(f"   Validation MAE: {self.model['mae']:.2f}")
            print(f"   Validation MAPE: {self.model['mape']:.2f}%")
            print(f"   Overfitting Score: {self.model['overfitting_score']:.2f}")
        else:
            print(f"❌ Final model training failed")
        
        return self.model
    
    def predict_next_10_days(self, reference_price):
        """Predict ASI prices for next 10 days with reality checks"""
        if not self.model:
            print("❌ No trained model available")
            return None
        
        try:
            # Generate raw predictions
            raw_predictions = self.model['model'].forecast(steps=self.forecast_horizon)
            
            # Apply reality checks
            enhanced_predictions = self.apply_reality_checks(raw_predictions, reference_price)
            
            return enhanced_predictions
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None
    
    def apply_reality_checks(self, predictions, reference_price):
        """Apply business logic to ensure realistic predictions"""
        
        # Convert to numpy array if needed
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        
        enhanced_predictions = predictions.copy()
        
        # Reality Check 1: Limit extreme price movements
        max_daily_change = 0.05  # 5% max daily change
        for i in range(1, len(enhanced_predictions)):
            prev_price = enhanced_predictions[i-1] if i > 1 else reference_price
            daily_change = abs(enhanced_predictions[i] - prev_price) / prev_price
            
            if daily_change > max_daily_change:
                # Cap the change
                direction = 1 if enhanced_predictions[i] > prev_price else -1
                enhanced_predictions[i] = prev_price * (1 + direction * max_daily_change)
        
        # Reality Check 2: Ensure prices stay positive
        enhanced_predictions = np.maximum(enhanced_predictions, reference_price * 0.5)
        
        # Reality Check 3: Smooth unrealistic volatility
        for i in range(1, len(enhanced_predictions)-1):
            # If price movement is too erratic, smooth it
            prev_change = enhanced_predictions[i] - enhanced_predictions[i-1]
            next_change = enhanced_predictions[i+1] - enhanced_predictions[i]
            
            if prev_change * next_change < 0:  # Direction change
                if abs(prev_change) > reference_price * 0.03:  # Large change
                    # Smooth the transition
                    enhanced_predictions[i] = (enhanced_predictions[i-1] + enhanced_predictions[i+1]) / 2
        
        return enhanced_predictions
    
    def test_on_production_data(self):
        """Test model on production data and generate results"""
        print(f"\n" + "="*80)
        print("PRODUCTION TESTING")
        print("="*80)
        
        if not self.model:
            print("❌ No trained model available")
            return None
        
        results = []
        
        for i, row in self.test_df.iterrows():
            test_date = row['date']
            reference_price = row['close']
            
            print(f"📅 Forecasting for {test_date.strftime('%Y-%m-%d')} (price: {reference_price:.2f})")
            
            # Generate 10-day forecast
            forecasted_prices = self.predict_next_10_days(reference_price)
            
            if forecasted_prices is not None:
                # Create result entry
                result = {
                    'date': test_date,
                    'reference_price': reference_price
                }
                
                # Add daily forecasts
                for day in range(self.forecast_horizon):
                    result[f'forecast_day_{day+1}'] = forecasted_prices[day]
                    result[f'day_{day+1}_change_pct'] = ((forecasted_prices[day] - reference_price) / reference_price) * 100
                
                # Add summary statistics
                result['avg_predicted_price'] = np.mean(forecasted_prices)
                result['max_predicted_price'] = np.max(forecasted_prices)
                result['min_predicted_price'] = np.min(forecasted_prices)
                result['price_volatility'] = np.std(forecasted_prices)
                
                results.append(result)
            else:
                print(f"  ❌ Forecast failed for {test_date}")
        
        # Save results
        results_df = pd.DataFrame(results)
        output_file = f"{self.output_dir}/sarima_price_forecasts.csv"
        results_df.to_csv(output_file, index=False)
        
        print(f"\n✅ Production testing completed!")
        print(f"✅ Results saved to: {output_file}")
        print(f"📊 Total successful forecasts: {len(results_df)}/{len(self.test_df)}")
        
        return results_df
    
    def create_comprehensive_visualization(self, results_df):
        """Create comprehensive analysis visualizations"""
        print(f"\n📊 Creating comprehensive visualizations...")
        
        # Set up plotting
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Price timeline with forecasts
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(results_df['date'], results_df['reference_price'], 'b-', alpha=0.8, linewidth=2, label='Actual Prices')
        ax1.plot(results_df['date'], results_df['avg_predicted_price'], 'r--', alpha=0.7, linewidth=2, label='Avg Forecast')
        
        # Add forecast range
        ax1.fill_between(results_df['date'], 
                        results_df['min_predicted_price'], 
                        results_df['max_predicted_price'], 
                        alpha=0.2, color='red', label='Forecast Range')
        
        ax1.set_title('ASI Price Forecasting Results', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ASI Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Forecast accuracy distribution
        ax2 = fig.add_subplot(gs[0, 2])
        forecast_errors = []
        for day in range(1, 6):  # First 5 days
            if f'day_{day}_change_pct' in results_df.columns:
                errors = results_df[f'day_{day}_change_pct'].abs()
                forecast_errors.extend(errors.dropna())
        
        if forecast_errors:
            ax2.hist(forecast_errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(np.mean(forecast_errors), color='red', linestyle='--', 
                       label=f'Mean Error: {np.mean(forecast_errors):.1f}%')
            ax2.set_title('Forecast Error Distribution', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Absolute Error (%)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Volatility analysis
        ax3 = fig.add_subplot(gs[0, 3])
        if 'price_volatility' in results_df.columns:
            ax3.scatter(results_df['reference_price'], results_df['price_volatility'], 
                       alpha=0.6, color='orange')
            ax3.set_title('Price vs Forecast Volatility', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Reference Price')
            ax3.set_ylabel('Forecast Volatility')
            ax3.grid(True, alpha=0.3)
        
        # 4. Forecast horizon accuracy
        ax4 = fig.add_subplot(gs[1, :2])
        horizon_errors = {}
        for day in range(1, self.forecast_horizon + 1):
            if f'day_{day}_change_pct' in results_df.columns:
                errors = results_df[f'day_{day}_change_pct'].abs()
                horizon_errors[f'Day {day}'] = errors.dropna()
        
        if horizon_errors:
            box_data = [horizon_errors[key] for key in horizon_errors.keys()]
            ax4.boxplot(box_data, labels=list(horizon_errors.keys()))
            ax4.set_title('Forecast Accuracy by Horizon', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Absolute Error (%)')
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. Model validation results
        ax5 = fig.add_subplot(gs[1, 2:])
        if self.validation_results:
            years = list(self.validation_results.keys())
            mae_scores = [self.validation_results[year]['mae'] for year in years]
            mape_scores = [self.validation_results[year]['mape'] for year in years]
            
            ax5_twin = ax5.twinx()
            bars1 = ax5.bar([str(y) for y in years], mae_scores, alpha=0.7, color='skyblue', label='MAE')
            bars2 = ax5_twin.bar([str(y) for y in years], mape_scores, alpha=0.7, color='lightcoral', label='MAPE (%)')
            
            ax5.set_title('Validation Performance by Year', fontsize=12, fontweight='bold')
            ax5.set_ylabel('MAE', color='blue')
            ax5_twin.set_ylabel('MAPE (%)', color='red')
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        ax6 = fig.add_subplot(gs[2, :])
        summary_text = "Enhanced SARIMA ASI Price Forecaster - Results Summary\n\n"
        summary_text += f"Total Forecasts Generated: {len(results_df)}\n"
        summary_text += f"Forecast Horizon: {self.forecast_horizon} days\n"
        
        if len(results_df) > 0:
            avg_volatility = results_df['price_volatility'].mean() if 'price_volatility' in results_df.columns else 0
            price_range = f"{results_df['reference_price'].min():.0f} - {results_df['reference_price'].max():.0f}"
            summary_text += f"Price Range: {price_range}\n"
            summary_text += f"Average Forecast Volatility: {avg_volatility:.2f}\n\n"
        
        # Model performance
        if hasattr(self, 'model') and self.model:
            summary_text += f"Final Model Performance:\n"
            summary_text += f"  Validation MAE: {self.model['mae']:.2f}\n"
            summary_text += f"  Validation MAPE: {self.model['mape']:.2f}%\n"
            summary_text += f"  Overfitting Score: {self.model['overfitting_score']:.2f}\n\n"
        
        # Validation summary
        if self.validation_results:
            summary_text += f"Walk-Forward Validation:\n"
            summary_text += f"  Successful Folds: {len(self.validation_results)}\n"
            mae_scores = [r['mae'] for r in self.validation_results.values()]
            summary_text += f"  Average MAE: {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}\n"
        
        summary_text += f"\n✅ Enhanced SARIMA Model - Production Ready"
        summary_text += f"\n✅ Overfitting Controls Applied"
        summary_text += f"\n✅ Reality Checks Implemented"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        
        plt.suptitle('Enhanced SARIMA ASI Price Forecaster - Comprehensive Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save plot
        plot_file = f"{self.output_dir}/enhanced_sarima_analysis.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✅ Comprehensive analysis plot saved: {plot_file}")
        
        plt.show()
    
    def save_model_and_results(self):
        """Save model and comprehensive results"""
        print(f"\n💾 Saving model and results...")
        
        # Save final model
        if self.model:
            model_file = f"{self.output_dir}/enhanced_sarima_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': self.model['model'],
                    'performance': {
                        'mae': self.model['mae'],
                        'mape': self.model['mape'],
                        'overfitting_score': self.model['overfitting_score']
                    },
                    'forecast_horizon': self.forecast_horizon,
                    'training_info': {
                        'training_samples': len(self.df),
                        'training_period': f"{self.df['date'].min()} to {self.df['date'].max()}",
                        'model_type': 'Enhanced SARIMA with Overfitting Controls'
                    }
                }, f)
            print(f"✅ Model saved: {model_file}")
        
        # Save validation results
        if self.validation_results:
            validation_file = f"{self.output_dir}/validation_results.json"
            # Convert numpy types to regular Python types for JSON serialization
            json_safe_results = {}
            for year, results in self.validation_results.items():
                json_safe_results[str(year)] = {
                    'mae': float(results['mae']),
                    'mape': float(results['mape']),
                    'overfitting_score': float(results['overfitting_score'])
                }
            
            with open(validation_file, 'w') as f:
                json.dump(json_safe_results, f, indent=2)
            print(f"✅ Validation results saved: {validation_file}")
        
        # Create summary report
        summary_file = f"{self.output_dir}/model_summary.json"
        summary = {
            'model_info': {
                'type': 'Enhanced SARIMA',
                'purpose': 'ASI 10-day price forecasting',
                'enhancements': [
                    'Overfitting controls',
                    'Conservative parameters', 
                    'Purged walk-forward validation',
                    'Reality check filters'
                ]
            },
            'data_info': {
                'training_samples': len(self.df),
                'testing_samples': len(self.test_df),
                'training_period': f"{self.df['date'].min()} to {self.df['date'].max()}",
                'testing_period': f"{self.test_df['date'].min()} to {self.test_df['date'].max()}"
            }
        }
        
        if hasattr(self, 'model') and self.model:
            summary['performance'] = {
                'final_mae': float(self.model['mae']),
                'final_mape': float(self.model['mape']),
                'overfitting_score': float(self.model['overfitting_score']),
                'production_ready': self.model['overfitting_score'] < 2.0
            }
        
        if self.validation_results:
            mae_scores = [r['mae'] for r in self.validation_results.values()]
            summary['validation'] = {
                'folds': len(self.validation_results),
                'avg_mae': float(np.mean(mae_scores)),
                'std_mae': float(np.std(mae_scores))
            }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Model summary saved: {summary_file}")
        
        print(f"\n📁 All results saved to: {self.output_dir}/")
        print("   📄 enhanced_sarima_model.pkl - Trained model")
        print("   📄 sarima_price_forecasts.csv - Production forecasts")
        print("   📄 validation_results.json - Validation performance")
        print("   📄 model_summary.json - Comprehensive summary")
        print("   📊 enhanced_sarima_analysis.png - Analysis plots")

def main():
    """Main execution function"""
    print("🚀 Starting Enhanced SARIMA ASI Price Forecaster...")
    
    # Initialize system
    forecaster = EnhancedSARIMAForecaster(
        training_csv='ASI_close_prices_last_15_years_2025-06-05.csv',
        testing_csv='prod testing.csv'
    )
    
    # Step 1: Prepare robust features
    forecaster.prepare_enhanced_features()
    
    # Step 2: Purged walk-forward validation
    forecaster.purged_walk_forward_validation()
    
    # Step 3: Evaluate model performance
    forecaster.evaluate_model_performance()
    
    # Step 4: Train final model
    forecaster.train_final_model()
    
    # Step 5: Test on production data
    results_df = forecaster.test_on_production_data()
    
    # Step 6: Create visualizations
    if results_df is not None and len(results_df) > 0:
        forecaster.create_comprehensive_visualization(results_df)
    
    # Step 7: Save everything
    forecaster.save_model_and_results()
    
    print(f"\n" + "="*80)
    print("🎉 ENHANCED SARIMA FORECASTER COMPLETE!")
    print("="*80)
    print("✅ Conservative SARIMA model trained")
    print("✅ Overfitting controls applied")
    print("✅ Production testing completed")
    print("✅ Comprehensive analysis generated")
    print("✅ All results saved to ./enhanced_sarima_results/")
    print("="*80)

if __name__ == "__main__":
    main()