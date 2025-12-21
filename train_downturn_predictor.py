#!/usr/bin/env python3
"""
ASI Market Downturn Predictor - Model Training & Validation
Trains models to predict -5% drawdowns in 10 days (best combination from analysis)
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Try importing advanced models
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

import matplotlib.pyplot as plt
import seaborn as sns

class DownturnPredictorTrainer:
    """Train models to predict -5% drawdowns in 10 days"""
    
    def __init__(self, csv_file):
        """Load and prepare data"""
        print("="*80)
        print("ASI MARKET DOWNTURN PREDICTOR - MODEL TRAINING")
        print("Target: -5% drawdown in 10 days (Best combination from analysis)")
        print("="*80)
        print("\nLoading data...")
        
        self.df = pd.read_csv(csv_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.df['close'] = pd.to_numeric(self.df['close'])
        self.df['returns'] = self.df['close'].pct_change()
        
        print(f"Loaded {len(self.df)} records from {self.df['date'].min()} to {self.df['date'].max()}")
    
    def compute_features(self):
        """Compute technical indicators"""
        print("\nComputing technical indicators...")
        
        # Moving Averages
        self.df['sma_20'] = self.df['close'].rolling(20).mean()
        self.df['sma_50'] = self.df['close'].rolling(50).mean()
        self.df['sma_200'] = self.df['close'].rolling(200).mean()
        self.df['ema_12'] = self.df['close'].ewm(span=12, adjust=False).mean()
        self.df['ema_26'] = self.df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
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
        
        # Volatility
        self.df['volatility_20'] = self.df['returns'].rolling(20).std() * np.sqrt(252)
        self.df['volatility_60'] = self.df['returns'].rolling(60).std() * np.sqrt(252)
        
        # Momentum
        self.df['momentum_10'] = self.df['close'].pct_change(10)
        self.df['momentum_20'] = self.df['close'].pct_change(20)
        self.df['momentum_50'] = self.df['close'].pct_change(50)
        
        # Trend indicators
        self.df['price_vs_sma20'] = (self.df['close'] - self.df['sma_20']) / self.df['sma_20']
        self.df['price_vs_sma50'] = (self.df['close'] - self.df['sma_50']) / self.df['sma_50']
        self.df['price_vs_sma200'] = (self.df['close'] - self.df['sma_200']) / self.df['sma_200']
        self.df['sma20_vs_sma50'] = (self.df['sma_20'] - self.df['sma_50']) / self.df['sma_50']
        self.df['sma50_vs_sma200'] = (self.df['sma_50'] - self.df['sma_200']) / self.df['sma_200']
        
        # Local drawdowns
        for w in [20, 60, 120]:
            peak_w = self.df['close'].rolling(w).max()
            self.df[f'local_dd_{w}'] = (self.df['close'] - peak_w) / peak_w
        
        # Returns
        self.df['return_1d'] = self.df['close'].pct_change(1)
        self.df['return_3d'] = self.df['close'].pct_change(3)
        self.df['return_5d'] = self.df['close'].pct_change(5)
        self.df['return_7d'] = self.df['close'].pct_change(7)
        self.df['return_10d'] = self.df['close'].pct_change(10)
        
        # Confirmation signals (from analysis)
        self.df['below_sma50'] = (self.df['close'] < self.df['sma_50']).astype(int)
        self.df['negative_momentum_20'] = (self.df['momentum_20'] < 0).astype(int)
        self.df['rsi_below_50'] = (self.df['rsi_14'] < 50).astype(int)
        self.df['macd_bearish'] = (self.df['macd'] < self.df['macd_signal']).astype(int)
        
        print("Features computed.")
    
    def create_labels(self, threshold=-0.05, prediction_days=10):
        """Create labels: will there be a -5% drawdown in next 10 days?"""
        print(f"\nCreating labels: Will there be a {threshold*100:.0f}% drawdown in next {prediction_days} days?")
        
        self.df['will_have_drawdown'] = False
        self.df['max_drawdown_in_period'] = 0.0
        
        for i in range(len(self.df) - prediction_days):
            current_price = self.df.loc[i, 'close']
            peak_price = current_price
            max_drawdown = 0.0
            
            lookahead_window = self.df.iloc[i+1:i+1+prediction_days]
            
            for j, row in lookahead_window.iterrows():
                price = row['close']
                if price > peak_price:
                    peak_price = price
                drawdown = (price - peak_price) / peak_price
                if drawdown < max_drawdown:
                    max_drawdown = drawdown
            
            if max_drawdown <= threshold:
                self.df.loc[i, 'will_have_drawdown'] = True
            self.df.loc[i, 'max_drawdown_in_period'] = max_drawdown
        
        positive_cases = self.df['will_have_drawdown'].sum()
        total_cases = len(self.df) - prediction_days
        
        print(f"Positive cases: {positive_cases} ({positive_cases/total_cases*100:.2f}%)")
        print(f"Negative cases: {total_cases - positive_cases} ({(total_cases - positive_cases)/total_cases*100:.2f}%)")
        
        return self.df
    
    def prepare_data(self, test_size=0.2, val_size=0.2):
        """Prepare train/validation/test splits"""
        print("\nPreparing train/validation/test splits...")
        
        # Feature columns (including confirmation signals)
        feature_cols = [
            # Core indicators
            'rsi_14', 'rsi_7', 'macd', 'macd_hist', 
            'momentum_10', 'momentum_20', 'momentum_50',
            'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
            'sma20_vs_sma50', 'sma50_vs_sma200',
            'volatility_20', 'volatility_60',
            'local_dd_20', 'local_dd_60', 'local_dd_120',
            'return_1d', 'return_3d', 'return_5d', 'return_7d', 'return_10d',
            # Confirmation signals
            'below_sma50', 'negative_momentum_20', 'rsi_below_50', 'macd_bearish'
        ]
        
        # Filter to rows with labels
        data = self.df[self.df['will_have_drawdown'].notna()].copy()
        
        # Select features
        available_features = [col for col in feature_cols if col in data.columns]
        X = data[available_features].copy()
        y = data['will_have_drawdown'].copy()
        dates = data['date'].copy()
        
        # Remove rows with NaN
        valid_mask = ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        dates = dates[valid_mask]
        
        print(f"Valid samples: {len(X)}")
        print(f"Features: {len(available_features)}")
        
        # Split: Train -> (Val + Test)
        X_train, X_temp, y_train, y_temp, dates_train, dates_temp = train_test_split(
            X, y, dates, test_size=test_size + val_size, random_state=42, stratify=y
        )
        
        # Split temp into Val and Test
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test, dates_val, dates_test = train_test_split(
            X_temp, y_temp, dates_temp, test_size=1-val_ratio, random_state=42, stratify=y_temp
        )
        
        print(f"\nTrain set: {len(X_train)} samples ({y_train.sum()} positive, {len(y_train)-y_train.sum()} negative)")
        print(f"Validation set: {len(X_val)} samples ({y_val.sum()} positive, {len(y_val)-y_val.sum()} negative)")
        print(f"Test set: {len(X_test)} samples ({y_test.sum()} positive, {len(y_test)-y_test.sum()} negative)")
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.feature_names = available_features
        self.dates_train = dates_train
        self.dates_val = dates_val
        self.dates_test = dates_test
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Train Logistic Regression"""
        print("\nTraining Logistic Regression...")
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_val_scaled)
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        return model, scaler, y_pred, y_pred_proba
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest"""
        print("\nTraining Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        return model, None, y_pred, y_pred_proba
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost"""
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available, skipping...")
            return None, None, None, None
        
        print("\nTraining XGBoost...")
        
        scale_pos_weight = (y_train == False).sum() / (y_train == True).sum()
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc'
        )
        
        # Try new API first, fallback to old API
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
        except TypeError:
            # New XGBoost API
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        return model, None, y_pred, y_pred_proba
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM"""
        if not LIGHTGBM_AVAILABLE:
            print("LightGBM not available, skipping...")
            return None, None, None, None
        
        print("\nTraining LightGBM...")
        
        scale_pos_weight = (y_train == False).sum() / (y_train == True).sum()
        
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        return model, None, y_pred, y_pred_proba
    
    def train_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """Train Gradient Boosting"""
        print("\nTraining Gradient Boosting...")
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        return model, None, y_pred, y_pred_proba
    
    def evaluate_model(self, y_true, y_pred, y_pred_proba, model_name):
        """Evaluate model performance"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)
        
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'average_precision': ap,
            'confusion_matrix': cm.tolist(),
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")
        print(f"  Avg Prec:  {ap:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"    FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        return results
    
    def train_all_models(self):
        """Train all models and compare"""
        print("\n" + "="*80)
        print("TRAINING ALL MODELS")
        print("="*80)
        
        models = {}
        results = {}
        
        # Train models
        model_functions = {
            'Logistic Regression': self.train_logistic_regression,
            'Random Forest': self.train_random_forest,
            'Gradient Boosting': self.train_gradient_boosting,
        }
        
        if XGBOOST_AVAILABLE:
            model_functions['XGBoost'] = self.train_xgboost
        if LIGHTGBM_AVAILABLE:
            model_functions['LightGBM'] = self.train_lightgbm
        
        for model_name, train_func in model_functions.items():
            try:
                model, scaler, y_pred, y_pred_proba = train_func(
                    self.X_train, self.y_train, self.X_val, self.y_val
                )
                
                if model is None:
                    continue
                
                # Evaluate on validation set
                val_results = self.evaluate_model(
                    self.y_val, y_pred, y_pred_proba, f"{model_name} (Val)"
                )
                
                # Evaluate on test set
                if scaler:
                    X_test_scaled = scaler.transform(self.X_test)
                    y_test_pred = model.predict(X_test_scaled)
                    y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_test_pred = model.predict(self.X_test)
                    y_test_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                test_results = self.evaluate_model(
                    self.y_test, y_test_pred, y_test_pred_proba, f"{model_name} (Test)"
                )
                
                models[model_name] = {
                    'model': model,
                    'scaler': scaler,
                    'val_results': val_results,
                    'test_results': test_results
                }
                results[model_name] = test_results
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        self.models = models
        self.results = results
        
        return models, results
    
    def find_best_model(self):
        """Find best model based on AUC"""
        if not hasattr(self, 'results'):
            print("No results available. Train models first.")
            return None
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc'])
        best_auc = self.results[best_model_name]['auc']
        
        print(f"\n{'='*80}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"AUC Score: {best_auc:.4f}")
        print(f"{'='*80}")
        
        return best_model_name
    
    def plot_results(self):
        """Plot model comparison and ROC curves"""
        if not hasattr(self, 'results'):
            print("No results to plot. Train models first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Model Comparison Bar Chart
        model_names = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            values = [self.results[name][metric] for name in model_names]
            axes[0, 0].bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Comparison')
        axes[0, 0].set_xticks(x + width * 2)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC Curves
        for model_name in model_names:
            model_data = self.models[model_name]
            model = model_data['model']
            scaler = model_data['scaler']
            
            if scaler:
                X_test_scaled = scaler.transform(self.X_test)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            axes[0, 1].plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})')
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature Importance (from best tree-based model)
        best_model_name = self.find_best_model()
        if best_model_name and hasattr(self.models[best_model_name]['model'], 'feature_importances_'):
            importances = self.models[best_model_name]['model'].feature_importances_
            indices = np.argsort(importances)[::-1][:15]
            
            axes[1, 0].barh(range(len(indices)), importances[indices])
            axes[1, 0].set_yticks(range(len(indices)))
            axes[1, 0].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title(f'Top 15 Feature Importance ({best_model_name})')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Confusion Matrix for Best Model
        best_results = self.results[best_model_name]
        cm = np.array(best_results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_title(f'Confusion Matrix ({best_model_name})')
        
        plt.tight_layout()
        plt.savefig('model_training_results.png', dpi=150)
        print("\nPlots saved to model_training_results.png")
        plt.show()
    
    def save_models(self, output_dir='models'):
        """Save trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not hasattr(self, 'models'):
            print("No models to save. Train models first.")
            return
        
        best_model_name = self.find_best_model()
        
        for model_name, model_data in self.models.items():
            model = model_data['model']
            scaler = model_data['scaler']
            
            # Save model
            model_file = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': scaler,
                    'feature_names': self.feature_names,
                    'model_name': model_name,
                    'target': '-5% drawdown in 10 days'
                }, f)
            
            # Save results
            results_file = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_results.json')
            with open(results_file, 'w') as f:
                json.dump({
                    'val_results': model_data['val_results'],
                    'test_results': model_data['test_results']
                }, f, indent=2)
        
        # Save summary
        summary = {
            'best_model': best_model_name,
            'best_auc': self.results[best_model_name]['auc'],
            'target': '-5% drawdown in 10 days',
            'all_results': {k: {
                'auc': v['auc'],
                'f1': v['f1_score'],
                'precision': v['precision'],
                'recall': v['recall']
            } for k, v in self.results.items()},
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(output_dir, 'model_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nModels saved to {output_dir}/")
        print(f"Best model: {best_model_name}")
    
    def save_test_results(self, output_file='test_results.csv'):
        """Save test set results with dates, ASI close, predictions, and actuals"""
        if not hasattr(self, 'models') or not hasattr(self, 'X_test'):
            print("No test data available. Train models first.")
            return None
        
        print("\nSaving test results to CSV...")
        
        # Get the best model
        best_model_name = self.find_best_model()
        if not best_model_name:
            print("No best model found.")
            return None
        
        best_model_data = self.models[best_model_name]
        model = best_model_data['model']
        scaler = best_model_data['scaler']
        
        # Get predictions
        if scaler:
            X_test_scaled = scaler.transform(self.X_test)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Create results dataframe
        test_results_df = pd.DataFrame({
            'date': self.dates_test.values,
            'actual': self.y_test.values,
            'predicted': y_pred,
            'predicted_probability': y_pred_proba
        })
        
        # Merge with original data to get ASI close prices
        test_results_df = test_results_df.merge(
            self.df[['date', 'close']],
            on='date',
            how='left'
        )
        
        # Reorder columns
        test_results_df = test_results_df[['date', 'close', 'actual', 'predicted', 'predicted_probability']]
        
        # Add prediction correctness
        test_results_df['correct'] = (test_results_df['actual'] == test_results_df['predicted']).astype(int)
        
        # Add prediction type
        test_results_df['prediction_type'] = test_results_df.apply(
            lambda row: 'TP' if (row['actual'] == True and row['predicted'] == True) else
                       'TN' if (row['actual'] == False and row['predicted'] == False) else
                       'FP' if (row['actual'] == False and row['predicted'] == True) else
                       'FN', axis=1
        )
        
        # Sort by date
        test_results_df = test_results_df.sort_values('date').reset_index(drop=True)
        
        # Save to CSV
        test_results_df.to_csv(output_file, index=False)
        print(f"Test results saved to {output_file}")
        print(f"Total test samples: {len(test_results_df)}")
        print(f"Positive predictions: {test_results_df['predicted'].sum()}")
        print(f"Actual positives: {test_results_df['actual'].sum()}")
        print(f"Correct predictions: {test_results_df['correct'].sum()} ({test_results_df['correct'].mean()*100:.2f}%)")
        
        return test_results_df


def main():
    """Main training function"""
    csv_file = 'ASI_close_prices_last_15_years_2025-12-05.csv'
    
    # Initialize trainer
    trainer = DownturnPredictorTrainer(csv_file)
    
    # Compute features
    trainer.compute_features()
    
    # Create labels (-5% in 10 days - best combination)
    trainer.create_labels(threshold=-0.05, prediction_days=10)
    
    # Prepare data splits
    trainer.prepare_data(test_size=0.2, val_size=0.2)
    
    # Train all models
    models, results = trainer.train_all_models()
    
    # Find best model
    best_model = trainer.find_best_model()
    
    # Plot results
    trainer.plot_results()
    
    # Save models
    trainer.save_models('models')
    
    # Save test results to CSV
    trainer.save_test_results('test_results.csv')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\nModels are ready to use for predicting -5% drawdowns in 10 days!")


if __name__ == "__main__":
    main()

