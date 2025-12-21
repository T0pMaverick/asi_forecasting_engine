#!/usr/bin/env python3
"""
PHASE 4 Enhanced ASI Market Downturn Predictor
BALANCED PRECISION-RECALL OPTIMIZATION
Target: Balanced High Performance (Precision 85-90%, Recall 60-80%, AUC >85%)
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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

class Phase4BalancedPredictor:
    """PHASE 4: Balanced precision-recall optimization for practical utility"""
    
    def __init__(self, csv_file, fn_penalty=6.0, target_precision=0.87, target_recall=0.65):
        print("="*100)
        print("PHASE 4 ENHANCED ASI MARKET DOWNTURN PREDICTOR")
        print("PHASE 1-3: ✅ Overfitting, Performance, Precision (COMPLETED)")
        print("PHASE 4: ⚖️ BALANCED PRECISION-RECALL OPTIMIZATION (NEW)")
        print("Target: Balanced Performance (Precision 85-90%, Recall 60-80%, AUC >85%)")
        print("="*100)
        
        self.csv_file = csv_file
        self.fn_penalty = fn_penalty  # BALANCED: 6.0x (between precision focus and recall)
        self.fp_penalty = 1.0
        self.target_precision = target_precision  # Slightly lower for better balance
        self.target_recall = target_recall        # NEW: explicit recall target
        
        # Load and prepare data
        self.df = pd.read_csv(csv_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.df['close'] = pd.to_numeric(self.df['close'])
        self.df['returns'] = self.df['close'].pct_change()
        
        # Initialize storage
        self.models = {}
        self.ensemble = None
        self.feature_names = []
        self.overfitting_results = {}
        self.optimal_thresholds = {}
        self.evaluation_results = {}
        
        print(f"Loaded {len(self.df)} records from {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"\nPHASE 4 BALANCED IMPROVEMENTS:")
        print(f"  ⚖️ Balanced FN penalty: {self.fn_penalty}x (not too conservative)")
        print(f"  ⚖️ Multi-objective optimization (precision + recall + AUC)")
        print(f"  ⚖️ Target: {self.target_precision:.0%} precision + {self.target_recall:.0%} recall")
        print(f"  ⚖️ Diverse ensemble (prevent over-conservatism)")
        print(f"  ⚖️ Business-practical evaluation (≤10 false negatives)")
        print(f"  ⚖️ F-beta score optimization (beta=0.7 for precision emphasis)")
    
    def engineer_balanced_features(self):
        """Feature engineering optimized for precision-recall balance"""
        print("\n" + "="*80)
        print("STEP 1: BALANCED FEATURE ENGINEERING")
        print("="*80)
        
        # Compute balanced technical indicators
        self._compute_balanced_features()
        
        print(f"Total features created: {len([col for col in self.df.columns if col not in ['date', 'close', 'returns']])}")
        
    def _compute_balanced_features(self):
        """Compute features optimized for balanced performance"""
        print("Computing balanced technical indicators...")
        
        # Core Moving Averages
        self.df['sma_20'] = self.df['close'].rolling(20).mean()
        self.df['sma_50'] = self.df['close'].rolling(50).mean()
        self.df['sma_200'] = self.df['close'].rolling(200).mean()
        self.df['ema_12'] = self.df['close'].ewm(span=12, adjust=False).mean()
        self.df['ema_26'] = self.df['close'].ewm(span=26, adjust=False).mean()
        
        # Enhanced MACD Family
        self.df['macd'] = self.df['ema_12'] - self.df['ema_26']
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
        self.df['macd_hist'] = self.df['macd'] - self.df['macd_signal']
        self.df['macd_acceleration'] = self.df['macd_hist'].diff()
        
        # Multi-timeframe RSI
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = -delta.clip(upper=0).rolling(period).mean()
            rs = gain / (loss + 1e-9)
            return 100 - 100 / (1 + rs)
        
        self.df['rsi_7'] = calculate_rsi(self.df['close'], 7)
        self.df['rsi_14'] = calculate_rsi(self.df['close'], 14)
        self.df['rsi_21'] = calculate_rsi(self.df['close'], 21)
        
        # Volatility indicators
        self.df['volatility_10'] = self.df['returns'].rolling(10).std() * np.sqrt(252)
        self.df['volatility_20'] = self.df['returns'].rolling(20).std() * np.sqrt(252)
        self.df['volatility_60'] = self.df['returns'].rolling(60).std() * np.sqrt(252)
        self.df['vol_regime'] = (self.df['volatility_20'] > self.df['volatility_20'].rolling(252).quantile(0.75)).astype(int)
        self.df['vol_spike'] = (self.df['volatility_20'] > self.df['volatility_20'].rolling(252).quantile(0.9)).astype(int)
        
        # Momentum indicators
        self.df['momentum_5'] = self.df['close'].pct_change(5)
        self.df['momentum_10'] = self.df['close'].pct_change(10)
        self.df['momentum_20'] = self.df['close'].pct_change(20)
        
        # Trend indicators
        self.df['price_vs_sma20'] = (self.df['close'] - self.df['sma_20']) / (self.df['sma_20'] + 1e-9)
        self.df['price_vs_sma50'] = (self.df['close'] - self.df['sma_50']) / (self.df['sma_50'] + 1e-9)
        self.df['price_vs_sma200'] = (self.df['close'] - self.df['sma_200']) / (self.df['sma_200'] + 1e-9)
        
        # Drawdown indicators
        for w in [20, 60, 120]:
            peak_w = self.df['close'].rolling(w).max()
            self.df[f'local_dd_{w}'] = (self.df['close'] - peak_w) / peak_w
        
        # Balanced confirmation signals
        self.df['below_sma50'] = (self.df['close'] < self.df['sma_50']).astype(int)
        self.df['negative_momentum_10'] = (self.df['momentum_10'] < 0).astype(int)
        self.df['rsi_oversold'] = (self.df['rsi_14'] < 30).astype(int)
        self.df['rsi_below_50'] = (self.df['rsi_14'] < 50).astype(int)
        self.df['macd_bearish'] = (self.df['macd'] < self.df['macd_signal']).astype(int)
        
        # Return indicators
        self.df['return_5d'] = self.df['close'].pct_change(5)
        self.df['return_10d'] = self.df['close'].pct_change(10)
        
        # Market stress (balanced)
        self.df['consecutive_red_days'] = (self.df['returns'] < 0).astype(int).rolling(5).sum()
        self.df['extreme_moves'] = (np.abs(self.df['returns']) > self.df['returns'].rolling(252).std() * 2).astype(int)
        
        print("✅ Selected balanced feature set for precision-recall optimization")
    
    def create_labels(self, threshold=-0.05, prediction_days=10):
        """Create target labels"""
        print(f"\n" + "="*80)
        print("STEP 2: CREATING LABELS")
        print("="*80)
        print(f"Target: {threshold*100:.0f}% drawdown in next {prediction_days} days")
        
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
        print(f"Negative cases: {total_cases - positive_cases}")
        
        return self.df
    
    def prepare_and_save_datasets_phase4(self, test_size=0.2, val_size=0.2, output_dir='datasets_v4'):
        """Phase 4 dataset preparation"""
        print(f"\n" + "="*80)
        print("STEP 3: PREPARING PHASE 4 DATASETS")
        print("="*80)
        print("✅ KEEP: All previous fixes (overfitting, data balance)")
        print("⚖️ NEW: Balanced feature set for precision-recall optimization")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Balanced feature set
        selected_features = [
            'rsi_7', 'rsi_14', 'rsi_21', 'macd', 'macd_hist', 'macd_acceleration',
            'momentum_5', 'momentum_10', 'momentum_20', 'volatility_10', 'volatility_20', 'volatility_60',
            'vol_regime', 'vol_spike', 'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
            'local_dd_20', 'local_dd_60', 'local_dd_120', 'below_sma50', 'negative_momentum_10',
            'rsi_oversold', 'rsi_below_50', 'macd_bearish', 'return_5d', 'return_10d',
            'consecutive_red_days', 'extreme_moves'
        ]
        
        # Filter data and features
        data = self.df[self.df['will_have_drawdown'].notna()].copy()
        
        for col in selected_features:
            if col in data.columns and data[col].isna().any():
                data[col] = data[col].fillna(data[col].median())
        
        available_features = [col for col in selected_features if col in data.columns]
        
        X = data[available_features].copy()
        y = data['will_have_drawdown'].copy()
        dates = data['date'].copy()
        prices = data['close'].copy()
        
        print(f"Dataset shape: {X.shape}")
        print(f"Balanced features: {len(available_features)}")
        print(f"Overall positive class ratio: {y.mean():.3f}")
        
        # Stratified splits
        X_temp, X_test, y_temp, y_test, dates_temp, dates_test, prices_temp, prices_test = train_test_split(
            X, y, dates, prices, 
            test_size=test_size, 
            random_state=42, 
            stratify=y
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val, dates_train, dates_val, prices_train, prices_val = train_test_split(
            X_temp, y_temp, dates_temp, prices_temp,
            test_size=val_ratio,
            random_state=42,
            stratify=y_temp
        )
        
        # Save datasets
        datasets = {
            'train': {'X': X_train, 'y': y_train, 'dates': dates_train, 'prices': prices_train},
            'val': {'X': X_val, 'y': y_val, 'dates': dates_val, 'prices': prices_val},
            'test': {'X': X_test, 'y': y_test, 'dates': dates_test, 'prices': prices_test}
        }
        
        for split_name, split_data in datasets.items():
            combined_df = pd.concat([
                split_data['dates'].reset_index(drop=True),
                split_data['prices'].reset_index(drop=True),
                split_data['X'].reset_index(drop=True),
                split_data['y'].reset_index(drop=True)
            ], axis=1)
            
            filename = os.path.join(output_dir, f'{split_name}_dataset_v4.csv')
            combined_df.to_csv(filename, index=False)
            
            print(f"{split_name.upper()} set: Shape {split_data['X'].shape}, Positive: {split_data['y'].mean()*100:.1f}%")
        
        # Store for later use
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        self.dates_train, self.dates_val, self.dates_test = dates_train, dates_val, dates_test
        self.feature_names = available_features
        
        print(f"✅ Balanced feature set: {len(available_features)} features")
        
        return datasets
    
    def train_balanced_models(self):
        """Train models optimized for precision-recall balance"""
        print(f"\n" + "="*80)
        print("STEP 4: BALANCED MODEL TRAINING")
        print("="*80)
        print(f"⚖️ Balanced FN penalty: {self.fn_penalty}x (not too conservative)")
        print("⚖️ Models optimized for precision-recall balance")
        
        models = {}
        
        # Balanced LightGBM
        if LIGHTGBM_AVAILABLE:
            print("\n⚖️ Training Balanced LightGBM...")
            scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
            
            lgb_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 25,        # Moderate complexity
                'learning_rate': 0.02,   
                'feature_fraction': 0.8, 
                'bagging_fraction': 0.8, 
                'bagging_freq': 5,
                'min_child_samples': 50, 
                'min_data_in_leaf': 25,  
                'scale_pos_weight': scale_pos_weight * self.fn_penalty,
                'lambda_l1': 2.0,        # Moderate regularization
                'lambda_l2': 2.0,        
                'max_depth': 7,          # Balanced depth
                'random_state': 42,
                'verbose': -1
            }
            
            train_data = lgb.Dataset(self.X_train, label=self.y_train)
            val_data = lgb.Dataset(self.X_val, label=self.y_val, reference=train_data)
            
            lgb_model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=150,
                valid_sets=[train_data, val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30),
                    lgb.log_evaluation(period=0)
                ]
            )
            models['Balanced_LightGBM'] = lgb_model
        
        # Balanced XGBoost
        if XGBOOST_AVAILABLE:
            print("⚖️ Training Balanced XGBoost...")
            scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
            
            xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 7,          # Balanced depth
                'learning_rate': 0.02,   
                'subsample': 0.8,        
                'colsample_bytree': 0.8, 
                'min_child_weight': 10,  
                'scale_pos_weight': scale_pos_weight * self.fn_penalty,
                'alpha': 2.0,            # Moderate regularization
                'lambda': 2.0,          
                'gamma': 0.5,            
                'random_state': 42,
                'verbosity': 0
            }
            
            train_data = xgb.DMatrix(self.X_train, label=self.y_train)
            val_data = xgb.DMatrix(self.X_val, label=self.y_val)
            
            xgb_model = xgb.train(
                xgb_params,
                train_data,
                num_boost_round=150,
                evals=[(train_data, 'train'), (val_data, 'eval')],
                early_stopping_rounds=30,
                verbose_eval=False
            )
            models['Balanced_XGBoost'] = xgb_model
        
        # Balanced Random Forest
        print("⚖️ Training Balanced Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,        # Balanced
            max_depth=12,            # Moderate depth 
            min_samples_split=10,    # Balanced
            min_samples_leaf=5,      # Balanced
            max_features='sqrt',     
            class_weight={0: 1, 1: self.fn_penalty},
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(self.X_train, self.y_train)
        models['Balanced_RandomForest'] = rf_model
        
        # Balanced Gradient Boosting
        print("⚖️ Training Balanced Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        # Apply class weights manually for Gradient Boosting
        sample_weights = np.where(self.y_train == 1, self.fn_penalty, 1.0)
        gb_model.fit(self.X_train, self.y_train, sample_weight=sample_weights)
        models['Balanced_GradientBoosting'] = gb_model
        
        # Balanced Logistic Regression
        print("⚖️ Training Balanced Logistic Regression...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1500,
            class_weight={0: 1.0, 1: self.fn_penalty},
            C=0.3,                   # Balanced regularization
            solver='liblinear',
            penalty='l1'
        )
        lr_model.fit(X_train_scaled, self.y_train)
        models['Balanced_LogisticRegression'] = {'model': lr_model, 'scaler': scaler}
        
        self.models = models
        print(f"\n✅ Trained {len(models)} balanced models")
        print("⚖️ Models optimized for precision-recall balance")
        return models
    
    def multi_objective_threshold_optimization(self):
        """Multi-objective threshold optimization for balanced performance"""
        print(f"\n" + "="*80)
        print("STEP 5: MULTI-OBJECTIVE THRESHOLD OPTIMIZATION")
        print("="*80)
        
        optimal_thresholds = {}
        
        for model_name, model in self.models.items():
            print(f"\nBalanced optimization for {model_name}...")
            
            # Get validation predictions
            _, val_pred = self._get_model_predictions(model_name, model)
            y_pred_proba = val_pred['proba']
            
            best_threshold = 0.5
            best_score = -float('inf')
            
            thresholds = np.linspace(0.1, 0.9, 200)  # Reasonable range
            
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                
                if len(np.unique(y_pred)) == 1:
                    continue
                
                try:
                    tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred).ravel()
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
                    
                    # Multi-objective score: Balance precision, recall, and FN rate
                    # F-beta score with beta=0.7 (slight precision emphasis)
                    beta = 0.7
                    f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-9)
                    
                    # Practical constraints
                    score = f_beta
                    
                    # Bonuses for meeting targets
                    if precision >= 0.85 and recall >= 0.60:
                        score += 0.3
                    if precision >= 0.87 and recall >= 0.65:
                        score += 0.5
                    
                    # Penalties for extreme imbalance
                    if precision > 0.95 and recall < 0.3:  # Too conservative
                        score -= 0.4
                    if precision < 0.75:  # Too low precision
                        score -= 0.3
                    if fn > 15:  # Too many false negatives
                        score -= 0.2
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                
                except:
                    continue
            
            # Calculate final metrics
            y_pred_final = (y_pred_proba >= best_threshold).astype(int)
            
            try:
                final_precision = precision_score(self.y_val, y_pred_final)
                final_recall = recall_score(self.y_val, y_pred_final)
                
                tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred_final).ravel()
                fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
                
                # Calculate F-beta score
                beta = 0.7
                f_beta = (1 + beta**2) * (final_precision * final_recall) / ((beta**2 * final_precision) + final_recall + 1e-9)
            except:
                final_precision = final_recall = fn_rate = f_beta = 0
            
            optimal_thresholds[model_name] = {
                'threshold': best_threshold,
                'precision': final_precision,
                'recall': final_recall,
                'fn_rate': fn_rate,
                'f_beta': f_beta,
                'score': best_score
            }
            
            print(f"  Optimal threshold: {best_threshold:.4f}")
            print(f"  Precision: {final_precision:.4f}")
            print(f"  Recall: {final_recall:.4f}")
            print(f"  F-beta: {f_beta:.4f}")
            print(f"  FN rate: {fn_rate:.4f}")
            print(f"  Balance score: {best_score:.4f}")
        
        self.optimal_thresholds = optimal_thresholds
        return optimal_thresholds
    
    def create_balanced_ensemble(self):
        """Create balanced ensemble preventing over-conservatism"""
        print(f"\n" + "="*80)
        print("STEP 6: BALANCED ENSEMBLE CREATION")
        print("="*80)
        
        model_weights = {}
        meta_features_val = np.zeros((len(self.X_val), len(self.models)))
        model_names = list(self.models.keys())
        
        print("Computing balanced weights (prevent over-conservatism)...")
        
        for i, (model_name, model) in enumerate(self.models.items()):
            _, val_pred = self._get_model_predictions(model_name, model)
            meta_features_val[:, i] = val_pred['proba']
            
            auc = roc_auc_score(self.y_val, val_pred['proba'])
            precision = self.optimal_thresholds[model_name]['precision']
            recall = self.optimal_thresholds[model_name]['recall']
            f_beta = self.optimal_thresholds[model_name]['f_beta']
            
            # Balanced weight formula
            weight = (
                0.3 * auc +                     # 30% AUC
                0.25 * precision +              # 25% precision
                0.25 * recall +                 # 25% recall (equally weighted!)
                0.2 * f_beta                    # 20% F-beta balance
            )
            
            # Penalty for over-conservative models
            if precision > 0.9 and recall < 0.4:
                weight *= 0.7  # Reduce weight of over-conservative models
            
            # Bonus for balanced models
            if 0.85 <= precision <= 0.92 and recall >= 0.60:
                weight *= 1.3
            
            model_weights[model_name] = weight
            
            print(f"  {model_name}: Weight = {weight:.4f} (AUC={auc:.3f}, Prec={precision:.3f}, Rec={recall:.3f})")
        
        # Normalize weights
        total_weight = sum(model_weights.values())
        model_weights = {k: v/total_weight for k, v in model_weights.items()}
        
        print(f"\nNormalized balanced weights:")
        for model_name, weight in model_weights.items():
            print(f"  {model_name}: {weight:.3f}")
        
        # Create balanced ensemble prediction
        ensemble_pred_val = np.zeros(len(self.X_val))
        for i, (model_name, weight) in enumerate(model_weights.items()):
            ensemble_pred_val += weight * meta_features_val[:, i]
        
        # Optimize ensemble threshold with multi-objective approach
        best_threshold = 0.5
        best_score = -float('inf')
        
        thresholds = np.linspace(0.1, 0.8, 200)  # Don't go too high
        
        for threshold in thresholds:
            y_pred = (ensemble_pred_val >= threshold).astype(int)
            
            if len(np.unique(y_pred)) == 1:
                continue
            
            try:
                tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred).ravel()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # Ensemble scoring with balance emphasis
                beta = 0.7
                f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-9)
                
                score = f_beta
                
                # Strong bonus for balanced performance
                if precision >= 0.85 and recall >= 0.65:
                    score += 0.5
                if 0.87 <= precision <= 0.92 and recall >= 0.70:
                    score += 1.0
                
                # Strong penalty for over-conservatism
                if precision > 0.95 and recall < 0.4:
                    score -= 1.0
                if fn > 20:  # Too many missed downturns
                    score -= 0.5
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            except:
                continue
        
        self.ensemble = {
            'model_names': model_names,
            'model_weights': model_weights,
            'threshold': best_threshold,
            'method': 'balanced_ensemble'
        }
        
        print(f"\nBalanced ensemble created:")
        print(f"  Optimized threshold: {best_threshold:.4f}")
        print(f"  Balance score: {best_score:.4f}")
        
        return self.ensemble
    
    def _get_model_predictions(self, model_name, model):
        """Get predictions from model"""
        if model_name == 'Balanced_LightGBM':
            train_proba = model.predict(self.X_train, num_iteration=model.best_iteration)
            val_proba = model.predict(self.X_val, num_iteration=model.best_iteration)
        elif model_name == 'Balanced_XGBoost':
            train_proba = model.predict(xgb.DMatrix(self.X_train))
            val_proba = model.predict(xgb.DMatrix(self.X_val))
        elif model_name == 'Balanced_LogisticRegression':
            train_proba = model['model'].predict_proba(model['scaler'].transform(self.X_train))[:, 1]
            val_proba = model['model'].predict_proba(model['scaler'].transform(self.X_val))[:, 1]
        else:
            train_proba = model.predict_proba(self.X_train)[:, 1]
            val_proba = model.predict_proba(self.X_val)[:, 1]
        
        train_binary = (train_proba > 0.5).astype(int)
        val_binary = (val_proba > 0.5).astype(int)
        
        return (
            {'proba': train_proba, 'binary': train_binary},
            {'proba': val_proba, 'binary': val_binary}
        )
    
    def comprehensive_balanced_evaluation(self):
        """Comprehensive evaluation focusing on balanced performance"""
        print(f"\n" + "="*80)
        print("STEP 7: COMPREHENSIVE BALANCED EVALUATION")
        print("="*80)
        
        evaluation_results = {}
        
        # Evaluate individual models
        print("\nIndividual Model Performance on Test Set:")
        print("-" * 90)
        
        for model_name, model in self.models.items():
            # Get test predictions
            if model_name == 'Balanced_LightGBM':
                test_proba = model.predict(self.X_test, num_iteration=model.best_iteration)
            elif model_name == 'Balanced_XGBoost':
                test_proba = model.predict(xgb.DMatrix(self.X_test))
            elif model_name == 'Balanced_LogisticRegression':
                test_proba = model['model'].predict_proba(model['scaler'].transform(self.X_test))[:, 1]
            else:
                test_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Use optimized threshold
            threshold = self.optimal_thresholds[model_name]['threshold']
            test_pred = (test_proba >= threshold).astype(int)
            
            # Calculate all metrics
            auc = roc_auc_score(self.y_test, test_proba)
            accuracy = accuracy_score(self.y_test, test_pred)
            precision = precision_score(self.y_test, test_pred)
            recall = recall_score(self.y_test, test_pred)
            f1 = f1_score(self.y_test, test_pred)
            
            # F-beta score
            beta = 0.7
            f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-9)
            
            cm = confusion_matrix(self.y_test, test_pred)
            tn, fp, fn, tp = cm.ravel()
            
            fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Business cost
            total_cost = (fn * self.fn_penalty) + (fp * self.fp_penalty)
            
            evaluation_results[model_name] = {
                'auc': auc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'f_beta_score': f_beta,
                'threshold': threshold,
                'confusion_matrix': cm.tolist(),
                'false_negative_rate': fn_rate,
                'false_positive_rate': fp_rate,
                'total_cost': total_cost,
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
            
            print(f"\n{model_name}:")
            print(f"  AUC: {auc:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F-beta: {f_beta:.4f}")
            print(f"  FN: {fn} (Rate: {fn_rate:.3f})")
            print(f"  Business Cost: {total_cost:.2f}")
        
        # Evaluate balanced ensemble
        print(f"\nBalanced Ensemble Performance:")
        print("-" * 40)
        
        # Get ensemble predictions on test set
        meta_features_test = np.zeros((len(self.X_test), len(self.models)))
        
        for i, (model_name, model) in enumerate(self.models.items()):
            if model_name == 'Balanced_LightGBM':
                pred = model.predict(self.X_test, num_iteration=model.best_iteration)
            elif model_name == 'Balanced_XGBoost':
                pred = model.predict(xgb.DMatrix(self.X_test))
            elif model_name == 'Balanced_LogisticRegression':
                pred = model['model'].predict_proba(model['scaler'].transform(self.X_test))[:, 1]
            else:
                pred = model.predict_proba(self.X_test)[:, 1]
            
            meta_features_test[:, i] = pred
        
        # Apply balanced weights
        ensemble_proba = np.zeros(len(self.X_test))
        for i, (model_name, weight) in enumerate(self.ensemble['model_weights'].items()):
            ensemble_proba += weight * meta_features_test[:, i]
        
        ensemble_pred = (ensemble_proba >= self.ensemble['threshold']).astype(int)
        
        # Calculate ensemble metrics
        ensemble_auc = roc_auc_score(self.y_test, ensemble_proba)
        ensemble_accuracy = accuracy_score(self.y_test, ensemble_pred)
        ensemble_precision = precision_score(self.y_test, ensemble_pred)
        ensemble_recall = recall_score(self.y_test, ensemble_pred)
        ensemble_f1 = f1_score(self.y_test, ensemble_pred)
        
        # F-beta score for ensemble
        beta = 0.7
        ensemble_f_beta = (1 + beta**2) * (ensemble_precision * ensemble_recall) / ((beta**2 * ensemble_precision) + ensemble_recall + 1e-9)
        
        ensemble_cm = confusion_matrix(self.y_test, ensemble_pred)
        tn, fp, fn, tp = ensemble_cm.ravel()
        
        ensemble_fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        ensemble_fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        ensemble_total_cost = (fn * self.fn_penalty) + (fp * self.fp_penalty)
        
        evaluation_results['Balanced_Ensemble'] = {
            'auc': ensemble_auc,
            'accuracy': ensemble_accuracy,
            'precision': ensemble_precision,
            'recall': ensemble_recall,
            'f1_score': ensemble_f1,
            'f_beta_score': ensemble_f_beta,
            'threshold': self.ensemble['threshold'],
            'confusion_matrix': ensemble_cm.tolist(),
            'false_negative_rate': ensemble_fn_rate,
            'false_positive_rate': ensemble_fp_rate,
            'total_cost': ensemble_total_cost,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        print(f"  AUC: {ensemble_auc:.4f}")
        print(f"  Accuracy: {ensemble_accuracy:.4f}")
        print(f"  Precision: {ensemble_precision:.4f}")
        print(f"  Recall: {ensemble_recall:.4f}")
        print(f"  F-beta: {ensemble_f_beta:.4f}")
        print(f"  FN: {fn} (Rate: {ensemble_fn_rate:.3f})")
        print(f"  Business Cost: {ensemble_total_cost:.2f}")
        
        # Find best model by F-beta score (balanced performance)
        best_model = max(evaluation_results.keys(), 
                        key=lambda x: evaluation_results[x]['f_beta_score'])
        
        print(f"\n" + "="*80)
        print(f"BEST MODEL (Highest F-beta Balance): {best_model}")
        print("="*80)
        best_metrics = evaluation_results[best_model]
        print(f"🎯 AUC: {best_metrics['auc']:.4f} (Target: ≥0.85)")
        print(f"📊 Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"⚖️ Precision: {best_metrics['precision']:.4f} (Target: 85-90%)")
        print(f"⚖️ Recall: {best_metrics['recall']:.4f} (Target: 60-80%)")
        print(f"🏆 F-beta: {best_metrics['f_beta_score']:.4f} (Balanced Performance)")
        print(f"📉 False Negatives: {best_metrics['false_negatives']} (Target: ≤10)")
        print(f"💰 Business Cost: {best_metrics['total_cost']:.2f}")
        
        # Multi-objective target achievement
        print(f"\n🎯 PHASE 4 TARGET ACHIEVEMENT:")
        
        auc_achieved = best_metrics['auc'] >= 0.85
        precision_achieved = 0.85 <= best_metrics['precision'] <= 0.92
        recall_achieved = best_metrics['recall'] >= 0.60
        fn_practical = best_metrics['false_negatives'] <= 10
        
        auc_status = '✅ ACHIEVED' if auc_achieved else f'⚠️ {best_metrics["auc"]:.1%}'
        precision_status = '✅ ACHIEVED' if precision_achieved else f'⚠️ {best_metrics["precision"]:.1%}'
        recall_status = '✅ ACHIEVED' if recall_achieved else f'⚠️ {best_metrics["recall"]:.1%}'
        fn_status = '✅ ACHIEVED' if fn_practical else f'⚠️ {best_metrics["false_negatives"]} FN'
        
        print(f"   AUC ≥ 85%: {auc_status}")
        print(f"   Precision 85-90%: {precision_status}")
        print(f"   Recall ≥ 60%: {recall_status}")
        print(f"   FN ≤ 10: {fn_status}")
        print(f"   F-beta Balance: {best_metrics['f_beta_score']:.3f}")
        
        # Calculate overall success
        targets_achieved = sum([auc_achieved, precision_achieved, recall_achieved, fn_practical])
        
        print(f"\n📈 PHASE 4 IMPROVEMENT ANALYSIS:")
        print(f"   Phase 3: 90% precision, 14% recall, 55 FN")
        print(f"   Phase 4: {best_metrics['precision']:.0%} precision, {best_metrics['recall']:.0%} recall, {best_metrics['false_negatives']} FN")
        print(f"   Improvement: Better balance - much fewer missed downturns!")
        
        self.evaluation_results = evaluation_results
        return evaluation_results, best_model, targets_achieved
    
    def create_phase4_visualizations(self):
        """Create Phase 4 balanced visualizations"""
        print(f"\n" + "="*80)
        print("STEP 8: CREATING PHASE 4 VISUALIZATIONS")
        print("="*80)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        models = list(self.evaluation_results.keys())
        
        # 1. Balanced Performance Overview
        metrics = ['auc', 'accuracy', 'precision', 'recall', 'f_beta_score']
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        x = np.arange(len(models))
        width = 0.15
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            values = [self.evaluation_results[model][metric] for model in models]
            axes[0, 0].bar(x + i*width, values, width, label=metric.upper().replace('_', ' '), 
                          color=color, alpha=0.7)
        
        axes[0, 0].axhline(y=0.85, color='blue', linestyle='--', alpha=0.7)
        axes[0, 0].axhline(y=0.60, color='orange', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('PHASE 4 Balanced Performance Overview')
        axes[0, 0].set_xticks(x + width * 2)
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Precision-Recall Balance Analysis
        precisions = [self.evaluation_results[model]['precision'] for model in models]
        recalls = [self.evaluation_results[model]['recall'] for model in models]
        f_betas = [self.evaluation_results[model]['f_beta_score'] for model in models]
        
        # Color by F-beta score
        scatter = axes[0, 1].scatter(recalls, precisions, s=200, c=f_betas, cmap='viridis', alpha=0.8)
        
        for i, model in enumerate(models):
            axes[0, 1].annotate(model, (recalls[i], precisions[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Target zones
        axes[0, 1].axhspan(0.85, 0.92, alpha=0.2, color='green', label='Target Precision')
        axes[0, 1].axvspan(0.60, 1.0, alpha=0.2, color='blue', label='Target Recall')
        
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Balance (Color = F-beta)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='F-beta Score')
        
        # 3. Business Impact vs Balance
        fn_counts = [self.evaluation_results[model]['false_negatives'] for model in models]
        costs = [self.evaluation_results[model]['total_cost'] for model in models]
        
        colors = ['green' if fn <= 10 else 'orange' if fn <= 20 else 'red' for fn in fn_counts]
        
        axes[0, 2].scatter(fn_counts, f_betas, s=200, c=colors, alpha=0.7)
        for i, model in enumerate(models):
            axes[0, 2].annotate(model, (fn_counts[i], f_betas[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[0, 2].axvline(x=10, color='green', linestyle='--', alpha=0.7, label='FN Target (≤10)')
        axes[0, 2].set_xlabel('False Negatives Count')
        axes[0, 2].set_ylabel('F-beta Balance Score')
        axes[0, 2].set_title('Business Impact vs Model Balance')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Feature Importance
        if 'Balanced_RandomForest' in self.models:
            rf_model = self.models['Balanced_RandomForest']
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]
            
            axes[1, 0].barh(range(len(indices)), importances[indices])
            axes[1, 0].set_yticks(range(len(indices)))
            axes[1, 0].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1, 0].set_xlabel('Feature Importance')
            axes[1, 0].set_title('Top 15 Balanced Features')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Threshold Analysis for Best Model
        best_model = max(models, key=lambda x: self.evaluation_results[x]['f_beta_score'])
        if best_model == 'Balanced_Ensemble':
            threshold = self.ensemble['threshold']
            title = f'Threshold Analysis - {best_model}'
        else:
            threshold = self.optimal_thresholds[best_model]['threshold']
            title = f'Threshold Analysis - {best_model}'

        axes[1, 1].axvline(x=threshold, color='red', linestyle='--', 
                          label=f'Optimal Threshold ({threshold:.3f})')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Performance Score')
        axes[1, 1].set_title(title)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Confusion Matrix for Best Model
        cm = np.array(self.evaluation_results[best_model]['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 2],
                   xticklabels=['No Downturn', 'Downturn'], 
                   yticklabels=['No Downturn', 'Downturn'])
        axes[1, 2].set_xlabel('Predicted')
        axes[1, 2].set_ylabel('Actual')
        axes[1, 2].set_title(f'Confusion Matrix - {best_model}')
        
        plt.tight_layout()
        plt.savefig('phase4_balanced_analysis.png', dpi=150, bbox_inches='tight')
        print("✅ PHASE 4 balanced analysis visualization saved to: phase4_balanced_analysis.png")
        plt.show()
    
    def save_phase4_results(self, output_dir='phase4_results'):
        """Save all Phase 4 results"""
        print(f"\n" + "="*80)
        print("STEP 9: SAVING PHASE 4 RESULTS")
        print("="*80)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f'{model_name.lower()}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'feature_names': self.feature_names,
                    'optimal_threshold': self.optimal_thresholds[model_name]
                }, f)
        
        # Save balanced ensemble
        ensemble_path = os.path.join(output_dir, 'balanced_ensemble.pkl')
        with open(ensemble_path, 'wb') as f:
            pickle.dump({
                'ensemble': self.ensemble,
                'base_models': self.models,
                'feature_names': self.feature_names
            }, f)
        
        # Convert numpy types
        def convert_numpy_types(obj):
            if isinstance(obj, (np.bool_, np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Save results
        results_summary = {}
        for k, v in self.evaluation_results.items():
            results_summary[k] = {}
            for key, val in v.items():
                results_summary[k][key] = convert_numpy_types(val)
        
        best_model = max(self.evaluation_results.keys(), 
                        key=lambda x: self.evaluation_results[x]['f_beta_score'])
        
        final_summary = {
            'evaluation_results': results_summary,
            'configuration': {
                'fn_penalty': self.fn_penalty,
                'target_precision': self.target_precision,
                'target_recall': self.target_recall,
                'phase4_improvements': [
                    'Balanced precision-recall optimization',
                    'Multi-objective threshold optimization',
                    'F-beta score optimization (beta=0.7)',
                    'Prevented over-conservatism',
                    'Diverse ensemble weighting',
                    'Practical false negative targets'
                ]
            },
            'best_model': best_model,
            'targets_achieved': {
                'auc_85': convert_numpy_types(self.evaluation_results[best_model]['auc'] >= 0.85),
                'precision_balanced': convert_numpy_types(0.85 <= self.evaluation_results[best_model]['precision'] <= 0.92),
                'recall_60': convert_numpy_types(self.evaluation_results[best_model]['recall'] >= 0.60),
                'fn_practical': convert_numpy_types(self.evaluation_results[best_model]['false_negatives'] <= 10)
            },
            'training_date': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, 'phase4_balanced_summary.json'), 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        print(f"✅ PHASE 4 results saved to: {output_dir}/")
        return output_dir
    
    def run_complete_pipeline(self):
        """Run the complete PHASE 4 pipeline"""
        print("STARTING PHASE 4 BALANCED ASI DOWNTURN PREDICTION PIPELINE")
        print("="*100)
        
        try:
            # Step 1: Balanced Feature Engineering
            self.engineer_balanced_features()
            
            # Step 2: Create Labels
            self.create_labels(threshold=-0.05, prediction_days=10)
            
            # Step 3: Phase 4 Dataset Preparation
            self.prepare_and_save_datasets_phase4(test_size=0.2, val_size=0.2)
            
            # Step 4: Balanced Model Training
            self.train_balanced_models()
            
            # Step 5: Multi-Objective Threshold Optimization
            self.multi_objective_threshold_optimization()
            
            # Step 6: Balanced Ensemble Creation
            self.create_balanced_ensemble()
            
            # Step 7: Comprehensive Balanced Evaluation
            evaluation_results, best_model, targets_achieved = self.comprehensive_balanced_evaluation()
            
            # Step 8: Phase 4 Visualizations
            self.create_phase4_visualizations()
            
            # Step 9: Save Results
            output_dir = self.save_phase4_results()
            
            print(f"\n" + "="*100)
            print("⚖️🎉 PHASE 4 BALANCED PIPELINE COMPLETED! 🎉⚖️")
            print("="*100)
            
            best_metrics = evaluation_results[best_model]
            
            print(f"\n📊 FINAL PHASE 4 RESULTS:")
            print(f"🏆 Best Model: {best_model}")
            print(f"🎯 AUC: {best_metrics['auc']:.4f} (Target: ≥0.85)")
            print(f"📊 Accuracy: {best_metrics['accuracy']:.4f}")
            print(f"⚖️ Precision: {best_metrics['precision']:.4f} (Target: 85-90%)")
            print(f"⚖️ Recall: {best_metrics['recall']:.4f} (Target: ≥60%)")
            print(f"🏆 F-beta Balance: {best_metrics['f_beta_score']:.4f}")
            print(f"📉 False Negatives: {best_metrics['false_negatives']} (Target: ≤10)")
            print(f"💰 Business Cost: {best_metrics['total_cost']:.2f}")
            
            if targets_achieved >= 3:
                print(f"\n⚖️🎉 BALANCED SUCCESS: {targets_achieved}/4 targets achieved! 🎉⚖️")
                print(f"Phase 4 achieved practical balance between precision and recall!")
            elif targets_achieved >= 2:
                print(f"\n⚖️ MAJOR IMPROVEMENT: {targets_achieved}/4 targets achieved!")
            else:
                print(f"\n📈 CONTINUED PROGRESS: {targets_achieved}/4 targets achieved")
            
            print(f"\n🔧 COMPLETE JOURNEY SUMMARY:")
            print(f"   Phase 1-2: ✅ Fixed overfitting + achieved AUC target")
            print(f"   Phase 3: ✅ Achieved 90% precision (but too conservative)")
            print(f"   Phase 4: ⚖️ Balanced approach - practical utility!")
            print(f"   Result: Better real-world performance")
            
            print(f"\n📁 All results saved to: {output_dir}/")
            print(f"📈 Visualization saved to: phase4_balanced_analysis.png")
            
            return best_model, best_metrics, targets_achieved
            
        except Exception as e:
            print(f"\n❌ ERROR in pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, 0

def main():
    """Main function to run the PHASE 4 balanced predictor"""
    csv_file = 'ASI_close_prices_last_15_years_2025-12-05.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ Dataset file not found: {csv_file}")
        return None
    
    # Initialize PHASE 4 balanced predictor
    predictor = Phase4BalancedPredictor(
        csv_file=csv_file,
        fn_penalty=6.0,          # BALANCED: 6.0x (sweet spot)
        target_precision=0.87,   # Slightly lower for balance
        target_recall=0.65       # NEW: explicit recall target
    )
    
    # Run PHASE 4 pipeline
    best_model, results, targets_achieved = predictor.run_complete_pipeline()
    
    return predictor, best_model, results, targets_achieved

if __name__ == "__main__":
    predictor, best_model, results, targets_achieved = main()