#!/usr/bin/env python3
"""
ASI Market Downturn Severity Analysis System
Extends downturn prediction with comprehensive severity, duration, and recovery analysis
Author: Data Science Team
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
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

try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available for advanced statistics")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

class ComprehensiveDownturnAnalyzer:
    """
    Complete system for downturn prediction, severity analysis, and recovery forecasting
    """
    
    def __init__(self, csv_file):
        """Initialize the comprehensive analyzer"""
        print("="*80)
        print("ASI COMPREHENSIVE DOWNTURN SEVERITY ANALYSIS SYSTEM")
        print("="*80)
        print("Features:")
        print("✓ Downturn Prediction (existing)")
        print("✓ Severity Analysis (how much further will it drop?)")
        print("✓ Duration Prediction (days to trough)")
        print("✓ Recovery Analysis (will it recover & when?)")
        print("✓ Statistical Summary Tables")
        print("✓ Business Impact Metrics")
        print("="*80)
        
        self.df = pd.read_csv(csv_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.df['close'] = pd.to_numeric(self.df['close'])
        self.df['returns'] = self.df['close'].pct_change()
        
        print(f"\nLoaded {len(self.df)} records from {self.df['date'].min()} to {self.df['date'].max()}")
        
        # Initialize model containers
        self.classification_models = {}
        self.severity_models = {}
        self.duration_models = {}
        self.recovery_models = {}
        
    def create_export_directory(self):
        """Create directory structure for exported datasets"""
        export_dir = './exported_datasets'
        os.makedirs(export_dir, exist_ok=True)
        os.makedirs(os.path.join(export_dir, 'classification'), exist_ok=True)
        os.makedirs(os.path.join(export_dir, 'severity'), exist_ok=True)
        os.makedirs(os.path.join(export_dir, 'features'), exist_ok=True)
        return export_dir
    
    def export_complete_feature_dataset(self, export_dir):
        """Export the complete dataset with all computed features"""
        
        # Save the main dataframe with all features
        feature_dataset_path = os.path.join(export_dir, 'features', 'complete_feature_dataset.csv')
        self.df.to_csv(feature_dataset_path, index=False)
        print(f"✓ Complete feature dataset saved: {feature_dataset_path}")
        
        # Save feature names list
        feature_names_path = os.path.join(export_dir, 'features', 'feature_names.json')
        with open(feature_names_path, 'w') as f:
            json.dump({
                'all_features': list(self.df.columns),
                'selected_features': self.feature_names,
                'feature_count': len(self.feature_names)
            }, f, indent=2)
        print(f"✓ Feature names saved: {feature_names_path}")
        
    def export_classification_data(self, export_dir):
        """Export classification training datasets"""
        
        if not hasattr(self, 'classification_data'):
            print("No classification data available to export")
            return
        
        class_dir = os.path.join(export_dir, 'classification')
        
        # Training set
        X_train_path = os.path.join(class_dir, 'X_train_classification.csv')
        y_train_path = os.path.join(class_dir, 'y_train_classification.csv')
        
        self.classification_data['X_train'].to_csv(X_train_path, index=False)
        self.classification_data['y_train'].to_csv(y_train_path, index=False, header=['will_have_downturn'])
        
        # Validation set
        X_val_path = os.path.join(class_dir, 'X_val_classification.csv')
        y_val_path = os.path.join(class_dir, 'y_val_classification.csv')
        
        self.classification_data['X_val'].to_csv(X_val_path, index=False)
        self.classification_data['y_val'].to_csv(y_val_path, index=False, header=['will_have_downturn'])
        
        # Test set
        X_test_path = os.path.join(class_dir, 'X_test_classification.csv')
        y_test_path = os.path.join(class_dir, 'y_test_classification.csv')
        
        self.classification_data['X_test'].to_csv(X_test_path, index=False)
        self.classification_data['y_test'].to_csv(y_test_path, index=False, header=['will_have_downturn'])
        
        print(f"✓ Classification training data saved to: {class_dir}")
        print(f"  - Training samples: {len(self.classification_data['X_train'])}")
        print(f"  - Validation samples: {len(self.classification_data['X_val'])}")
        print(f"  - Test samples: {len(self.classification_data['X_test'])}")
    
    def export_severity_data(self, export_dir):
        """Export severity regression training datasets"""
        
        if not hasattr(self, 'severity_data'):
            print("No severity data available to export")
            return
        
        sev_dir = os.path.join(export_dir, 'severity')
        
        # Training set
        X_train_path = os.path.join(sev_dir, 'X_train_severity.csv')
        y_train_path = os.path.join(sev_dir, 'y_train_severity.csv')
        
        self.severity_data['X_train'].to_csv(X_train_path, index=False)
        self.severity_data['y_train'].to_csv(y_train_path, index=False)
        
        # Validation set
        X_val_path = os.path.join(sev_dir, 'X_val_severity.csv')
        y_val_path = os.path.join(sev_dir, 'y_val_severity.csv')
        
        self.severity_data['X_val'].to_csv(X_val_path, index=False)
        self.severity_data['y_val'].to_csv(y_val_path, index=False)
        
        # Test set
        X_test_path = os.path.join(sev_dir, 'X_test_severity.csv')
        y_test_path = os.path.join(sev_dir, 'y_test_severity.csv')
        
        self.severity_data['X_test'].to_csv(X_test_path, index=False)
        self.severity_data['y_test'].to_csv(y_test_path, index=False)
        
        print(f"✓ Severity training data saved to: {sev_dir}")
        print(f"  - Training samples: {len(self.severity_data['X_train'])}")
        print(f"  - Validation samples: {len(self.severity_data['X_val'])}")
        print(f"  - Test samples: {len(self.severity_data['X_test'])}")
    
    def export_historical_downturns(self, export_dir):
        """Export historical downturns analysis"""
        
        if not hasattr(self, 'historical_downturns'):
            print("No historical downturns data available to export")
            return
        
        # Save historical downturns analysis
        downturns_path = os.path.join(export_dir, 'historical_downturns_complete.csv')
        self.historical_downturns.to_csv(downturns_path, index=False)
        
        print(f"✓ Historical downturns analysis saved: {downturns_path}")
        print(f"  - Total downturns: {len(self.historical_downturns)}")
    
    def create_dataset_metadata(self, export_dir):
        """Create comprehensive metadata for exported datasets"""
        
        metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'original_data_range': {
                'start_date': str(self.df['date'].min()),
                'end_date': str(self.df['date'].max()),
                'total_records': len(self.df)
            },
            'feature_engineering': {
                'total_features_computed': len(self.df.columns),
                'selected_features_for_training': len(self.feature_names),
                'feature_categories': {
                    'technical_indicators': 21,
                    'severity_specific': 8,
                    'derived_metrics': len(self.feature_names) - 29
                }
            },
            'classification_data': {},
            'severity_data': {},
            'historical_analysis': {}
        }
        
        # Add classification metadata
        if hasattr(self, 'classification_data'):
            pos_samples = self.classification_data['y_train'].sum()
            neg_samples = len(self.classification_data['y_train']) - pos_samples
            
            metadata['classification_data'] = {
                'total_samples': len(self.classification_data['X_train']) + len(self.classification_data['X_val']) + len(self.classification_data['X_test']),
                'positive_samples': int(pos_samples),
                'negative_samples': int(neg_samples),
                'class_ratio': f"1:{neg_samples/pos_samples:.1f}",
                'train_samples': len(self.classification_data['X_train']),
                'val_samples': len(self.classification_data['X_val']),
                'test_samples': len(self.classification_data['X_test'])
            }
        
        # Add severity metadata
        if hasattr(self, 'severity_data'):
            metadata['severity_data'] = {
                'total_samples': len(self.severity_data['X_train']) + len(self.severity_data['X_val']) + len(self.severity_data['X_test']),
                'train_samples': len(self.severity_data['X_train']),
                'val_samples': len(self.severity_data['X_val']),
                'test_samples': len(self.severity_data['X_test']),
                'targets': ['additional_severity', 'days_to_trough']
            }
        
        # Add historical analysis metadata
        if hasattr(self, 'historical_downturns'):
            metadata['historical_analysis'] = {
                'total_downturns_identified': len(self.historical_downturns),
                'recovery_rate': f"{self.historical_downturns['recovered'].mean()*100:.1f}%",
                'avg_severity': f"{self.historical_downturns['max_drawdown_pct'].mean():.2f}%",
                'avg_duration': f"{self.historical_downturns['days_to_trough'].mean():.1f} days"
            }
        
        # Save metadata
        metadata_path = os.path.join(export_dir, 'dataset_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✓ Dataset metadata saved: {metadata_path}")
    
    def export_all_training_datasets(self):
        """Main function to export all training datasets"""
        print("\n" + "="*60)
        print("EXPORTING ALL TRAINING DATASETS")
        print("="*60)
        
        # Create export directory
        export_dir = self.create_export_directory()
        
        # Export all datasets
        self.export_complete_feature_dataset(export_dir)
        self.export_classification_data(export_dir)
        self.export_severity_data(export_dir)
        self.export_historical_downturns(export_dir)
        self.create_dataset_metadata(export_dir)
        
        
        print(f"\n✓ All datasets exported to: {export_dir}")
        return export_dir
    
    def create_export_summary(self, export_dir):
        """Create a summary report of exported datasets"""
        
        summary_lines = [
            "# ASI Downturn Analysis - Exported Datasets Summary",
            f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Directory Structure:",
            "```",
            "exported_datasets/",
            "├── features/",
            "│   ├── complete_feature_dataset.csv",
            "│   └── feature_names.json",
            "├── classification/",
            "│   ├── X_train_classification.csv",
            "│   ├── y_train_classification.csv",
            "│   ├── X_val_classification.csv",
            "│   ├── y_val_classification.csv",
            "│   ├── X_test_classification.csv",
            "│   └── y_test_classification.csv",
            "├── severity/",
            "│   ├── X_train_severity.csv",
            "│   ├── y_train_severity.csv",
            "│   ├── X_val_severity.csv",
            "│   ├── y_val_severity.csv",
            "│   ├── X_test_severity.csv",
            "│   └── y_test_severity.csv",
            "├── historical_downturns_complete.csv",
            "├── dataset_metadata.json",
            "└── export_summary.md",
            "```",
            "",
            "## Dataset Descriptions:",
            "",
            "### 1. Feature Dataset (`features/`)",
            "- **complete_feature_dataset.csv**: Full dataset with all 25+ computed features",
            "- **feature_names.json**: List of selected features for training",
            "",
            "### 2. Classification Data (`classification/`)",
            "- **Purpose**: Binary prediction - Will -5% downturn occur?",
            "- **X_files**: Feature matrices (train/val/test splits)",
            "- **y_files**: Binary targets (True/False for downturn occurrence)",
            "",
            "### 3. Severity Data (`severity/`)",
            "- **Purpose**: Regression - Additional severity beyond -5% + Days to trough",
            "- **X_files**: Feature matrices (train/val/test splits)",
            "- **y_files**: Multi-target regression (additional_severity, days_to_trough)",
            "",
            "### 4. Historical Analysis",
            "- **historical_downturns_complete.csv**: Complete analysis of all identified downturns",
            "- **Columns**: start_date, severity metrics, duration metrics, recovery analysis",
            "",
            "## Usage Instructions:",
            "",
            "### Loading Classification Data:",
            "```python",
            "import pandas as pd",
            "X_train = pd.read_csv('classification/X_train_classification.csv')",
            "y_train = pd.read_csv('classification/y_train_classification.csv')",
            "```",
            "",
            "### Loading Severity Data:",
            "```python",
            "X_train = pd.read_csv('severity/X_train_severity.csv')",
            "y_train = pd.read_csv('severity/y_train_severity.csv')",
            "```",
            "",
            "### Loading Complete Feature Dataset:",
            "```python",
            "df = pd.read_csv('features/complete_feature_dataset.csv')",
            "df['date'] = pd.to_datetime(df['date'])",
            "```"
        ]
        
        # Add dataset statistics
        if hasattr(self, 'classification_data'):
            summary_lines.extend([
                "",
                "## Dataset Statistics:",
                f"- **Total Records**: {len(self.df):,}",
                f"- **Classification Samples**: {len(self.classification_data['X_train']) + len(self.classification_data['X_val']) + len(self.classification_data['X_test']):,}",
            ])
            
            if hasattr(self, 'severity_data'):
                summary_lines.append(f"- **Severity Samples**: {len(self.severity_data['X_train']) + len(self.severity_data['X_val']) + len(self.severity_data['X_test']):,}")
            
            if hasattr(self, 'historical_downturns'):
                summary_lines.append(f"- **Historical Downturns**: {len(self.historical_downturns):,}")
        
        # Save summary
        summary_path = os.path.join(export_dir, 'export_summary.md')
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"✓ Export summary created: {summary_path}")
    
    
    def compute_enhanced_features(self):
        """Compute enhanced technical indicators for severity analysis"""
        print("\nComputing enhanced technical indicators...")
        
        # Basic features (from your existing model)
        self._compute_basic_features()
        
        # Enhanced features for severity analysis
        self._compute_severity_features()
        
        print("Enhanced features computed.")
    
    def _compute_basic_features(self):
        """Basic technical indicators"""
        # Moving Averages
        for period in [10, 20, 50, 100, 200]:
            self.df[f'sma_{period}'] = self.df['close'].rolling(period).mean()
            
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
        for period in [10, 20, 60]:
            self.df[f'volatility_{period}'] = self.df['returns'].rolling(period).std() * np.sqrt(252)
        
        # Momentum & Returns
        for period in [1, 3, 5, 7, 10, 20, 50]:
            self.df[f'return_{period}d'] = self.df['close'].pct_change(period)
            self.df[f'momentum_{period}'] = self.df['close'].pct_change(period)
        
        # Price relative to moving averages
        for period in [20, 50, 200]:
            self.df[f'price_vs_sma{period}'] = (self.df['close'] - self.df[f'sma_{period}']) / self.df[f'sma_{period}']
        
        # Moving average relationships
        self.df['sma20_vs_sma50'] = (self.df['sma_20'] - self.df['sma_50']) / self.df['sma_50']
        self.df['sma50_vs_sma200'] = (self.df['sma_50'] - self.df['sma_200']) / self.df['sma_200']
        
    def _compute_severity_features(self):
        """Enhanced features specifically for severity analysis"""
        print("Computing severity-specific features...")
        
        # Volatility regime
        self.df['vol_regime'] = self.df['volatility_20'].rolling(252).rank(pct=True)
        
        # Volume surge (if volume data available)
        # For now, use price-based volume proxy
        self.df['price_velocity'] = self.df['close'].diff().abs().rolling(5).mean()
        self.df['volume_proxy'] = self.df['price_velocity'].rolling(20).rank(pct=True)
        
        # Market stress indicators
        self.df['drawdown_velocity'] = self.df['return_1d'].rolling(5).min()
        self.df['consecutive_down_days'] = (self.df['return_1d'] < 0).astype(int)
        for i in range(1, len(self.df)):
            if self.df.loc[i, 'return_1d'] >= 0:
                self.df.loc[i, 'consecutive_down_days'] = 0
            else:
                self.df.loc[i, 'consecutive_down_days'] = self.df.loc[i-1, 'consecutive_down_days'] + 1
        
        # Fear indicators
        self.df['fear_index'] = (
            (self.df['rsi_14'] < 30).astype(int) * 0.3 +
            (self.df['volatility_20'] > self.df['volatility_20'].rolling(252).quantile(0.8)).astype(int) * 0.4 +
            (self.df['consecutive_down_days'] > 3).astype(int) * 0.3
        )
        
        # Support/Resistance levels
        for period in [20, 60, 120, 252]:
            peak_period = self.df['close'].rolling(period).max()
            trough_period = self.df['close'].rolling(period).min()
            self.df[f'local_dd_{period}'] = (self.df['close'] - peak_period) / peak_period
            self.df[f'local_recovery_{period}'] = (self.df['close'] - trough_period) / trough_period
        
        # Trend strength
        self.df['trend_strength'] = np.abs(self.df['price_vs_sma50'])
        
        # Momentum acceleration
        self.df['momentum_accel'] = self.df['momentum_10'].diff()
        
    def identify_historical_downturns(self, threshold=-0.05, min_days=5, max_lookforward=30):
        """
        Identify all historical downturns with detailed analysis
        Enhanced to include severity and recovery metrics
        """
        print(f"\nIdentifying historical downturns (threshold: {threshold*100:.0f}%)...")
        
        downturns = []
        i = 0
        
        while i < len(self.df) - max_lookforward:
            current_price = self.df.loc[i, 'close']
            current_date = self.df.loc[i, 'date']
            
            # Look forward to find if -5% threshold is breached
            downturn_triggered = False
            trigger_day = None
            
            for j in range(i + 1, min(i + max_lookforward + 1, len(self.df))):
                future_price = self.df.loc[j, 'close']
                drop = (future_price - current_price) / current_price
                
                if drop <= threshold:
                    downturn_triggered = True
                    trigger_day = j
                    break
            
            if downturn_triggered:
                # Analyze the complete downturn
                downturn_analysis = self._analyze_complete_downturn(i, trigger_day)
                if downturn_analysis:
                    downturns.append(downturn_analysis)
                
                # Skip ahead to avoid overlapping downturns
                i = downturn_analysis.get('recovery_end_idx', trigger_day + 10)
            else:
                i += 1
        
        self.historical_downturns = pd.DataFrame(downturns)
        print(f"Identified {len(downturns)} historical downturns")
        
        if len(downturns) > 0:
            print(f"Average additional drop: {self.historical_downturns['additional_drop_pct'].mean():.2f}%")
            print(f"Average days to trough: {self.historical_downturns['days_to_trough'].mean():.1f}")
            print(f"Recovery rate: {self.historical_downturns['recovered'].mean()*100:.1f}%")
        
        return self.historical_downturns
    
    def _analyze_complete_downturn(self, start_idx, trigger_idx, recovery_threshold=0.95):
        """Analyze complete downturn cycle with all metrics"""
        start_price = self.df.loc[start_idx, 'close']
        start_date = self.df.loc[start_idx, 'date']
        trigger_price = self.df.loc[trigger_idx, 'close']
        trigger_date = self.df.loc[trigger_idx, 'date']
        
        # Find the trough (lowest point)
        trough_idx = trigger_idx
        trough_price = trigger_price
        
        # Look for the actual bottom (up to 100 days from trigger)
        search_end = min(trigger_idx + 100, len(self.df))
        for j in range(trigger_idx, search_end):
            if self.df.loc[j, 'close'] < trough_price:
                trough_price = self.df.loc[j, 'close']
                trough_idx = j
        
        # Calculate severity metrics
        initial_drop = (trigger_price - start_price) / start_price
        additional_drop = (trough_price - trigger_price) / trigger_price
        total_drop = (trough_price - start_price) / start_price
        
        # Find recovery (if any)
        recovery_target = start_price * recovery_threshold
        recovery_idx = None
        recovered = False
        
        # Look for recovery up to 300 days
        recovery_search_end = min(trough_idx + 300, len(self.df))
        for j in range(trough_idx + 1, recovery_search_end):
            if self.df.loc[j, 'close'] >= recovery_target:
                recovery_idx = j
                recovered = True
                break
        
        # Time metrics
        days_to_trigger = (trigger_date - start_date).days
        days_to_trough = (self.df.loc[trough_idx, 'date'] - start_date).days
        
        if recovered:
            days_to_recovery = (self.df.loc[recovery_idx, 'date'] - start_date).days
            total_duration = days_to_recovery
        else:
            days_to_recovery = None
            total_duration = (self.df.loc[recovery_search_end-1, 'date'] - start_date).days
        
        return {
            'start_idx': start_idx,
            'start_date': start_date,
            'start_price': start_price,
            'trigger_idx': trigger_idx,
            'trigger_date': trigger_date,
            'trigger_price': trigger_price,
            'trough_idx': trough_idx,
            'trough_date': self.df.loc[trough_idx, 'date'],
            'trough_price': trough_price,
            'recovery_idx': recovery_idx,
            'recovery_end_idx': recovery_idx if recovery_idx else recovery_search_end-1,
            'initial_drop_pct': initial_drop * 100,
            'additional_drop_pct': additional_drop * 100,
            'total_drop_pct': total_drop * 100,
            'max_drawdown_pct': total_drop * 100,
            'days_to_trigger': days_to_trigger,
            'days_to_trough': days_to_trough,
            'days_to_recovery': days_to_recovery,
            'total_duration': total_duration,
            'recovered': recovered,
            'recovery_rate': (self.df.loc[recovery_idx, 'close'] / start_price - 1) * 100 if recovered else None
        }
    
    def create_severity_labels(self):
        """Create labels for severity analysis training"""
        print("\nCreating labels for severity analysis...")
        
        # Create binary downturn labels (your existing approach)
        self.df['will_have_downturn'] = False
        prediction_days = 10
        
        for i in range(len(self.df) - prediction_days):
            current_price = self.df.loc[i, 'close']
            
            # Check next 10 days for -5% drop
            for j in range(i + 1, i + prediction_days + 1):
                if j >= len(self.df):
                    break
                future_price = self.df.loc[j, 'close']
                drop = (future_price - current_price) / current_price
                
                if drop <= -0.05:
                    self.df.loc[i, 'will_have_downturn'] = True
                    break
        
        # Create severity and duration labels
        self.df['additional_severity'] = np.nan
        self.df['days_to_trough'] = np.nan
        self.df['will_recover'] = np.nan
        self.df['days_to_recovery'] = np.nan
        
        # Map historical downturns to create training labels
        for _, downturn in self.historical_downturns.iterrows():
            start_idx = downturn['start_idx']
            
            # Only label if within our prediction window
            if start_idx < len(self.df) - 10:
                self.df.loc[start_idx, 'additional_severity'] = downturn['additional_drop_pct'] / 100
                self.df.loc[start_idx, 'days_to_trough'] = downturn['days_to_trough']
                self.df.loc[start_idx, 'will_recover'] = downturn['recovered']
                
                if downturn['recovered']:
                    self.df.loc[start_idx, 'days_to_recovery'] = downturn['days_to_recovery']
        
        print(f"Created severity labels for {self.df['additional_severity'].notna().sum()} downturns")
        
    def prepare_severity_data(self, test_size=0.2, val_size=0.2):
        """Prepare data for severity analysis"""
        print("\nPreparing data for comprehensive analysis...")
        
        # Feature columns
        feature_cols = [
            # Technical indicators
            'rsi_14', 'rsi_7', 'macd', 'macd_hist',
            'momentum_10', 'momentum_20', 'momentum_50',
            'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
            'sma20_vs_sma50', 'sma50_vs_sma200',
            'volatility_10', 'volatility_20', 'volatility_60',
            'local_dd_20', 'local_dd_60', 'local_dd_120',
            'return_1d', 'return_3d', 'return_5d', 'return_7d', 'return_10d',
            
            # Severity-specific features
            'vol_regime', 'volume_proxy', 'drawdown_velocity',
            'consecutive_down_days', 'fear_index', 'trend_strength',
            'momentum_accel', 'local_recovery_20'
        ]
        
        # Classification data (will downturn occur?)
        downturn_data = self.df[self.df['will_have_downturn'].notna()].copy()
        available_features = [col for col in feature_cols if col in downturn_data.columns]
        
        X_classification = downturn_data[available_features].copy()
        y_classification = downturn_data['will_have_downturn'].copy()
        dates_classification = downturn_data['date'].copy()
        
        # Remove rows with NaN
        valid_mask = ~X_classification.isna().any(axis=1)
        X_classification = X_classification[valid_mask]
        y_classification = y_classification[valid_mask]
        dates_classification = dates_classification[valid_mask]
        
        # Severity data (only downturns with severity labels)
        severity_data = self.df[self.df['additional_severity'].notna()].copy()
        X_severity = severity_data[available_features].copy()
        y_severity = severity_data[['additional_severity', 'days_to_trough']].copy()
        dates_severity = severity_data['date'].copy()
        
        # Remove rows with NaN
        valid_mask = ~X_severity.isna().any(axis=1) & ~y_severity.isna().any(axis=1)
        X_severity = X_severity[valid_mask]
        y_severity = y_severity[valid_mask]
        dates_severity = dates_severity[valid_mask]
        
        # Recovery data (only recovered downturns)
        recovery_data = self.df[
            (self.df['will_recover'].notna()) & 
            (self.df['days_to_recovery'].notna())
        ].copy()
        
        if len(recovery_data) > 0:
            X_recovery = recovery_data[available_features].copy()
            y_recovery = recovery_data[['will_recover', 'days_to_recovery']].copy()
            dates_recovery = recovery_data['date'].copy()
            
            valid_mask = ~X_recovery.isna().any(axis=1) & ~y_recovery.isna().any(axis=1)
            X_recovery = X_recovery[valid_mask]
            y_recovery = y_recovery[valid_mask]
            dates_recovery = dates_recovery[valid_mask]
        else:
            X_recovery = pd.DataFrame()
            y_recovery = pd.DataFrame()
            dates_recovery = pd.Series()
        
        print(f"Classification data: {len(X_classification)} samples")
        print(f"Severity data: {len(X_severity)} samples")
        print(f"Recovery data: {len(X_recovery)} samples")
        
        # Store data splits
        self.feature_names = available_features
        
        # Split classification data
        if len(X_classification) > 0:
            X_class_train, X_class_temp, y_class_train, y_class_temp = train_test_split(
                X_classification, y_classification, 
                test_size=test_size + val_size, random_state=42, stratify=y_classification
            )
            
            val_ratio = val_size / (test_size + val_size)
            X_class_val, X_class_test, y_class_val, y_class_test = train_test_split(
                X_class_temp, y_class_temp,
                test_size=1-val_ratio, random_state=42, stratify=y_class_temp
            )
            
            self.classification_data = {
                'X_train': X_class_train, 'X_val': X_class_val, 'X_test': X_class_test,
                'y_train': y_class_train, 'y_val': y_class_val, 'y_test': y_class_test
            }
        
        # Split severity data
        if len(X_severity) > 10:  # Need minimum samples
            X_sev_train, X_sev_temp, y_sev_train, y_sev_temp = train_test_split(
                X_severity, y_severity, test_size=test_size + val_size, random_state=42
            )
            
            if len(X_sev_temp) > 4:
                val_ratio = val_size / (test_size + val_size)
                X_sev_val, X_sev_test, y_sev_val, y_sev_test = train_test_split(
                    X_sev_temp, y_sev_temp, test_size=1-val_ratio, random_state=42
                )
            else:
                X_sev_val, y_sev_val = X_sev_temp, y_sev_temp
                X_sev_test, y_sev_test = X_sev_temp, y_sev_temp
            
            self.severity_data = {
                'X_train': X_sev_train, 'X_val': X_sev_val, 'X_test': X_sev_test,
                'y_train': y_sev_train, 'y_val': y_sev_val, 'y_test': y_sev_test
            }
        
        return True
    
    def train_classification_models(self):
        """Train downturn classification models (your existing models)"""
        print("\n" + "="*50)
        print("TRAINING CLASSIFICATION MODELS")
        print("="*50)
        
        if not hasattr(self, 'classification_data'):
            print("No classification data available")
            return
        
        X_train = self.classification_data['X_train']
        X_val = self.classification_data['X_val']
        y_train = self.classification_data['y_train']
        y_val = self.classification_data['y_val']
        
        models = {}
        
        # Logistic Regression
        print("\nTraining Logistic Regression...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        lr_model.fit(X_train_scaled, y_train)
        
        models['Logistic Regression'] = {
            'model': lr_model,
            'scaler': scaler,
            'val_score': lr_model.score(X_val_scaled, y_val)
        }
        
        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        models['Random Forest'] = {
            'model': rf_model,
            'scaler': None,
            'val_score': rf_model.score(X_val, y_val)
        }
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            print("Training XGBoost...")
            scale_pos_weight = (y_train == False).sum() / (y_train == True).sum()
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                scale_pos_weight=scale_pos_weight, random_state=42
            )
            xgb_model.fit(X_train, y_train)
            
            models['XGBoost'] = {
                'model': xgb_model,
                'scaler': None,
                'val_score': xgb_model.score(X_val, y_val)
            }
        
        self.classification_models = models
        
        # Select best model
        best_model_name = max(models.keys(), key=lambda x: models[x]['val_score'])
        self.best_classification_model = best_model_name
        
        print(f"\nBest classification model: {best_model_name}")
        for name, model_data in models.items():
            print(f"{name}: {model_data['val_score']:.4f}")
        
    def train_severity_models(self):
        """Train severity prediction models"""
        print("\n" + "="*50)
        print("TRAINING SEVERITY MODELS")
        print("="*50)
        
        if not hasattr(self, 'severity_data'):
            print("No severity data available")
            return
        
        X_train = self.severity_data['X_train']
        X_val = self.severity_data['X_val']
        y_train = self.severity_data['y_train']
        y_val = self.severity_data['y_val']
        
        print(f"Training on {len(X_train)} severity samples...")
        
        models = {}
        
        # Multi-output XGBoost Regressor
        if XGBOOST_AVAILABLE:
            print("\nTraining Multi-output XGBoost...")
            
            # Separate models for each target
            severity_model = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )
            duration_model = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )
            
            # Train separate models
            severity_model.fit(X_train, y_train['additional_severity'])
            duration_model.fit(X_train, y_train['days_to_trough'])
            
            # Evaluate
            sev_pred = severity_model.predict(X_val)
            dur_pred = duration_model.predict(X_val)
            
            sev_mae = mean_absolute_error(y_val['additional_severity'], sev_pred)
            dur_mae = mean_absolute_error(y_val['days_to_trough'], dur_pred)
            
            models['XGBoost Multi-output'] = {
                'severity_model': severity_model,
                'duration_model': duration_model,
                'scaler': None,
                'severity_mae': sev_mae,
                'duration_mae': dur_mae,
                'combined_score': -(sev_mae + dur_mae/100)  # Normalized combined score
            }
            
            print(f"Severity MAE: {sev_mae:.4f}, Duration MAE: {dur_mae:.1f} days")
        
        # Random Forest Multi-output
        print("\nTraining Random Forest Multi-output...")
        
        rf_multi = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        )
        rf_multi.fit(X_train, y_train)
        
        y_pred = rf_multi.predict(X_val)
        sev_mae = mean_absolute_error(y_val['additional_severity'], y_pred[:, 0])
        dur_mae = mean_absolute_error(y_val['days_to_trough'], y_pred[:, 1])
        
        models['Random Forest Multi-output'] = {
            'model': rf_multi,
            'scaler': None,
            'severity_mae': sev_mae,
            'duration_mae': dur_mae,
            'combined_score': -(sev_mae + dur_mae/100)
        }
        
        print(f"Severity MAE: {sev_mae:.4f}, Duration MAE: {dur_mae:.1f} days")
        
        # Quantile Regression (if enough data)
        if len(X_train) > 20:
            print("\nTraining Quantile Regression...")
            
            from sklearn.ensemble import GradientBoostingRegressor
            
            quantile_models = {}
            for quantile in [0.25, 0.5, 0.75]:
                model = GradientBoostingRegressor(
                    loss='quantile', alpha=quantile,
                    n_estimators=100, max_depth=6, random_state=42
                )
                model.fit(X_train, y_train['additional_severity'])
                quantile_models[f'q{int(quantile*100)}'] = model
            
            # Evaluate median quantile
            median_pred = quantile_models['q50'].predict(X_val)
            median_mae = mean_absolute_error(y_val['additional_severity'], median_pred)
            
            models['Quantile Regression'] = {
                'quantile_models': quantile_models,
                'scaler': None,
                'severity_mae': median_mae,
                'combined_score': -median_mae
            }
            
            print(f"Median Quantile MAE: {median_mae:.4f}")
        
        self.severity_models = models
        
        if models:
            # Select best model
            best_model_name = max(models.keys(), key=lambda x: models[x]['combined_score'])
            self.best_severity_model = best_model_name
            print(f"\nBest severity model: {best_model_name}")
    
    def train_integrated_pipeline(self):
        """Train the complete integrated prediction pipeline"""
        print("\n" + "="*80)
        print("TRAINING INTEGRATED PREDICTION PIPELINE")
        print("="*80)
        
        # Train all model components
        self.train_classification_models()
        self.train_severity_models()
        
        print("\n✓ Integrated pipeline training complete!")
        print("Available predictions:")
        print("  1. Downturn probability (binary classification)")
        print("  2. Additional severity beyond -5% (regression)")
        print("  3. Days to trough (regression)")
        if hasattr(self, 'severity_models') and 'Quantile Regression' in self.severity_models:
            print("  4. Severity confidence intervals (quantile regression)")
    
    def predict_comprehensive(self, features_df, downturn_threshold=0.5):
        """
        Comprehensive prediction pipeline
        Returns: downturn probability, severity, duration predictions
        """
        if not hasattr(self, 'classification_models') or not self.classification_models:
            print("No trained models available")
            return None
        
        # Step 1: Downturn Classification
        best_class_model = self.classification_models[self.best_classification_model]
        class_model = best_class_model['model']
        class_scaler = best_class_model['scaler']
        
        if class_scaler:
            features_scaled = class_scaler.transform(features_df)
            downturn_prob = class_model.predict_proba(features_scaled)[:, 1]
        else:
            downturn_prob = class_model.predict_proba(features_df)[:, 1]
        
        results = {
            'downturn_probability': downturn_prob,
            'downturn_prediction': downturn_prob > downturn_threshold
        }
        
        # Step 2: Severity Analysis (only for predicted downturns)
        downturn_mask = downturn_prob > downturn_threshold
        
        if hasattr(self, 'severity_models') and self.severity_models and np.any(downturn_mask):
            best_sev_model = self.severity_models[self.best_severity_model]
            
            if 'XGBoost Multi-output' in self.best_severity_model:
                # Separate models
                sev_model = best_sev_model['severity_model']
                dur_model = best_sev_model['duration_model']
                
                additional_severity = np.zeros(len(features_df))
                days_to_trough = np.zeros(len(features_df))
                
                if np.any(downturn_mask):
                    additional_severity[downturn_mask] = sev_model.predict(features_df[downturn_mask])
                    days_to_trough[downturn_mask] = dur_model.predict(features_df[downturn_mask])
                    
            else:
                # Multi-output model
                sev_model = best_sev_model['model']
                predictions = np.zeros((len(features_df), 2))
                
                if np.any(downturn_mask):
                    predictions[downturn_mask] = sev_model.predict(features_df[downturn_mask])
                
                additional_severity = predictions[:, 0]
                days_to_trough = predictions[:, 1]
            
            results['additional_severity'] = additional_severity
            results['total_severity'] = -0.05 + additional_severity  # Add to initial -5%
            results['days_to_trough'] = days_to_trough
            
            # Quantile predictions if available
            if 'Quantile Regression' in self.severity_models:
                quantile_models = self.severity_models['Quantile Regression']['quantile_models']
                
                q25 = np.zeros(len(features_df))
                q50 = np.zeros(len(features_df))
                q75 = np.zeros(len(features_df))
                
                if np.any(downturn_mask):
                    q25[downturn_mask] = quantile_models['q25'].predict(features_df[downturn_mask])
                    q50[downturn_mask] = quantile_models['q50'].predict(features_df[downturn_mask])
                    q75[downturn_mask] = quantile_models['q75'].predict(features_df[downturn_mask])
                
                results['severity_q25'] = -0.05 + q25
                results['severity_q50'] = -0.05 + q50
                results['severity_q75'] = -0.05 + q75
        
        return results
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        evaluation_results = {}
        
        # Classification Evaluation
        if hasattr(self, 'classification_data') and self.classification_models:
            print("\nClassification Model Evaluation:")
            X_test = self.classification_data['X_test']
            y_test = self.classification_data['y_test']
            
            for name, model_data in self.classification_models.items():
                model = model_data['model']
                scaler = model_data['scaler']
                
                if scaler:
                    X_test_scaled = scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                evaluation_results[f'{name}_classification'] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc
                }
                
                print(f"  {name}:")
                print(f"    Accuracy: {accuracy:.4f}")
                print(f"    Precision: {precision:.4f}")
                print(f"    Recall: {recall:.4f}")
                print(f"    F1: {f1:.4f}")
                print(f"    AUC: {auc:.4f}")
        
        # Severity Evaluation
        if hasattr(self, 'severity_data') and self.severity_models:
            print("\nSeverity Model Evaluation:")
            X_test = self.severity_data['X_test']
            y_test = self.severity_data['y_test']
            
            for name, model_data in self.severity_models.items():
                print(f"  {name}:")
                print(f"    Severity MAE: {model_data['severity_mae']:.4f}")
                if 'duration_mae' in model_data:
                    print(f"    Duration MAE: {model_data['duration_mae']:.1f} days")
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def generate_statistical_summary(self):
        """Generate comprehensive statistical summary table"""
        print("\nGenerating statistical summary...")
        
        if not hasattr(self, 'historical_downturns') or len(self.historical_downturns) == 0:
            print("No historical downturns data available")
            return None
        
        def calculate_stats(series, is_percentage=False):
            """Calculate comprehensive statistics"""
            mean_val = series.mean()
            std_val = series.std()
            median_val = series.median()
            min_val = series.min()
            max_val = series.max()
            
            # Ranges
            range_1std_low = mean_val - std_val
            range_1std_high = mean_val + std_val
            range_95_low = series.quantile(0.025)
            range_95_high = series.quantile(0.975)
            
            if is_percentage:
                return {
                    'Average': f"{mean_val:.2f}%",
                    'Std Dev': f"{std_val:.2f}%",
                    'Median': f"{median_val:.2f}%",
                    'Min': f"{min_val:.2f}%",
                    'Max': f"{max_val:.2f}%",
                    'Range (±1 std)': f"{range_1std_low:.2f}% to {range_1std_high:.2f}%",
                    'Range (95%)': f"{range_95_low:.2f}% to {range_95_high:.2f}%"
                }
            else:
                return {
                    'Average': f"{mean_val:.1f}",
                    'Std Dev': f"{std_val:.1f}",
                    'Median': f"{median_val:.1f}",
                    'Min': f"{min_val:.0f}",
                    'Max': f"{max_val:.0f}",
                    'Range (±1 std)': f"{range_1std_low:.1f} to {range_1std_high:.1f} days",
                    'Range (95%)': f"{range_95_low:.1f} to {range_95_high:.1f} days"
                }
        
        # Calculate statistics for each metric
        summary_data = {}
        
        # Max Drawdown (total drop)
        summary_data['Max Drawdown'] = calculate_stats(
            self.historical_downturns['max_drawdown_pct'], is_percentage=True
        )
        
        # Days to Trough
        summary_data['Days to Trough'] = calculate_stats(
            self.historical_downturns['days_to_trough'], is_percentage=False
        )
        
        # Total Duration (for recovered downturns)
        recovered_downturns = self.historical_downturns[self.historical_downturns['recovered'] == True]
        if len(recovered_downturns) > 0:
            summary_data['Total Duration'] = calculate_stats(
                recovered_downturns['total_duration'], is_percentage=False
            )
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data).T
        
        # Add additional insights
        total_downturns = len(self.historical_downturns)
        recovery_rate = self.historical_downturns['recovered'].mean() * 100
        
        print("\n" + "="*100)
        print("COMPREHENSIVE DOWNTURN STATISTICS SUMMARY")
        print("="*100)
        print(summary_df.to_string())
        print("\n" + "="*100)
        print(f"Total Downturns Analyzed: {total_downturns}")
        print(f"Recovery Rate: {recovery_rate:.1f}% of cases recover")
        print(f"Average Additional Drop (beyond -5%): {self.historical_downturns['additional_drop_pct'].mean():.2f}%")
        print("="*100)
        
        # Save summary
        self.downturn_summary = summary_df
        return summary_df
    
    def plot_comprehensive_analysis(self, save_plots=True):
        """Create comprehensive visualization plots"""
        print("\nCreating comprehensive analysis plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # Create subplots
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Price chart with downturns
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(self.df['date'], self.df['close'], 'b-', alpha=0.7, linewidth=1)
        
        # Mark historical downturns
        if hasattr(self, 'historical_downturns') and len(self.historical_downturns) > 0:
            for _, downturn in self.historical_downturns.iterrows():
                start_date = downturn['start_date']
                end_date = downturn['trough_date']
                ax1.axvspan(start_date, end_date, alpha=0.3, color='red')
        
        ax1.set_title('ASI Price History with Identified Downturns', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ASI Close Price')
        ax1.grid(True, alpha=0.3)
        
        # 2. Severity Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        if hasattr(self, 'historical_downturns') and len(self.historical_downturns) > 0:
            ax2.hist(self.historical_downturns['max_drawdown_pct'], bins=15, alpha=0.7, color='red', edgecolor='black')
            ax2.axvline(self.historical_downturns['max_drawdown_pct'].mean(), color='orange', linestyle='--', linewidth=2)
            ax2.set_title('Max Drawdown Distribution', fontsize=10, fontweight='bold')
            ax2.set_xlabel('Max Drawdown (%)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
        
        # 3. Duration Distribution
        ax3 = fig.add_subplot(gs[0, 3])
        if hasattr(self, 'historical_downturns') and len(self.historical_downturns) > 0:
            ax3.hist(self.historical_downturns['days_to_trough'], bins=15, alpha=0.7, color='blue', edgecolor='black')
            ax3.axvline(self.historical_downturns['days_to_trough'].mean(), color='orange', linestyle='--', linewidth=2)
            ax3.set_title('Days to Trough Distribution', fontsize=10, fontweight='bold')
            ax3.set_xlabel('Days to Trough')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # 4. Model Performance Comparison
        ax4 = fig.add_subplot(gs[1, :2])
        if hasattr(self, 'evaluation_results') and self.evaluation_results:
            model_names = []
            auc_scores = []
            
            for key, metrics in self.evaluation_results.items():
                if 'classification' in key and 'auc' in metrics:
                    model_names.append(key.replace('_classification', ''))
                    auc_scores.append(metrics['auc'])
            
            if model_names:
                bars = ax4.bar(model_names, auc_scores, color=['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)])
                ax4.set_title('Classification Model Performance (AUC)', fontsize=12, fontweight='bold')
                ax4.set_ylabel('AUC Score')
                ax4.set_ylim(0, 1)
                ax4.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, score in zip(bars, auc_scores):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Severity vs Duration Scatter
        ax5 = fig.add_subplot(gs[1, 2:])
        if hasattr(self, 'historical_downturns') and len(self.historical_downturns) > 0:
            scatter = ax5.scatter(
                self.historical_downturns['days_to_trough'], 
                self.historical_downturns['max_drawdown_pct'],
                c=self.historical_downturns['recovered'].astype(int),
                cmap='RdYlGn', alpha=0.7, s=50
            )
            ax5.set_title('Severity vs Duration (Color: Recovery)', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Days to Trough')
            ax5.set_ylabel('Max Drawdown (%)')
            ax5.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax5)
            cbar.set_label('Recovered (1) / Not Recovered (0)')
        
        # 6. Feature Importance (if available)
        ax6 = fig.add_subplot(gs[2, :2])
        if hasattr(self, 'classification_models') and self.classification_models:
            best_model_name = self.best_classification_model
            if best_model_name in self.classification_models:
                model = self.classification_models[best_model_name]['model']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1][:10]
                    
                    ax6.barh(range(len(indices)), importances[indices])
                    ax6.set_yticks(range(len(indices)))
                    ax6.set_yticklabels([self.feature_names[i] for i in indices])
                    ax6.set_title(f'Top 10 Feature Importance ({best_model_name})', fontsize=12, fontweight='bold')
                    ax6.set_xlabel('Importance')
                    ax6.grid(True, alpha=0.3)
        
        # 7. Recovery Analysis
        ax7 = fig.add_subplot(gs[2, 2:])
        if hasattr(self, 'historical_downturns') and len(self.historical_downturns) > 0:
            recovery_data = self.historical_downturns['recovered'].value_counts()
            
            ax7.pie(recovery_data.values, labels=['Recovered', 'Not Recovered'], 
                   autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
            ax7.set_title('Recovery Rate Analysis', fontsize=12, fontweight='bold')
        
        # 8. Time Series of Predictions (if test results available)
        ax8 = fig.add_subplot(gs[3, :])
        ax8.text(0.5, 0.5, 'Comprehensive ASI Downturn Analysis System\n\n' +
                           'This system provides:\n' +
                           '• Binary downturn prediction (will -5% drop occur?)\n' +
                           '• Severity analysis (how much further will it drop?)\n' +
                           '• Duration prediction (days to reach trough)\n' +
                           '• Recovery analysis (will it recover and when?)\n' +
                           '• Statistical confidence intervals\n' +
                           '• Business impact metrics',
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        ax8.set_xlim(0, 1)
        ax8.set_ylim(0, 1)
        ax8.axis('off')
        
        plt.suptitle('ASI Comprehensive Downturn Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_plots:
            plt.savefig('./risk_analysis/asi_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
            print("Comprehensive analysis plot saved to 'asi_comprehensive_analysis.png'")
        
        plt.show()
    
    def save_comprehensive_models(self, output_dir='./risk_analysis/comprehensive_models'):
        """Save all trained models and analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving comprehensive models to {output_dir}/...")
        
        # Save classification models
        if hasattr(self, 'classification_models'):
            for name, model_data in self.classification_models.items():
                filename = f"classification_{name.lower().replace(' ', '_')}.pkl"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    pickle.dump({
                        'model': model_data['model'],
                        'scaler': model_data['scaler'],
                        'feature_names': self.feature_names,
                        'model_type': 'classification',
                        'target': 'downturn_prediction'
                    }, f)
        
        # Save severity models
        if hasattr(self, 'severity_models'):
            for name, model_data in self.severity_models.items():
                filename = f"severity_{name.lower().replace(' ', '_').replace('-', '_')}.pkl"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    pickle.dump({
                        'model_data': model_data,
                        'feature_names': self.feature_names,
                        'model_type': 'severity',
                        'targets': ['additional_severity', 'days_to_trough']
                    }, f)
        
        # Save comprehensive pipeline
        pipeline_data = {
            'feature_names': self.feature_names,
            'best_classification_model': getattr(self, 'best_classification_model', None),
            'best_severity_model': getattr(self, 'best_severity_model', None),
            'historical_downturns_stats': {
                'total_downturns': len(getattr(self, 'historical_downturns', [])),
                'avg_additional_drop': getattr(self, 'historical_downturns', pd.DataFrame()).get('additional_drop_pct', pd.Series()).mean(),
                'avg_days_to_trough': getattr(self, 'historical_downturns', pd.DataFrame()).get('days_to_trough', pd.Series()).mean(),
                'recovery_rate': getattr(self, 'historical_downturns', pd.DataFrame()).get('recovered', pd.Series()).mean()
            },
            'training_date': datetime.now().isoformat(),
            'model_description': 'Comprehensive ASI downturn prediction with severity and duration analysis'
        }
        
        with open(os.path.join(output_dir, 'comprehensive_pipeline.json'), 'w') as f:
            json.dump(pipeline_data, f, indent=2, default=str)
        
        # Save historical downturns analysis
        if hasattr(self, 'historical_downturns'):
            self.historical_downturns.to_csv(
                os.path.join(output_dir, 'historical_downturns_analysis.csv'), 
                index=False
            )
        
        # Save statistical summary
        if hasattr(self, 'downturn_summary'):
            self.downturn_summary.to_csv(
                os.path.join(output_dir, 'downturn_statistical_summary.csv')
            )
        
        print("✓ All models and analysis results saved!")
        print(f"✓ Use load_comprehensive_model() to load the complete pipeline")
        
        return output_dir

def load_comprehensive_model(model_dir='./risk_analysis/comprehensive_models'):
    """Load a trained comprehensive model for predictions"""
    print(f"Loading comprehensive model from {model_dir}...")
    
    # Load pipeline info
    with open(os.path.join(model_dir, 'comprehensive_pipeline.json'), 'r') as f:
        pipeline_info = json.load(f)
    
    print("✓ Comprehensive ASI downturn prediction model loaded!")
    print(f"✓ Trained on: {pipeline_info['training_date']}")
    print(f"✓ Features: {len(pipeline_info['feature_names'])}")
    print(f"✓ Best classification model: {pipeline_info['best_classification_model']}")
    print(f"✓ Best severity model: {pipeline_info['best_severity_model']}")
    
    return pipeline_info

def main():
    """Main execution function"""
    print("Starting Comprehensive ASI Downturn Analysis...")
    
    # Initialize analyzer
    csv_file = 'ASI_close_prices_last_15_years_2025-06-05.csv'
    analyzer = ComprehensiveDownturnAnalyzer(csv_file)
    
    # Step 1: Compute enhanced features
    analyzer.compute_enhanced_features()
    
    # Step 2: Identify historical downturns with full analysis
    analyzer.identify_historical_downturns()
    
    # Step 3: Create severity and duration labels
    analyzer.create_severity_labels()
    
    # Step 4: Prepare data for training
    analyzer.prepare_severity_data()
    
    # Step 5: Prepare data for training
    analyzer.export_all_training_datasets()
    
    # Step 6: Train integrated pipeline
    analyzer.train_integrated_pipeline()
    
    # Step 7: Evaluate models
    analyzer.evaluate_models()
    
    # Step 8: Generate statistical summary
    analyzer.generate_statistical_summary()
    
    # Step 9: Create comprehensive visualizations
    analyzer.plot_comprehensive_analysis()
    
    # Step 10: Save everything
    analyzer.save_comprehensive_models()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ASI DOWNTURN ANALYSIS COMPLETE!")
    print("="*80)
    print("✓ Historical downturns identified and analyzed")
    print("✓ Classification models trained (downturn prediction)")
    print("✓ Severity models trained (additional drop prediction)")
    print("✓ Duration models trained (days to trough)")
    print("✓ Statistical summary generated")
    print("✓ Comprehensive visualizations created")
    print("✓ All models saved for production use")
    print("\nYour system can now predict:")
    print("• Will a -5% downturn occur? (Binary + Probability)")
    print("• How much further will it drop beyond -5%? (Regression)")
    print("• How many days until the trough? (Duration)")
    print("• Confidence intervals for severity (Quantile Regression)")
    print("• Business impact metrics and risk scenarios")
    print("="*80)

if __name__ == "__main__":
    main()