#!/usr/bin/env python3
"""
ASI Market Downturn Comprehensive Predictor API - Two-Stage ML Pipeline
- Stage 1: Classification (will -5% downturn occur?)
- Stage 2: Severity Analysis (additional drop + days to trough)
- PostgreSQL database integration
- Real-time data fetching with TradingView
- Comprehensive severity predictions with confidence intervals
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any
import logging
import asyncio
import warnings
import time
warnings.filterwarnings('ignore')

# FastAPI and database
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# PostgreSQL
import psycopg2
from psycopg2.extras import RealDictCursor
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Scheduling
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_ADMIN_USER")
DB_PASSWORD = os.getenv("DB_ADMIN_PASSWORD") 
DB_NAME = os.getenv("DB_NAME")

# Create PostgreSQL connection string
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Enhanced Pydantic models for comprehensive predictions
class SeverityPrediction(BaseModel):
    additional_severity: Optional[float] = None  # Additional drop beyond -5%
    total_severity: Optional[float] = None      # Total expected drop
    days_to_trough: Optional[float] = None      # Days to reach lowest point
    severity_q25: Optional[float] = None        # 25th percentile (conservative)
    severity_q50: Optional[float] = None        # 50th percentile (median)
    severity_q75: Optional[float] = None        # 75th percentile (severe)

class HistoricalPrediction(BaseModel):
    date: str
    asi_value: float
    prediction: bool
    confidence: float
    risk_level: str
    severity_prediction: Optional[SeverityPrediction] = None
    actual_change_10d: Optional[float] = None
    actual_asi_value_10d: Optional[float] = None

class ComprehensivePredictionResponse(BaseModel):
    date: str
    asi_value: float
    
    # Stage 1: Classification Results
    downturn_prediction: bool
    downturn_probability: float
    risk_level: str
    explanation: str
    
    # Stage 2: Severity Analysis (conditional)
    severity_analysis: Optional[SeverityPrediction] = None
    
    # Historical context
    past_10_days: List[HistoricalPrediction]

class BusinessImpactAnalysis(BaseModel):
    portfolio_scenarios: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    recommended_actions: List[str]

class TrendComparison(BaseModel):
    current_prediction: ComprehensivePredictionResponse
    trend_analysis: Dict[str, Any]
    recent_predictions: List[Dict[str, Any]]
    business_impact: BusinessImpactAnalysis

# Initialize FastAPI app
app = FastAPI(
    title="ASI Comprehensive Downturn Predictor - Two-Stage ML Pipeline",
    description="Predicts downturn probability + severity analysis using comprehensive ML approach",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for comprehensive models
classification_models = {}
severity_models = {}
feature_names = None
scheduler = None

# In-memory cache for ASI data
ASI_DATA_CACHE = {
    'data': None,
    'last_updated': None,
    'cache_duration': 3600,  # 1 hour in seconds
    'data_days': 300
}

class ComprehensiveFeatureCalculator:
    """Calculate comprehensive technical indicators for severity analysis"""
    
    @staticmethod
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)
    
    @staticmethod
    def compute_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute all 25+ features for comprehensive analysis"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df['close'] = pd.to_numeric(df['close'])
        df['returns'] = df['close'].pct_change()
        
        # Basic Technical Indicators
        # Moving Averages
        for period in [10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        df['rsi_14'] = ComprehensiveFeatureCalculator.calculate_rsi(df['close'], 14)
        df['rsi_7'] = ComprehensiveFeatureCalculator.calculate_rsi(df['close'], 7)
        
        # Volatility
        for period in [10, 20, 60]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std() * np.sqrt(252)
        
        # Momentum & Returns
        for period in [1, 3, 5, 7, 10, 20, 50]:
            df[f'return_{period}d'] = df['close'].pct_change(period)
            df[f'momentum_{period}'] = df['close'].pct_change(period)
        
        # Price relative to moving averages
        for period in [20, 50, 200]:
            df[f'price_vs_sma{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        
        # Moving average relationships
        df['sma20_vs_sma50'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
        df['sma50_vs_sma200'] = (df['sma_50'] - df['sma_200']) / df['sma_200']
        
        # Local drawdowns
        for w in [20, 60, 120]:
            peak_w = df['close'].rolling(w).max()
            df[f'local_dd_{w}'] = (df['close'] - peak_w) / peak_w
        
        # Confirmation signals
        df['below_sma50'] = (df['close'] < df['sma_50']).astype(int)
        df['negative_momentum_20'] = (df['momentum_20'] < 0).astype(int)
        df['rsi_below_50'] = (df['rsi_14'] < 50).astype(int)
        df['macd_bearish'] = (df['macd'] < df['macd_signal']).astype(int)
        
        # Enhanced Severity-Specific Features
        # Volatility regime
        df['vol_regime'] = df['volatility_20'].rolling(252).rank(pct=True)
        
        # Volume surge proxy (price-based)
        df['price_velocity'] = df['close'].diff().abs().rolling(5).mean()
        df['volume_proxy'] = df['price_velocity'].rolling(20).rank(pct=True)
        
        # Market stress indicators
        df['drawdown_velocity'] = df['return_1d'].rolling(5).min()
        df['consecutive_down_days'] = (df['return_1d'] < 0).astype(int)
        for i in range(1, len(df)):
            if df.loc[i, 'return_1d'] >= 0:
                df.loc[i, 'consecutive_down_days'] = 0
            else:
                df.loc[i, 'consecutive_down_days'] = df.loc[i-1, 'consecutive_down_days'] + 1
        
        # Fear indicators
        df['fear_index'] = (
            (df['rsi_14'] < 30).astype(int) * 0.3 +
            (df['volatility_20'] > df['volatility_20'].rolling(252).quantile(0.8)).astype(int) * 0.4 +
            (df['consecutive_down_days'] > 3).astype(int) * 0.3
        )
        
        # Support/Resistance levels
        for period in [20, 60, 120, 252]:
            peak_period = df['close'].rolling(period).max()
            trough_period = df['close'].rolling(period).min()
            df[f'local_recovery_{period}'] = (df['close'] - trough_period) / trough_period
        
        # Trend strength
        df['trend_strength'] = np.abs(df['price_vs_sma50'])
        
        # Momentum acceleration
        df['momentum_accel'] = df['momentum_10'].diff()
        
        return df

# PostgreSQL database functions for comprehensive predictions
def init_comprehensive_postgresql_db():
    """Initialize PostgreSQL database with comprehensive prediction schema"""
    try:
        with engine.connect() as conn:
            # Create main predictions table with severity analysis
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS comprehensive_predictions (
                    id SERIAL PRIMARY KEY,
                    date DATE UNIQUE NOT NULL,
                    asi_value DECIMAL(10, 4) NOT NULL,
                    
                    -- Stage 1: Classification Results
                    downturn_prediction BOOLEAN NOT NULL,
                    downturn_probability DECIMAL(5, 4) NOT NULL,
                    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('HIGH', 'MODERATE', 'LOW', 'STABLE')),
                    explanation TEXT NOT NULL,
                    
                    -- Stage 2: Severity Analysis (conditional)
                    has_severity_prediction BOOLEAN DEFAULT FALSE,
                    additional_severity DECIMAL(8, 4) DEFAULT NULL,
                    total_severity DECIMAL(8, 4) DEFAULT NULL,
                    days_to_trough DECIMAL(8, 2) DEFAULT NULL,
                    severity_q25 DECIMAL(8, 4) DEFAULT NULL,
                    severity_q50 DECIMAL(8, 4) DEFAULT NULL,
                    severity_q75 DECIMAL(8, 4) DEFAULT NULL,
                    
                    -- Actual outcomes tracking
                    actual_change_10d DECIMAL(8, 4) DEFAULT NULL,
                    actual_asi_value_10d DECIMAL(10, 4) DEFAULT NULL,
                    actual_days_to_trough INTEGER DEFAULT NULL,
                    actual_max_drawdown DECIMAL(8, 4) DEFAULT NULL,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            
            # Create indexes for performance
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_comp_pred_date ON comprehensive_predictions(date);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_comp_pred_risk ON comprehensive_predictions(risk_level);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_comp_pred_downturn ON comprehensive_predictions(downturn_prediction);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_comp_pred_severity ON comprehensive_predictions(has_severity_prediction);"))
            
            # Create trigger for updated_at
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """))
            
            conn.execute(text("""
                DROP TRIGGER IF EXISTS update_comp_predictions_updated_at ON comprehensive_predictions;
                CREATE TRIGGER update_comp_predictions_updated_at 
                    BEFORE UPDATE ON comprehensive_predictions 
                    FOR EACH ROW 
                    EXECUTE FUNCTION update_updated_at_column();
            """))
            
            conn.commit()
            logger.info("Comprehensive PostgreSQL database initialized successfully")
            return True
            
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return False

def store_comprehensive_prediction_postgresql(prediction_data):
    """Store comprehensive prediction in PostgreSQL"""
    try:
        with engine.connect() as conn:
            # Prepare severity data
            severity_analysis = prediction_data.get('severity_analysis')
            has_severity = severity_analysis is not None
            
            params = {
                'date': prediction_data['date'],
                'asi_value': prediction_data['asi_value'],
                'downturn_prediction': prediction_data['downturn_prediction'],
                'downturn_probability': prediction_data['downturn_probability'],
                'risk_level': prediction_data['risk_level'],
                'explanation': prediction_data['explanation'],
                'has_severity_prediction': has_severity
            }
            
            # Add severity parameters if available
            if has_severity:
                params.update({
                    'additional_severity': severity_analysis.get('additional_severity'),
                    'total_severity': severity_analysis.get('total_severity'),
                    'days_to_trough': severity_analysis.get('days_to_trough'),
                    'severity_q25': severity_analysis.get('severity_q25'),
                    'severity_q50': severity_analysis.get('severity_q50'),
                    'severity_q75': severity_analysis.get('severity_q75')
                })
            else:
                params.update({
                    'additional_severity': None,
                    'total_severity': None,
                    'days_to_trough': None,
                    'severity_q25': None,
                    'severity_q50': None,
                    'severity_q75': None
                })
            
            conn.execute(text("""
                INSERT INTO comprehensive_predictions 
                (date, asi_value, downturn_prediction, downturn_probability, risk_level, explanation,
                 has_severity_prediction, additional_severity, total_severity, days_to_trough,
                 severity_q25, severity_q50, severity_q75)
                VALUES (:date, :asi_value, :downturn_prediction, :downturn_probability, :risk_level, :explanation,
                        :has_severity_prediction, :additional_severity, :total_severity, :days_to_trough,
                        :severity_q25, :severity_q50, :severity_q75)
                ON CONFLICT (date) 
                DO UPDATE SET
                    asi_value = EXCLUDED.asi_value,
                    downturn_prediction = EXCLUDED.downturn_prediction,
                    downturn_probability = EXCLUDED.downturn_probability,
                    risk_level = EXCLUDED.risk_level,
                    explanation = EXCLUDED.explanation,
                    has_severity_prediction = EXCLUDED.has_severity_prediction,
                    additional_severity = EXCLUDED.additional_severity,
                    total_severity = EXCLUDED.total_severity,
                    days_to_trough = EXCLUDED.days_to_trough,
                    severity_q25 = EXCLUDED.severity_q25,
                    severity_q50 = EXCLUDED.severity_q50,
                    severity_q75 = EXCLUDED.severity_q75,
                    updated_at = CURRENT_TIMESTAMP
            """), params)
            conn.commit()
            logger.info(f"Comprehensive prediction stored for {prediction_data['date']}")
            
    except Exception as e:
        logger.error(f"Error storing comprehensive prediction: {e}")

def get_comprehensive_past_10_days_predictions(target_date: str, asi_data: pd.DataFrame) -> List[HistoricalPrediction]:
    """Get past 10 days of comprehensive predictions"""
    try:
        target_dt = datetime.strptime(target_date, '%Y-%m-%d').date()
        start_date = target_dt - timedelta(days=15)
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT date, asi_value, downturn_prediction, downturn_probability, risk_level,
                       has_severity_prediction, additional_severity, total_severity, days_to_trough,
                       severity_q25, severity_q50, severity_q75, actual_change_10d, actual_asi_value_10d
                FROM comprehensive_predictions 
                WHERE date < :target_date AND date >= :start_date
                ORDER BY date DESC 
                LIMIT 10
            """), {
                'target_date': target_date,
                'start_date': start_date.strftime('%Y-%m-%d')
            })
            
            results = result.fetchall()
        
        historical_predictions = []
        
        # Convert ASI data to dict for faster lookup
        asi_dict = {}
        if not asi_data.empty:
            asi_data_copy = asi_data.copy()
            asi_data_copy['date'] = pd.to_datetime(asi_data_copy['date']).dt.date
            for _, row in asi_data_copy.iterrows():
                date_key = row['date'].strftime('%Y-%m-%d')
                asi_dict[date_key] = float(row['close'])
        
        for row in results:
            pred_date = row[0].strftime('%Y-%m-%d')
            
            # Build severity prediction if available
            severity_pred = None
            if row[5]:  # has_severity_prediction
                severity_pred = SeverityPrediction(
                    additional_severity=float(row[6]) if row[6] is not None else None,
                    total_severity=float(row[7]) if row[7] is not None else None,
                    days_to_trough=float(row[8]) if row[8] is not None else None,
                    severity_q25=float(row[9]) if row[9] is not None else None,
                    severity_q50=float(row[10]) if row[10] is not None else None,
                    severity_q75=float(row[11]) if row[11] is not None else None
                )
            
            # Calculate actual values if needed (similar to original logic)
            actual_change_10d = float(row[12]) if row[12] is not None else None
            actual_asi_value_10d = float(row[13]) if row[13] is not None else None
            
            if actual_change_10d is None or actual_asi_value_10d is None:
                # Calculate if we have future data
                try:
                    pred_date_dt = datetime.strptime(pred_date, '%Y-%m-%d').date()
                    future_date = pred_date_dt + timedelta(days=10)
                    future_date_str = future_date.strftime('%Y-%m-%d')
                    
                    if future_date_str in asi_dict:
                        pred_asi_value = float(row[1])
                        future_price = asi_dict[future_date_str]
                        
                        if actual_change_10d is None:
                            actual_change_10d = ((future_price - pred_asi_value) / pred_asi_value) * 100
                        if actual_asi_value_10d is None:
                            actual_asi_value_10d = future_price
                        
                        # Update database with calculated values
                        try:
                            with engine.connect() as update_conn:
                                update_conn.execute(text("""
                                    UPDATE comprehensive_predictions 
                                    SET actual_change_10d = :actual_change_10d,
                                        actual_asi_value_10d = :actual_asi_value_10d
                                    WHERE date = :pred_date
                                """), {
                                    'actual_change_10d': actual_change_10d,
                                    'actual_asi_value_10d': actual_asi_value_10d,
                                    'pred_date': pred_date
                                })
                                update_conn.commit()
                        except Exception as e:
                            logger.error(f"Error updating actual values: {e}")
                except Exception as e:
                    logger.debug(f"Could not calculate values for {pred_date}: {e}")
            
            historical_predictions.append(HistoricalPrediction(
                date=pred_date,
                asi_value=float(row[1]),
                prediction=bool(row[2]),
                confidence=float(row[3]),
                risk_level=str(row[4]),
                severity_prediction=severity_pred,
                actual_change_10d=round(actual_change_10d, 2) if actual_change_10d is not None else None,
                actual_asi_value_10d=round(actual_asi_value_10d, 2) if actual_asi_value_10d is not None else None
            ))
        
        return historical_predictions
        
    except Exception as e:
        logger.error(f"Error getting comprehensive past predictions: {e}")
        return []

def fetch_asi_data_cached(days=300, force_refresh=False):
    """
    Fetch ASI data with in-memory caching (same as original)
    """
    global ASI_DATA_CACHE
    
    current_time = time.time()
    cache_valid = (
        not force_refresh and
        ASI_DATA_CACHE['data'] is not None and
        ASI_DATA_CACHE['last_updated'] is not None and
        (current_time - ASI_DATA_CACHE['last_updated']) < ASI_DATA_CACHE['cache_duration']
    )
    
    if cache_valid:
        logger.info("Using cached ASI data")
        return ASI_DATA_CACHE['data'].copy()
    
    logger.info("Fetching fresh ASI data...")
    
    try:
        from tvDatafeed import TvDatafeed, Interval
    except ImportError:
        logger.error("tvDatafeed not available. Install with: pip install tvDatafeed")
        return None

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"Fetching ASI data from {start_date.date()} to {end_date.date()} ({days} days)")

    tv = TvDatafeed()
    symbols = ['ASI', 'ASI.N0000', 'CSEASI', 'CSE:ASI']
    n_bars = days + 100 

    for symbol in symbols:
        try:
            logger.info(f"Trying symbol: {symbol}")
            data = tv.get_hist(
                symbol=symbol,
                exchange='CSELK',
                interval=Interval.in_daily,
                n_bars=n_bars
            )

            if data is not None and not data.empty:
                logger.info(f"Successfully fetched data for {symbol}")
                
                data = data.reset_index()
                data['datetime'] = pd.to_datetime(data['datetime'])
                mask = (data['datetime'] >= start_date) & (data['datetime'] <= end_date)
                filtered_data = data.loc[mask].copy()
                
                if filtered_data.empty:
                    logger.warning(f"Data fetched but empty after date filtering for {symbol}")
                    continue

                filtered_data['date'] = filtered_data['datetime'].dt.date
                final_df = filtered_data[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                final_df.sort_values('date', inplace=True)
                
                ASI_DATA_CACHE['data'] = final_df.copy()
                ASI_DATA_CACHE['last_updated'] = current_time
                
                logger.info(f"ASI data cached: {len(final_df)} records, cache updated at {datetime.fromtimestamp(current_time)}")
                
                return final_df

        except Exception as e:
            logger.warning(f"Failed for {symbol}: {e}")

    logger.error("Failed to fetch ASI data from all symbols.")
    return None

class ComprehensivePredictionEngine:
    """Two-stage prediction engine for comprehensive analysis"""
    
    def __init__(self):
        self.feature_calculator = ComprehensiveFeatureCalculator()
    
    def load_comprehensive_models(self, model_dir: str = "comprehensive_models"):
        """Load both classification and severity models"""
        global classification_models, severity_models, feature_names
        
        try:
            # Load classification models
            classification_files = [
                'classification_random_forest.pkl',
                'classification_logistic_regression.pkl',
                'classification_xgboost.pkl'
            ]
            
            for filename in classification_files:
                filepath = os.path.join(model_dir, filename)
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    model_name = filename.replace('classification_', '').replace('.pkl', '').replace('_', ' ').title()
                    classification_models[model_name] = model_data
                    
                    if feature_names is None:
                        feature_names = model_data['feature_names']
                    
                    logger.info(f"Loaded classification model: {model_name}")
            
            # Load severity models
            severity_files = [
                'severity_random_forest_multi_output.pkl',
                'severity_xgboost_multi_output.pkl',
                'severity_quantile_regression.pkl'
            ]
            
            for filename in severity_files:
                filepath = os.path.join(model_dir, filename)
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    model_name = filename.replace('severity_', '').replace('.pkl', '').replace('_', ' ').title()
                    severity_models[model_name] = model_data
                    
                    logger.info(f"Loaded severity model: {model_name}")
            
            logger.info(f"Comprehensive models loaded successfully")
            logger.info(f"Classification models: {list(classification_models.keys())}")
            logger.info(f"Severity models: {list(severity_models.keys())}")
            logger.info(f"Features: {len(feature_names) if feature_names else 0}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading comprehensive models: {e}")
            return False
    
    def generate_comprehensive_explanation(self, downturn_pred: bool, downturn_prob: float, 
                                         severity_pred: Optional[dict], features: Dict) -> str:
        """Generate comprehensive explanation"""
        
        # Base explanation
        if downturn_pred and downturn_prob > 0.8:
            explanation = f"HIGH RISK: Model predicts {downturn_prob*100:.1f}% probability of -5% drawdown within 10 days. "
        elif downturn_pred and downturn_prob > 0.6:
            explanation = f"MODERATE RISK: Model predicts {downturn_prob*100:.1f}% probability of -5% drawdown within 10 days. "
        elif downturn_pred:
            explanation = f"LOW RISK: Model predicts {downturn_prob*100:.1f}% probability of -5% drawdown within 10 days. "
        else:
            explanation = f"STABLE: Model predicts {(1-downturn_prob)*100:.1f}% probability of stability (no -5% drawdown). "
        
        # Add severity information if available
        if severity_pred and downturn_pred:
            total_drop = severity_pred.get('total_severity', 0) * 100
            days_to_trough = severity_pred.get('days_to_trough', 0)
            
            explanation += f"If downturn occurs: Expected total drop {total_drop:.1f}%, "
            explanation += f"reaching trough in ~{days_to_trough:.0f} days. "
            
            # Add confidence intervals if available
            if severity_pred.get('severity_q25') and severity_pred.get('severity_q75'):
                q25 = severity_pred['severity_q25'] * 100
                q75 = severity_pred['severity_q75'] * 100
                explanation += f"Severity range: {q25:.1f}% to {q75:.1f}% (50% confidence). "
        
        # Technical factors
        risk_factors = []
        positive_factors = []
        
        rsi = features.get('rsi_14', 50)
        if rsi < 30:
            risk_factors.append("RSI oversold")
        elif rsi > 70:
            positive_factors.append("RSI overbought")
        
        if features.get('below_sma50', 0) == 1:
            risk_factors.append("Below SMA50")
        
        if features.get('macd_bearish', 0) == 1:
            risk_factors.append("MACD bearish")
        
        volatility = features.get('volatility_20', 0)
        if volatility > 0.3:
            risk_factors.append("High volatility")
        
        if risk_factors:
            explanation += f"Risk factors: {', '.join(risk_factors)}. "
        if positive_factors:
            explanation += f"Supporting: {', '.join(positive_factors)}. "
        
        return explanation
    
    def get_risk_level(self, downturn_pred: bool, downturn_prob: float) -> str:
        """Get risk level based on prediction"""
        if downturn_pred:
            if downturn_prob >= 0.8:
                return "HIGH"
            elif downturn_prob >= 0.6:
                return "MODERATE" 
            else:
                return "LOW"
        else:
            return "STABLE"
    
    async def make_comprehensive_prediction(self, target_date: date = None) -> Dict[str, Any]:
        """Make comprehensive two-stage prediction"""
        if not classification_models:
            raise ValueError("Classification models not loaded")
        
        if target_date is None:
            target_date = date.today()
        
        # Fetch and prepare data
        df = fetch_asi_data_cached(300)
        if df is None or df.empty:
            raise ValueError("No data available")
        
        # Compute comprehensive features
        df = self.feature_calculator.compute_comprehensive_features(df)
        
        # Get latest data
        df['date'] = pd.to_datetime(df['date']).dt.date
        available_dates = df[df['date'] <= target_date]
        
        if available_dates.empty:
            raise ValueError(f"No data for {target_date}")
        
        latest_row = available_dates.iloc[-1]
        
        # Prepare features
        feature_values = []
        feature_dict = {}
        
        for feature in feature_names:
            value = latest_row.get(feature, 0)
            if pd.isna(value):
                value = 0
            feature_values.append(value)
            feature_dict[feature] = value
        
        X = np.array([feature_values])
        
        # STAGE 1: CLASSIFICATION PREDICTION
        # Use the best performing classification model (Random Forest from your results)
        best_class_model = None
        best_class_score = -1
        
        for name, model_data in classification_models.items():
            if 'Random Forest' in name:  # Use Random Forest as it had highest AUC (94.7%)
                best_class_model = model_data
                break
        
        if best_class_model is None:
            best_class_model = list(classification_models.values())[0]
        
        # Make classification prediction
        model = best_class_model['model']
        scaler = best_class_model.get('scaler')
        
        X_class = X.copy()
        if scaler:
            X_class = scaler.transform(X_class)
        
        downturn_prediction = model.predict(X_class)[0]
        downturn_probability = model.predict_proba(X_class)[0, 1]
        
        # STAGE 2: SEVERITY ANALYSIS (conditional on downturn prediction)
        severity_analysis = None
        
        if downturn_prediction and severity_models:
            # Use the best severity model
            best_sev_model = None
            
            for name, model_data in severity_models.items():
                if 'Random Forest Multi Output' in name:  # Use Random Forest Multi-output
                    best_sev_model = model_data
                    break
            
            if best_sev_model is None and severity_models:
                best_sev_model = list(severity_models.values())[0]
            
            if best_sev_model:
                try:
                    model_data = best_sev_model['model_data']
                    
                    if 'XGBoost Multi-output' in str(best_sev_model):
                        # Separate XGBoost models
                        sev_model = model_data['severity_model']
                        dur_model = model_data['duration_model']
                        
                        additional_severity = sev_model.predict(X)[0]
                        days_to_trough = dur_model.predict(X)[0]
                        
                    elif 'model' in model_data:
                        # Multi-output model
                        sev_model = model_data['model']
                        predictions = sev_model.predict(X)[0]
                        additional_severity = predictions[0]
                        days_to_trough = predictions[1]
                    else:
                        additional_severity = -0.07  # Default based on historical average
                        days_to_trough = 87.5
                    
                    total_severity = -0.05 + additional_severity
                    
                    severity_analysis = {
                        'additional_severity': float(additional_severity),
                        'total_severity': float(total_severity),
                        'days_to_trough': float(days_to_trough)
                    }
                    
                    # Add quantile predictions if available
                    if 'Quantile Regression' in severity_models:
                        quantile_model_data = severity_models['Quantile Regression']['model_data']
                        if 'quantile_models' in quantile_model_data:
                            quantile_models = quantile_model_data['quantile_models']
                            
                            severity_analysis['severity_q25'] = float(-0.05 + quantile_models['q25'].predict(X)[0])
                            severity_analysis['severity_q50'] = float(-0.05 + quantile_models['q50'].predict(X)[0])
                            severity_analysis['severity_q75'] = float(-0.05 + quantile_models['q75'].predict(X)[0])
                
                except Exception as e:
                    logger.error(f"Error in severity prediction: {e}")
                    # Use historical averages as fallback
                    severity_analysis = {
                        'additional_severity': -0.0733,  # Historical average
                        'total_severity': -0.1244,      # Historical average total
                        'days_to_trough': 87.5          # Historical average days
                    }
        
        # Generate comprehensive explanation
        risk_level = self.get_risk_level(bool(downturn_prediction), downturn_probability)
        explanation = self.generate_comprehensive_explanation(
            bool(downturn_prediction), downturn_probability, severity_analysis, feature_dict
        )
        
        # Get past 10 days predictions
        target_date_str = latest_row['date'].strftime('%Y-%m-%d')
        past_10_days = get_comprehensive_past_10_days_predictions(target_date_str, df)
        
        return {
            'date': target_date_str,
            'asi_value': float(latest_row['close']),
            'downturn_prediction': bool(downturn_prediction),
            'downturn_probability': float(downturn_probability),
            'risk_level': risk_level,
            'explanation': explanation,
            'severity_analysis': severity_analysis,
            'past_10_days': [pred.dict() for pred in past_10_days]
        }

# Initialize prediction engine
prediction_engine = ComprehensivePredictionEngine()

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup event"""
    global scheduler
    
    # Initialize database
    init_comprehensive_postgresql_db()
    
    # Load comprehensive models
    prediction_engine.load_comprehensive_models()
    
    # Pre-warm cache
    logger.info("Pre-warming ASI data cache...")
    fetch_asi_data_cached(300)
    
    # Initialize scheduler
    scheduler = AsyncIOScheduler()
    
    # Add daily prediction job
    async def daily_comprehensive_prediction_task():
        try:
            logger.info("Running daily comprehensive prediction task...")
            prediction_data = await prediction_engine.make_comprehensive_prediction()
            store_comprehensive_prediction_postgresql(prediction_data)
            logger.info(f"Daily comprehensive prediction completed: {prediction_data['risk_level']}")
        except Exception as e:
            logger.error(f"Daily comprehensive prediction failed: {e}")
    
    # Add cache refresh job
    def refresh_cache_task():
        try:
            logger.info("Refreshing ASI data cache...")
            fetch_asi_data_cached(300, force_refresh=True)
            logger.info("Cache refresh completed")
        except Exception as e:
            logger.error(f"Cache refresh failed: {e}")
    
    scheduler.add_job(
        daily_comprehensive_prediction_task,
        CronTrigger(hour=14, minute=35),
        id='daily_comprehensive_prediction',
        replace_existing=True
    )
    
    scheduler.add_job(
        refresh_cache_task,
        CronTrigger(minute=0),  # Every hour
        id='cache_refresh',
        replace_existing=True
    )
    
    scheduler.start()
    logger.info("Scheduler started - Daily comprehensive predictions at 2:35 PM")

@app.on_event("shutdown")
async def shutdown_event():
    if scheduler:
        scheduler.shutdown()

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "ASI Comprehensive Downturn Predictor API - Two-Stage ML Pipeline",
        "version": "2.0.0",
        "features": [
            "Stage 1: Classification (downturn probability)",
            "Stage 2: Severity Analysis (additional drop + days to trough)",
            "Comprehensive feature engineering (25+ indicators)",
            "PostgreSQL database integration",
            "Historical predictions with severity analysis",
            "Quantile regression confidence intervals",
            "Business impact metrics"
        ],
        "pipeline": {
            "stage_1": "Random Forest Classification (94.7% AUC)",
            "stage_2": "Multi-output Regression (conditional)",
            "features": len(feature_names) if feature_names else "Not loaded",
            "models_loaded": {
                "classification": list(classification_models.keys()),
                "severity": list(severity_models.keys())
            }
        },
        "status": "running",
        "database": "PostgreSQL (comprehensive schema)",
        "data_source": "TradingView (cached)",
        "cache_info": {
            "last_updated": datetime.fromtimestamp(ASI_DATA_CACHE['last_updated']).isoformat() if ASI_DATA_CACHE['last_updated'] else None,
            "cache_duration_minutes": ASI_DATA_CACHE['cache_duration'] // 60,
            "data_days": ASI_DATA_CACHE['data_days']
        }
    }

@app.get("/predict", response_model=ComprehensivePredictionResponse)
async def get_comprehensive_prediction():
    """Get comprehensive prediction with two-stage analysis"""
    try:
        if not classification_models:
            prediction_engine.load_comprehensive_models()
            if not classification_models:
                raise HTTPException(status_code=500, detail="Models not loaded")
        
        prediction_data = await prediction_engine.make_comprehensive_prediction()
        
        # Format severity analysis for response
        severity_analysis = None
        if prediction_data.get('severity_analysis'):
            severity_data = prediction_data['severity_analysis']
            severity_analysis = SeverityPrediction(**severity_data)
        
        return ComprehensivePredictionResponse(
            date=prediction_data['date'],
            asi_value=prediction_data['asi_value'],
            downturn_prediction=prediction_data['downturn_prediction'],
            downturn_probability=prediction_data['downturn_probability'],
            risk_level=prediction_data['risk_level'],
            explanation=prediction_data['explanation'],
            severity_analysis=severity_analysis,
            past_10_days=prediction_data['past_10_days']
        )
        
    except Exception as e:
        logger.error(f"Comprehensive prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Comprehensive prediction failed: {str(e)}")

@app.get("/predict/{target_date}", response_model=ComprehensivePredictionResponse)
async def get_comprehensive_prediction_for_date(target_date: str):
    """Get comprehensive prediction for specific date"""
    try:
        if not classification_models:
            prediction_engine.load_comprehensive_models()
            if not classification_models:
                raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Validate date format
        try:
            target_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Check if date is reasonable
        today = date.today()
        if target_date_obj > today + timedelta(days=7):
            raise HTTPException(status_code=400, detail="Cannot predict more than 7 days in the future")
        if target_date_obj < today - timedelta(days=365):
            raise HTTPException(status_code=400, detail="Cannot predict more than 1 year in the past")
        
        prediction_data = await prediction_engine.make_comprehensive_prediction(target_date_obj)
        
        # Format severity analysis for response
        severity_analysis = None
        if prediction_data.get('severity_analysis'):
            severity_data = prediction_data['severity_analysis']
            severity_analysis = SeverityPrediction(**severity_data)
        
        return ComprehensivePredictionResponse(
            date=prediction_data['date'],
            asi_value=prediction_data['asi_value'],
            downturn_prediction=prediction_data['downturn_prediction'],
            downturn_probability=prediction_data['downturn_probability'],
            risk_level=prediction_data['risk_level'],
            explanation=prediction_data['explanation'],
            severity_analysis=severity_analysis,
            past_10_days=prediction_data['past_10_days']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Date comprehensive prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Date prediction failed: {str(e)}")

@app.post("/predict/manual")
async def manual_comprehensive_prediction():
    """Manually trigger comprehensive prediction"""
    try:
        prediction_data = await prediction_engine.make_comprehensive_prediction()
        store_comprehensive_prediction_postgresql(prediction_data)
        
        return {
            "message": "Manual comprehensive prediction completed",
            "prediction": prediction_data
        }
    except Exception as e:
        logger.error(f"Manual comprehensive prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Manual prediction failed: {str(e)}")

@app.get("/business-impact/{portfolio_value}")
async def get_business_impact_analysis(portfolio_value: float):
    """Generate business impact analysis for given portfolio value"""
    try:
        prediction_data = await prediction_engine.make_comprehensive_prediction()
        
        scenarios = {}
        recommendations = []
        
        if prediction_data['downturn_prediction'] and prediction_data.get('severity_analysis'):
            severity = prediction_data['severity_analysis']
            
            # Portfolio impact scenarios
            scenarios = {
                "conservative_scenario": {
                    "severity": severity.get('severity_q25', -0.05) * 100,
                    "portfolio_loss": portfolio_value * abs(severity.get('severity_q25', -0.05)),
                    "description": "25th percentile scenario (mild case)"
                },
                "expected_scenario": {
                    "severity": severity.get('total_severity', -0.1244) * 100,
                    "portfolio_loss": portfolio_value * abs(severity.get('total_severity', -0.1244)),
                    "description": "Expected case based on historical analysis"
                },
                "severe_scenario": {
                    "severity": severity.get('severity_q75', -0.15) * 100,
                    "portfolio_loss": portfolio_value * abs(severity.get('severity_q75', -0.15)),
                    "description": "75th percentile scenario (severe case)"
                }
            }
            
            # Recommendations
            recommendations = [
                f"Consider reducing ASI exposure by 20-30%",
                f"Set stop-loss at {severity.get('total_severity', -0.1244) * 60:.1f}%",
                f"Plan for {severity.get('days_to_trough', 87.5):.0f}-day downturn period",
                f"Keep cash reserve for potential reentry opportunities"
            ]
        else:
            scenarios = {
                "current_assessment": {
                    "severity": 0,
                    "portfolio_loss": 0,
                    "description": "No downturn predicted - portfolio stable"
                }
            }
            
            recommendations = [
                "Continue current allocation",
                "Monitor for any risk level changes",
                "Consider slight position increases on any weakness"
            ]
        
        return BusinessImpactAnalysis(
            portfolio_scenarios=scenarios,
            risk_metrics={
                "downturn_probability": prediction_data['downturn_probability'],
                "risk_level": prediction_data['risk_level'],
                "expected_duration_days": prediction_data.get('severity_analysis', {}).get('days_to_trough', None)
            },
            recommended_actions=recommendations
        )
        
    except Exception as e:
        logger.error(f"Business impact analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Business impact analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        # Model status
        classification_status = "loaded" if classification_models else "not_loaded"
        severity_status = "loaded" if severity_models else "not_loaded"
        
        # Database status
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM comprehensive_predictions"))
                prediction_count = result.fetchone()[0]
                db_status = f"connected ({prediction_count} predictions)"
        except Exception as e:
            db_status = f"error: {str(e)}"
        
        # Cache status
        cache_status = {
            "is_cached": ASI_DATA_CACHE['data'] is not None,
            "last_updated": datetime.fromtimestamp(ASI_DATA_CACHE['last_updated']).isoformat() if ASI_DATA_CACHE['last_updated'] else None,
            "records_count": len(ASI_DATA_CACHE['data']) if ASI_DATA_CACHE['data'] is not None else 0,
            "cache_age_minutes": (time.time() - ASI_DATA_CACHE['last_updated']) // 60 if ASI_DATA_CACHE['last_updated'] else None
        }
        
        scheduler_status = "running" if scheduler and scheduler.running else "stopped"
        
        return {
            "status": "healthy",
            "pipeline": {
                "stage_1_classification": classification_status,
                "stage_2_severity": severity_status,
                "models_loaded": {
                    "classification_count": len(classification_models),
                    "severity_count": len(severity_models),
                    "classification_models": list(classification_models.keys()),
                    "severity_models": list(severity_models.keys())
                }
            },
            "database": db_status,
            "scheduler": scheduler_status,
            "cache": cache_status,
            "features": [
                "Two-stage ML pipeline",
                "Comprehensive severity analysis",
                "Quantile regression confidence intervals",
                "Business impact analysis",
                "Historical prediction tracking",
                "Automated daily predictions at 2:35 PM"
            ],
            "performance_metrics": {
                "classification_auc": "94.7% (Random Forest)",
                "severity_mae": "8.5% prediction error",
                "duration_mae": "46.9 days prediction error"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    
    # Load comprehensive models
    prediction_engine.load_comprehensive_models()
    
    # Run the app
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )