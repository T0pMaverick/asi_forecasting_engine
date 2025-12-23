#!/usr/bin/env python3
"""
ASI Market Downturn Predictor API - Enhanced with Caching and PostgreSQL
- In-memory caching for ASI data (300 days)
- PostgreSQL database integration
- Real-time data fetching with TradingView
- Added actual ASI values in past 10 days data
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
DB_PORT =os.getenv("DB_PORT")
DB_USER = os.getenv("DB_ADMIN_USER")
DB_PASSWORD =os.getenv("DB_ADMIN_PASSWORD") 
DB_NAME = os.getenv("DB_NAME")

# Create PostgreSQL connection string
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Enhanced Pydantic models
class HistoricalPrediction(BaseModel):
    date: str
    asi_value: float  # ASI value at prediction time (forecasted)
    prediction: bool
    confidence: float
    risk_level: str
    actual_change_10d: Optional[float] = None  # Percentage change after 10 days
    actual_asi_value_10d: Optional[float] = None  # Actual ASI value after 10 days

class PredictionResponse(BaseModel):
    date: str
    asi_value: float
    prediction: bool
    confidence: float
    explanation: str
    risk_level: str
    past_10_days: List[HistoricalPrediction]

class TrendComparison(BaseModel):
    current_prediction: PredictionResponse
    trend_analysis: Dict[str, Any]
    recent_predictions: List[Dict[str, Any]]

# Initialize FastAPI app
app = FastAPI(
    title="ASI Market Downturn Predictor - Enhanced with Caching & PostgreSQL",
    description="Predicts -5% drawdowns in ASI within 10 days using XGBoost with real-time data and caching",
    version="1.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
scaler = None
feature_names = None
scheduler = None

# In-memory cache for ASI data
ASI_DATA_CACHE = {
    'data': None,
    'last_updated': None,
    'cache_duration': 3600,  # 1 hour in seconds
    'data_days': 300
}

class FeatureCalculator:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)
    
    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df['close'] = pd.to_numeric(df['close'])
        df['returns'] = df['close'].pct_change()
        
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
        df['rsi_14'] = FeatureCalculator.calculate_rsi(df['close'], 14)
        df['rsi_7'] = FeatureCalculator.calculate_rsi(df['close'], 7)
        
        # Volatility
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

# PostgreSQL database functions
def init_postgresql_db():
    """Initialize PostgreSQL database with enhanced schema"""
    try:
        with engine.connect() as conn:
            # Create main predictions table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    date DATE UNIQUE NOT NULL,
                    asi_value DECIMAL(10, 4) NOT NULL,
                    prediction BOOLEAN NOT NULL,
                    confidence DECIMAL(5, 4) NOT NULL,
                    explanation TEXT NOT NULL,
                    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('HIGH', 'MODERATE', 'LOW', 'STABLE')),
                    actual_change_10d DECIMAL(8, 4) DEFAULT NULL,
                    actual_asi_value_10d DECIMAL(10, 4) DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            
            # Create indexes
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(date);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_predictions_risk_level ON predictions(risk_level);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON predictions(prediction);"))
            
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
                DROP TRIGGER IF EXISTS update_predictions_updated_at ON predictions;
                CREATE TRIGGER update_predictions_updated_at 
                    BEFORE UPDATE ON predictions 
                    FOR EACH ROW 
                    EXECUTE FUNCTION update_updated_at_column();
            """))
            
            # Add new columns if they don't exist
            try:
                conn.execute(text("ALTER TABLE predictions ADD COLUMN actual_change_10d DECIMAL(8, 4) DEFAULT NULL;"))
            except Exception:
                pass  # Column already exists
                
            try:
                conn.execute(text("ALTER TABLE predictions ADD COLUMN actual_asi_value_10d DECIMAL(10, 4) DEFAULT NULL;"))
            except Exception:
                pass  # Column already exists
            
            conn.commit()
            logger.info("PostgreSQL database initialized successfully")
            return True
            
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return False

def store_prediction_postgresql(prediction_data):
    """Store prediction in PostgreSQL"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO predictions 
                (date, asi_value, prediction, confidence, explanation, risk_level)
                VALUES (:date, :asi_value, :prediction, :confidence, :explanation, :risk_level)
                ON CONFLICT (date) 
                DO UPDATE SET
                    asi_value = EXCLUDED.asi_value,
                    prediction = EXCLUDED.prediction,
                    confidence = EXCLUDED.confidence,
                    explanation = EXCLUDED.explanation,
                    risk_level = EXCLUDED.risk_level,
                    updated_at = CURRENT_TIMESTAMP
            """), {
                'date': prediction_data['date'],
                'asi_value': prediction_data['asi_value'],
                'prediction': prediction_data['prediction'],
                'confidence': prediction_data['confidence'],
                'explanation': prediction_data['explanation'],
                'risk_level': prediction_data['risk_level']
            })
            conn.commit()
            logger.info(f"Prediction stored for {prediction_data['date']}")
            
    except Exception as e:
        logger.error(f"Error storing prediction: {e}")

def get_past_10_days_predictions(target_date: str, asi_data: pd.DataFrame) -> List[HistoricalPrediction]:
    """Get past 10 days of predictions with real ASI values, actual changes, and actual ASI values after 10 days"""
    try:
        target_dt = datetime.strptime(target_date, '%Y-%m-%d').date()
        start_date = target_dt - timedelta(days=15)
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT date, asi_value, prediction, confidence, risk_level, actual_change_10d, actual_asi_value_10d
                FROM predictions 
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
            pred_asi_value = float(row[1])
            prediction = bool(row[2])
            confidence = float(row[3])
            risk_level = str(row[4])
            stored_change = float(row[5]) if row[5] is not None else None
            stored_actual_value = float(row[6]) if row[6] is not None else None
            
            # Calculate actual 10-day change and actual ASI value if we have future data and it's not already stored
            actual_change_10d = stored_change
            actual_asi_value_10d = stored_actual_value
            
            if actual_change_10d is None or actual_asi_value_10d is None:
                try:
                    pred_date_dt = datetime.strptime(pred_date, '%Y-%m-%d').date()
                    future_date = pred_date_dt + timedelta(days=10)
                    future_date_str = future_date.strftime('%Y-%m-%d')
                    
                    if future_date_str in asi_dict:
                        future_price = asi_dict[future_date_str]
                        
                        # Calculate values if not already stored
                        if actual_change_10d is None:
                            actual_change_10d = ((future_price - pred_asi_value) / pred_asi_value) * 100
                        if actual_asi_value_10d is None:
                            actual_asi_value_10d = future_price
                        
                        # Update the database with calculated values
                        try:
                            with engine.connect() as update_conn:
                                update_conn.execute(text("""
                                    UPDATE predictions 
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
                    actual_change_10d = None
                    actual_asi_value_10d = None
            
            historical_predictions.append(HistoricalPrediction(
                date=pred_date,
                asi_value=pred_asi_value,
                prediction=prediction,
                confidence=confidence,
                risk_level=risk_level,
                actual_change_10d=round(actual_change_10d, 2) if actual_change_10d is not None else None,
                actual_asi_value_10d=round(actual_asi_value_10d, 2) if actual_asi_value_10d is not None else None
            ))
        
        return historical_predictions
        
    except Exception as e:
        logger.error(f"Error getting past predictions: {e}")
        return []

def get_recent_predictions_postgresql(limit=10):
    """Get recent predictions from PostgreSQL"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT date, asi_value, prediction, confidence, explanation, risk_level, actual_change_10d, actual_asi_value_10d, created_at
                FROM predictions 
                ORDER BY date DESC 
                LIMIT :limit
            """), {'limit': limit})
            
            results = result.fetchall()
        
        predictions = []
        for row in results:
            predictions.append({
                'date': row[0].strftime('%Y-%m-%d'),
                'asi_value': float(row[1]),
                'prediction': bool(row[2]),
                'confidence': float(row[3]),
                'explanation': str(row[4]),
                'risk_level': str(row[5]),
                'actual_change_10d': float(row[6]) if row[6] is not None else None,
                'actual_asi_value_10d': float(row[7]) if row[7] is not None else None,
                'created_at': row[8].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        return []

def fetch_asi_data_cached(days=300, force_refresh=False):
    """
    Fetch ASI data with in-memory caching
    """
    global ASI_DATA_CACHE
    
    # Check if cache is valid
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
    
    # Fetch fresh data
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
    
    # Symbols to try
    symbols = ['ASI', 'ASI.N0000', 'CSEASI', 'CSE:ASI']
    
    # Buffer for trading days vs calendar days
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
                
                # Reset index to get datetime as a column
                data = data.reset_index()
                
                # Filter by date
                data['datetime'] = pd.to_datetime(data['datetime'])
                mask = (data['datetime'] >= start_date) & (data['datetime'] <= end_date)
                filtered_data = data.loc[mask].copy()
                
                if filtered_data.empty:
                    logger.warning(f"Data fetched but empty after date filtering for {symbol}")
                    continue

                # Clean up
                filtered_data['date'] = filtered_data['datetime'].dt.date
                final_df = filtered_data[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                final_df.sort_values('date', inplace=True)
                
                # Update cache
                ASI_DATA_CACHE['data'] = final_df.copy()
                ASI_DATA_CACHE['last_updated'] = current_time
                
                logger.info(f"ASI data cached: {len(final_df)} records, cache updated at {datetime.fromtimestamp(current_time)}")
                
                return final_df

        except Exception as e:
            logger.warning(f"Failed for {symbol}: {e}")

    logger.error("Failed to fetch ASI data from all symbols.")
    return None

class PredictionEngine:
    """Main prediction engine"""
    
    def __init__(self):
        self.feature_calculator = FeatureCalculator()
    
    def load_model(self, model_path: str = "models/xgboost.pkl"):
        """Load trained model"""
        global model, scaler, feature_names
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data['model']
            scaler = model_data.get('scaler')
            feature_names = model_data['feature_names']
            
            logger.info(f"Model loaded successfully: {model_data.get('model_name', 'Unknown')}")
            logger.info(f"Features: {len(feature_names)}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def generate_explanation(self, prediction: bool, confidence: float, features: Dict) -> str:
        """Generate explanation"""
        risk_factors = []
        positive_factors = []
        
        # Analyze key indicators
        rsi = features.get('rsi_14', 50)
        if rsi < 30:
            risk_factors.append("RSI indicates oversold conditions")
        elif rsi > 70:
            positive_factors.append("RSI indicates overbought conditions")
        
        if features.get('below_sma50', 0) == 1:
            risk_factors.append("Price is below 50-day moving average")
        else:
            positive_factors.append("Price is above 50-day moving average")
        
        if features.get('negative_momentum_20', 0) == 1:
            risk_factors.append("Negative 20-day momentum")
        
        if features.get('macd_bearish', 0) == 1:
            risk_factors.append("MACD showing bearish signals")
        
        volatility = features.get('volatility_20', 0)
        if volatility > 0.3:
            risk_factors.append("High volatility detected")
        
        # Generate explanation
        if prediction and confidence > 0.8:
            explanation = f"HIGH RISK: Model predicts {confidence*100:.1f}% probability of -5% drawdown within 10 days. "
        elif prediction and confidence > 0.6:
            explanation = f"MODERATE RISK: Model predicts {confidence*100:.1f}% probability of -5% drawdown within 10 days. "
        elif prediction:
            explanation = f"LOW RISK: Model predicts {confidence*100:.1f}% probability of -5% drawdown within 10 days. "
        else:
            explanation = f"STABLE: Model predicts {(1-confidence)*100:.1f}% probability of stability (no -5% drawdown). "
        
        if risk_factors:
            explanation += f"Risk factors: {', '.join(risk_factors)}. "
        if positive_factors:
            explanation += f"Supporting factors: {', '.join(positive_factors)}. "
        
        # Add current RSI and volatility
        explanation += f"Current RSI: {rsi:.1f}, Volatility: {volatility*100:.1f}%."
        
        return explanation
    
    def get_risk_level(self, prediction: bool, confidence: float) -> str:
        """Get risk level"""
        if prediction:
            if confidence >= 0.8:
                return "HIGH"
            elif confidence >= 0.6:
                return "MODERATE" 
            else:
                return "LOW"
        else:
            return "STABLE"
    
    async def make_prediction(self, target_date: date = None) -> Dict[str, Any]:
        """Make prediction with past 10 days historical data"""
        if model is None:
            raise ValueError("Model not loaded")
        
        if target_date is None:
            target_date = date.today()
        
        # Fetch data with caching
        df = fetch_asi_data_cached(300)
        if df is None or df.empty:
            raise ValueError("No data available")
        
        # Compute features
        df = self.feature_calculator.compute_features(df)
        
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
        
        # Make prediction
        X = np.array([feature_values])
        
        if scaler:
            X = scaler.transform(X)
        
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X)[0, 1]
        
        # Generate explanation
        explanation = self.generate_explanation(bool(prediction), confidence, feature_dict)
        risk_level = self.get_risk_level(bool(prediction), confidence)
        
        # Get past 10 days predictions with real values
        target_date_str = latest_row['date'].strftime('%Y-%m-%d')
        past_10_days = get_past_10_days_predictions(target_date_str, df)
        
        return {
            'date': target_date_str,
            'asi_value': float(latest_row['close']),
            'prediction': bool(prediction),
            'confidence': float(confidence),
            'explanation': explanation,
            'risk_level': risk_level,
            'past_10_days': [pred.dict() for pred in past_10_days]
        }

# Initialize prediction engine
prediction_engine = PredictionEngine()

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup event"""
    global scheduler
    
    # Initialize database
    init_postgresql_db()
    
    # Load model
    prediction_engine.load_model()
    
    # Pre-warm cache
    logger.info("Pre-warming ASI data cache...")
    fetch_asi_data_cached(300)
    
    # Initialize scheduler
    scheduler = AsyncIOScheduler()
    
    # Add daily prediction job
    async def daily_prediction_task():
        try:
            logger.info("Running daily prediction task...")
            prediction_data = await prediction_engine.make_prediction()
            store_prediction_postgresql(prediction_data)
            logger.info(f"Daily prediction completed: {prediction_data['risk_level']}")
        except Exception as e:
            logger.error(f"Daily prediction failed: {e}")
    
    # Add cache refresh job (every hour)
    def refresh_cache_task():
        try:
            logger.info("Refreshing ASI data cache...")
            fetch_asi_data_cached(300, force_refresh=True)
            logger.info("Cache refresh completed")
        except Exception as e:
            logger.error(f"Cache refresh failed: {e}")
    
    scheduler.add_job(
        daily_prediction_task,
        CronTrigger(hour=14, minute=35),
        id='daily_prediction',
        replace_existing=True
    )
    
    scheduler.add_job(
        refresh_cache_task,
        CronTrigger(minute=0),  # Every hour
        id='cache_refresh',
        replace_existing=True
    )
    
    scheduler.start()
    logger.info("Scheduler started - Daily predictions at 2:35 PM, Cache refresh every hour")

@app.on_event("shutdown")
async def shutdown_event():
    if scheduler:
        scheduler.shutdown()

# Enhanced API Endpoints
@app.get("/")
async def root():
    return {
        "message": "ASI Market Downturn Predictor API - Enhanced with Forecasted vs Actual Values",
        "version": "1.3.0",
        "features": [
            "In-memory caching for 300 days ASI data",
            "PostgreSQL database integration",
            "Specific date predictions: GET /predict/{date}",
            "Past 10 days data with forecasted vs actual ASI values",
            "Actual 10-day change tracking",
            "Hourly cache refresh",
            "Daily predictions at 2:35 PM"
        ],
        "status": "running",
        "database": "PostgreSQL",
        "data_source": "TradingView (cached)",
        "cache_info": {
            "last_updated": datetime.fromtimestamp(ASI_DATA_CACHE['last_updated']).isoformat() if ASI_DATA_CACHE['last_updated'] else None,
            "cache_duration_minutes": ASI_DATA_CACHE['cache_duration'] // 60,
            "data_days": ASI_DATA_CACHE['data_days']
        }
    }

@app.get("/predict", response_model=PredictionResponse)
async def get_latest_prediction():
    """Get latest prediction with past 10 days historical data including forecasted vs actual values"""
    try:
        if model is None:
            prediction_engine.load_model()
            if model is None:
                raise HTTPException(status_code=500, detail="Model not loaded")
        
        prediction_data = await prediction_engine.make_prediction()
        return PredictionResponse(**prediction_data)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predict/{target_date}", response_model=PredictionResponse)
async def get_prediction_for_date(target_date: str):
    """
    Get prediction for specific date with past 10 days data
    
    Parameters:
    - target_date: Date in YYYY-MM-DD format (e.g., 2025-12-22)
    """
    try:
        if model is None:
            prediction_engine.load_model()
            if model is None:
                raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Validate date format
        try:
            target_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Check if date is not too far in the future or past
        today = date.today()
        if target_date_obj > today + timedelta(days=7):
            raise HTTPException(status_code=400, detail="Cannot predict more than 7 days in the future")
        if target_date_obj < today - timedelta(days=365):
            raise HTTPException(status_code=400, detail="Cannot predict more than 1 year in the past")
        
        prediction_data = await prediction_engine.make_prediction(target_date_obj)
        return PredictionResponse(**prediction_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Date prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Date prediction failed: {str(e)}")

@app.get("/trends", response_model=TrendComparison)
async def get_trend_analysis():
    try:
        current_prediction = await prediction_engine.make_prediction()
        recent_predictions = get_recent_predictions_postgresql(10)
        
        if len(recent_predictions) >= 2:
            confidences = [p['confidence'] for p in recent_predictions[:5]]
            avg_confidence = sum(confidences) / len(confidences)
            confidence_change = confidences[0] - confidences[-1] if len(confidences) > 1 else 0
            
            risk_levels = [p['risk_level'] for p in recent_predictions[:5]]
            risk_counts = {level: risk_levels.count(level) for level in set(risk_levels)}
            
            # Calculate accuracy if actual changes are available
            completed_predictions = [p for p in recent_predictions if p.get('actual_change_10d') is not None]
            if completed_predictions:
                correct_predictions = 0
                for p in completed_predictions:
                    predicted_drawdown = p['prediction']
                    actual_drawdown = p['actual_change_10d'] <= -5.0
                    if predicted_drawdown == actual_drawdown:
                        correct_predictions += 1
                accuracy_rate = round((correct_predictions / len(completed_predictions)) * 100, 2)
            else:
                accuracy_rate = None
            
            trend_analysis = {
                "avg_confidence_5d": round(avg_confidence, 4),
                "confidence_change_5d": round(confidence_change, 4),
                "prediction_count_5d": sum(1 for p in recent_predictions[:5] if p['prediction']),
                "risk_level_distribution": risk_counts,
                "trend_direction": "INCREASING" if confidence_change > 0.05 else "DECREASING" if confidence_change < -0.05 else "STABLE",
                "accuracy_rate": accuracy_rate,
                "completed_evaluations": len(completed_predictions)
            }
        else:
            trend_analysis = {
                "avg_confidence_5d": current_prediction['confidence'],
                "confidence_change_5d": 0,
                "prediction_count_5d": 1 if current_prediction['prediction'] else 0,
                "risk_level_distribution": {current_prediction['risk_level']: 1},
                "trend_direction": "INSUFFICIENT_DATA",
                "accuracy_rate": None,
                "completed_evaluations": 0
            }
        
        return TrendComparison(
            current_prediction=PredictionResponse(**current_prediction),
            trend_analysis=trend_analysis,
            recent_predictions=recent_predictions
        )
        
    except Exception as e:
        logger.error(f"Trend analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")

@app.get("/history")
async def get_prediction_history(limit: int = 30):
    """Get prediction history with actual change tracking"""
    try:
        predictions = get_recent_predictions_postgresql(limit)
        
        # Calculate summary statistics
        if predictions:
            completed = [p for p in predictions if p.get('actual_change_10d') is not None]
            if completed:
                correct = sum(1 for p in completed 
                            if (p['prediction'] and p['actual_change_10d'] <= -5.0) or 
                               (not p['prediction'] and p['actual_change_10d'] > -5.0))
                accuracy = round((correct / len(completed)) * 100, 2)
            else:
                accuracy = None
            
            summary = {
                "total_predictions": len(predictions),
                "completed_evaluations": len(completed),
                "accuracy_rate": accuracy
            }
        else:
            summary = {"total_predictions": 0, "completed_evaluations": 0, "accuracy_rate": None}
        
        return {
            "predictions": predictions,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(status_code=500, detail=f"History failed: {str(e)}")

@app.post("/predict/manual")
async def manual_prediction():
    try:
        prediction_data = await prediction_engine.make_prediction()
        store_prediction_postgresql(prediction_data)
        
        return {
            "message": "Manual prediction completed",
            "prediction": prediction_data
        }
    except Exception as e:
        logger.error(f"Manual prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Manual prediction failed: {str(e)}")

@app.post("/cache/refresh")
async def refresh_cache():
    """Manually refresh ASI data cache"""
    try:
        fetch_asi_data_cached(300, force_refresh=True)
        return {
            "message": "Cache refreshed successfully",
            "last_updated": datetime.fromtimestamp(ASI_DATA_CACHE['last_updated']).isoformat(),
            "records_count": len(ASI_DATA_CACHE['data']) if ASI_DATA_CACHE['data'] is not None else 0
        }
    except Exception as e:
        logger.error(f"Cache refresh error: {e}")
        raise HTTPException(status_code=500, detail=f"Cache refresh failed: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        model_status = "loaded" if model is not None else "not_loaded"
        
        # Test database and get prediction count
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM predictions"))
                prediction_count = result.fetchone()[0]
                db_status = f"connected ({prediction_count} predictions)"
        except Exception as e:
            db_status = f"error: {str(e)}"
        
        # Check cache status
        cache_status = {
            "is_cached": ASI_DATA_CACHE['data'] is not None,
            "last_updated": datetime.fromtimestamp(ASI_DATA_CACHE['last_updated']).isoformat() if ASI_DATA_CACHE['last_updated'] else None,
            "records_count": len(ASI_DATA_CACHE['data']) if ASI_DATA_CACHE['data'] is not None else 0,
            "cache_age_minutes": (time.time() - ASI_DATA_CACHE['last_updated']) // 60 if ASI_DATA_CACHE['last_updated'] else None
        }
        
        scheduler_status = "running" if scheduler and scheduler.running else "stopped"
        
        return {
            "status": "healthy",
            "model": model_status,
            "database": db_status,
            "scheduler": scheduler_status,
            "cache": cache_status,
            "data_source": "TradingView (cached)",
            "features": [
                "Latest predictions with past 10 days",
                "Forecasted vs Actual ASI values",
                "Date-specific predictions",
                "Actual change tracking",
                "In-memory caching",
                "PostgreSQL storage",
                "Daily automation at 2:35 PM",
                "Hourly cache refresh"
            ],
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
    
    # Load model
    prediction_engine.load_model()
    
    # Run the app
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )