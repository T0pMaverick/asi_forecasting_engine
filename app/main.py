# app/main.py

from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from datetime import date
from pydantic import BaseModel

from .database import SessionLocal, engine
from .models import Base, ASIAlert
from .scheduler import run_engine,start_scheduler
from .data_fetcher import fetch_asi_data
from datetime import datetime
from app.cache import cache





Base.metadata.create_all(bind=engine)

app = FastAPI(title="ASI Risk Alert Service")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------
# PRODUCTION ENDPOINTS
# -------------------------

@app.on_event("startup")
def startup():

    print("Loading ASI historical data into cache...")

    df = fetch_asi_data(days=6000)  # ~15 years
    if df is None or df.empty:
        raise RuntimeError("Failed to load ASI data at startup")

    cache.data = df
    cache.last_updated = datetime.now()
    print(df)
    print(f"ASI cache loaded: {len(df)} rows")

    start_scheduler()


@app.get("/alerts/latest")
def latest_alert(db: Session = Depends(get_db)):
    result = db.query(ASIAlert).order_by(ASIAlert.run_date.desc()).first()
    print(result)
    return result

@app.get("/alerts/history")
def alert_history(limit: int = 30, db: Session = Depends(get_db)):
    return (
        db.query(ASIAlert)
        .order_by(ASIAlert.run_date.desc())
        .limit(limit)
        .all()
    )

# -------------------------
# BACKTEST / REPLAY ENDPOINT
# -------------------------

class BacktestRequest(BaseModel):
    as_of_date: date

@app.post("/alerts/run-as-of")
def run_as_of(req: BacktestRequest):
    results = run_engine(as_of_date=req.as_of_date, save=False)

    if results is None:
        return {
            "status": "error",
            "message": "Insufficient or unavailable data for this date"
        }

    return {
        "as_of_date": req.as_of_date,
        **results
    }


@app.get("/health")
def health():
    return {"status": "ok"}
