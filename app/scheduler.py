# app/scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler
from datetime import date
from .database import SessionLocal
from .models import ASIAlert
from .data_fetcher import fetch_asi_data
from .risk_engine import run_risk_engine

def daily_job():
    df = fetch_asi_data(days=1200)
    if df is None:
        return

    results = run_risk_engine(df)
    db = SessionLocal()

    record = ASIAlert(
        run_date=date.today(),
        **results,
        alert_minus_5="WATCH" if results["prob_minus_5"] >= 0.10 else "NONE",
        alert_minus_10="WARNING" if results["prob_minus_10"] >= 0.10 and results["vol_regime"] else "NONE",
        alert_minus_20="CRITICAL" if results["prob_minus_20"] >= 0.05 and results["vol_regime"] else "NONE",
    )

    db.add(record)
    db.commit()
    db.close()

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(daily_job, "cron", hour=17, minute=30)  # after market close
    scheduler.start()
