# app/scheduler.py

from datetime import date, datetime
from apscheduler.schedulers.background import BackgroundScheduler
import logging

from app.database import SessionLocal
from app.models import ASIAlert
from app.risk_engine import run_risk_engine
from app.cache import cache

logger = logging.getLogger(__name__)

scheduler: BackgroundScheduler | None = None

ALERT_THRESHOLDS = {
    "minus_5": 0.05,
    "minus_10": 0.11,
    "minus_20": 0.08,
    "plus_5": 0.23,
    "plus_10": 0.08,
    "plus_20": 0.05,
}


def run_engine(as_of_date: date | None = None, save: bool = True):
    """
    Core engine runner.

    - as_of_date = None  -> production mode (today)
    - save = False       -> backtest mode (no DB write)
    """

    if cache.data is None or cache.data.empty:
        logger.error("ASI data cache is empty. Engine cannot run.")
        return None

    df = cache.data.copy()
    
    if as_of_date is not None:
        df = df[df["date"] <= as_of_date]

    if len(df) < 300:
        logger.warning("Insufficient data for engine run.")
        return None
    results = run_risk_engine(df)

    if results is None:
        return None

    if not save:
        return results

    db = SessionLocal()

    try:
        record = ASIAlert(
            run_date=as_of_date or date.today(),
            vol_regime=results["vol_regime"],
            alert_minus_5="WATCH"
            if results["prob_minus_5"] >= ALERT_THRESHOLDS["minus_5"]
            else "NONE",

        alert_minus_10="WARNING"
            if results["prob_minus_10"] >= ALERT_THRESHOLDS["minus_10"]
            and results["vol_regime"]
            else "NONE",

        alert_minus_20="CRITICAL"
            if results["prob_minus_20"] >= ALERT_THRESHOLDS["minus_20"]
            and results["vol_regime"]
            else "NONE",

        alert_plus_5="WATCH"
            if results["prob_plus_5"] >= ALERT_THRESHOLDS["plus_5"]
            else "NONE",

        alert_plus_10="WARNING"
            if results["prob_plus_10"] >= ALERT_THRESHOLDS["plus_10"]
            and results["vol_regime"]
            else "NONE",

        alert_plus_20="CRITICAL"
            if results["prob_plus_20"] >= ALERT_THRESHOLDS["plus_20"]
            and results["vol_regime"]
            else "NONE",

        )

        db.add(record)
        db.commit()
        logger.info(f"Saved ASI alert for {record.run_date}")

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save alert: {e}")

    finally:
        db.close()

    return results


def daily_job():
    """
    Production daily job.
    Runs using today's date and saves results to DB.
    """
    logger.info("Running daily ASI risk engine...")
    run_engine(as_of_date=None, save=True)


def start_scheduler():
    """
    Starts APScheduler safely (prevents double start).
    """
    global scheduler

    if scheduler is not None and scheduler.running:
        logger.info("Scheduler already running. Skipping start.")
        return

    scheduler = BackgroundScheduler(timezone="Asia/Colombo")

    scheduler.add_job(
        daily_job,
        trigger="cron",
        hour=17,
        minute=30,
        id="asi_daily_risk_job",
        replace_existing=True
    )

    scheduler.start()
    logger.info("ASI daily scheduler started.")
