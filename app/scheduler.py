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


def run_engine(
    as_of_date: date | None = None,
    save: bool = True,
    run_type: str = "DAILY",
    lookback_rows: int | None = None
):
    """
    Core engine runner.

    run_type:
      - DAILY
      - INTRADAY_30M

    lookback_rows:
      - None for daily
      - ~200 for intraday
    """

    if cache.data is None or cache.data.empty:
        logger.error("ASI data cache is empty. Engine cannot run.")
        return None

    df = cache.data.copy()

    if as_of_date is not None:
        df = df[df["date"] <= as_of_date]

    if lookback_rows is not None:
        df = df.tail(lookback_rows)

    if len(df) < 300:
        logger.warning("Insufficient data for engine run.")
        return None

    results = run_risk_engine(df)
    if results is None:
        return None

    # Backtest mode
    if not save:
        return results

    db = SessionLocal()

    try:
        record = ASIAlert(
            run_date=as_of_date or date.today(),
            run_type=run_type,  # ✅ NEW

            # ===== Regime =====
            vol_regime=results["vol_regime"],

            # ===== Probabilities (SAVE THEM) =====
            prob_minus_5=results["prob_minus_5"],
            prob_minus_10=results["prob_minus_10"],
            prob_minus_20=results["prob_minus_20"],
            prob_plus_5=results["prob_plus_5"],
            prob_plus_10=results["prob_plus_10"],
            prob_plus_20=results["prob_plus_20"],

            # ===== Alerts (threshold-based) =====
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
        logger.info(
            f"Saved {run_type} ASI alert for {record.run_date}"
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save alert: {e}")

    finally:
        db.close()

    return results

def explain_results(results: dict) -> str:
    """
    Convert model outputs into layman-friendly explanation
    """

    vol_text = (
        "High volatility (unstable market)"
        if results["vol_regime"]
        else "Normal volatility (stable market)"
    )

    explanation = (
        f"Market Condition: {vol_text}\n\n"
        f"Downside Risk (next 10 days):\n"
        f"- {results['prob_minus_5']*100:.1f}% chance of a 5% drop\n"
        f"- {results['prob_minus_10']*100:.1f}% chance of a 10% drop\n"
        f"- {results['prob_minus_20']*100:.1f}% chance of a 20% drop\n\n"
        f"Upside Potential (next 10 days):\n"
        f"- {results['prob_plus_5']*100:.1f}% chance of a 5% rise\n"
        f"- {results['prob_plus_10']*100:.1f}% chance of a 10% rise\n"
        f"- {results['prob_plus_20']*100:.1f}% chance of a 20% rise"
    )

    return explanation



def intraday_job():
    from app.logger import write_log
    try:
        results = run_engine(
            as_of_date=None,
            save=True,
            run_type="INTRADAY_30M",
            lookback_rows=400
        )

        if results is None:
            raise RuntimeError("Model did not return results")

        explanation = explain_results(results)


        write_log(
            "INTRADAY_RUN",
            f"30-minute ASI forecast executed successfully.\n\n{explanation}",
            "SUCCESS"
        )

    except Exception as e:
        write_log(
            "INTRADAY_RUN",
            str(e),
            "FAILURE"
        )


def daily_job():
    from app.logger import write_log
    try:
        results = run_engine(
            as_of_date=None,
            save=True,
            run_type="DAILY"
        )

        if results is None:
            raise RuntimeError("Daily model did not return results")

        explanation = explain_results(results)

        write_log(
            "DAILY_RUN",
            f"Daily ASI forecast executed successfully.\n\n{explanation}",
            "SUCCESS"
        )

    except Exception as e:
        write_log(
            "DAILY_RUN",
            str(e),
            "FAILURE"
        )



def start_scheduler():
    global scheduler

    scheduler = BackgroundScheduler(timezone="Asia/Colombo")

    # 🔹 Intraday every 30 mins (9:30–14:30, Mon–Fri)
    scheduler.add_job(
        intraday_job,
        trigger="cron",
        day_of_week="mon-fri",
        hour="9-14",
        minute="0,30",
        id="asi_intraday_30m",
        replace_existing=True
    )

    # 🔹 Daily at 14:30 (Mon–Fri)
    scheduler.add_job(
        daily_job,
        trigger="cron",
        day_of_week="mon-fri",
        hour=14,
        minute=30,
        id="asi_daily",
        replace_existing=True
    )

    scheduler.start()

