import time
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def fetch_asi_data(
    days: int = 1200,
    end_date=None,
    max_retries: int = 10,
    retry_delay: int = 5,   # seconds
):
    """
    Fetch ASI data up to end_date (inclusive) with retries.

    - Retries each symbol up to max_retries
    - Waits retry_delay seconds between retries
    """

    try:
        from tvDatafeed import TvDatafeed, Interval
    except ImportError:
        logger.error("tvDatafeed not installed")
        return None

    if end_date is None:
        end_date = datetime.now()
    else:
        end_date = pd.to_datetime(end_date)

    start_date = end_date - timedelta(days=days)

    tv = TvDatafeed()
    symbols = ["ASI", "ASI.N0000", "CSEASI", "CSE:ASI"]
    n_bars = days + 100

    for symbol in symbols:
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Fetching {symbol} (attempt {attempt}/{max_retries})")

                data = tv.get_hist(
                    symbol=symbol,
                    exchange="CSELK",
                    interval=Interval.in_daily,
                    n_bars=n_bars,
                )

                if data is None or data.empty:
                    raise ValueError("Empty response")

                data = data.reset_index()
                data["datetime"] = pd.to_datetime(data["datetime"])

                mask = (data["datetime"] >= start_date) & (data["datetime"] <= end_date)
                df = data.loc[mask].copy()

                if df.empty:
                    raise ValueError("No data after date filtering")

                df["date"] = df["datetime"].dt.date
                df = df[["date", "open", "high", "low", "close", "volume"]]
                df.sort_values("date", inplace=True)

                logger.info(f"Successfully fetched ASI data using {symbol}")
                return df

            except Exception as e:
                logger.warning(
                    f"{symbol} failed (attempt {attempt}/{max_retries}): {e}"
                )

                if attempt < max_retries:
                    time.sleep(retry_delay)

    logger.error("All symbols failed. ASI data fetch unsuccessful.")
    return None
