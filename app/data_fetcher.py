# app/data_fetcher.py

import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def fetch_asi_data(days=300):
    try:
        from tvDatafeed import TvDatafeed, Interval
    except ImportError:
        logger.error("Install tvDatafeed")
        return None

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    tv = TvDatafeed()
    symbols = ['ASI', 'ASI.N0000', 'CSEASI', 'CSE:ASI']
    n_bars = days + 100

    for symbol in symbols:
        try:
            data = tv.get_hist(
                symbol=symbol,
                exchange='CSELK',
                interval=Interval.in_daily,
                n_bars=n_bars
            )

            if data is not None and not data.empty:
                data = data.reset_index()
                data['datetime'] = pd.to_datetime(data['datetime'])
                mask = (data['datetime'] >= start_date) & (data['datetime'] <= end_date)
                df = data.loc[mask].copy()

                df['date'] = df['datetime'].dt.date
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                df.sort_values('date', inplace=True)
                return df

        except Exception as e:
            logger.warning(f"{symbol} failed: {e}")

    return None
