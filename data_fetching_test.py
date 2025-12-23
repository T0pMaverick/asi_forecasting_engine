import pandas as pd
import os
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_asi_data(days=300):
    """
    Fetch ASI data for the last days days.
    """
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
                final_df = filtered_data[['date', 'open', 'high', 'low', 'close', 'volume']].copy() # Keep OHLCV
                final_df.sort_values('date', inplace=True)
                
                return final_df

        except Exception as e:
            logger.warning(f"Failed for {symbol}: {e}")

    logger.error("Failed to fetch ASI data from all symbols.")
    return None

def main():
    days = 300
    df = fetch_asi_data(days)
    
    if df is not None:
        filename = f"ASI_last_{days}days{datetime.now().strftime('%Y-%m-%d')}.csv"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved to {file_path}")
        logger.info(f"Rows: {len(df)}")
        logger.info(f"Date Range: {df['date'].min()} to {df['date'].max()}")
        print(f"\n[SUCCESS] Extracted ASI data to {filename}")
    else:
        print("\n[FAILED] Could not extract ASI data.")

if __name__ == "__main__":
    main()