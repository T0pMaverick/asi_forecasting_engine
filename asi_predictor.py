#!/usr/bin/env python3
"""
ASI (All Share Index) Close Prices Fetcher - Last 15 Years
Fetches ASI close prices from last 15 years to present and saves to CSV
"""

import pandas as pd
import os
from datetime import datetime, timedelta
import logging

# TradingView data source
try:
    from tvDatafeed import TvDatafeed, Interval
    TV_AVAILABLE = True
except ImportError:
    print("tvDatafeed not available. Install with: pip install tvDatafeed")
    TV_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('asi_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ASIDataFetcher:
    """ASI close prices fetcher for last 15 years"""
    
    def __init__(self):
        # Calculate date range - last 15 years
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=15*365)  # 15 years
        
        self.start_date_str = self.start_date.strftime('%Y-%m-%d')
        self.end_date_str = self.end_date.strftime('%Y-%m-%d')
        
        self.symbol = 'ASI'
        self.csv_file = f'ASI_close_prices_last_15_years_{self.end_date_str}.csv'
        
        logger.info(f"Initialized fetcher for {self.symbol}")
        logger.info(f"Date range: {self.start_date_str} to {self.end_date_str}")
        
    def fetch_from_tradingview(self) -> pd.DataFrame:
        """Fetch ASI close prices from TradingView"""
        if not TV_AVAILABLE:
            logger.warning("TradingView not available")
            return pd.DataFrame()
            
        try:
            logger.info(f"Fetching ASI data from TradingView...")
            tv = TvDatafeed()
            
            # Calculate number of days needed (add buffer for weekends/holidays)
            n_bars = (self.end_date - self.start_date).days + 100  # Extra buffer
            
            # Try different symbol formats for ASI
            tv_symbols = [
                'ASI',           # Standard format
                'ASI.N0000',     # With suffix
                'CSEASI',        # Alternative format
                'CSE:ASI',       # Exchange prefix format
            ]
            
            for tv_symbol in tv_symbols:
                try:
                    logger.info(f"Trying TradingView symbol: {tv_symbol}")
                    data = tv.get_hist(
                        symbol=tv_symbol,
                        exchange='CSELK',
                        interval=Interval.in_daily,
                        n_bars=n_bars
                    )
                    
                    if data is not None and not data.empty:
                        # Process the data
                        data = data.reset_index()
                        data['date'] = pd.to_datetime(data['datetime']).dt.date
                        
                        # Filter to last 15 years
                        data = data[data['date'] >= self.start_date.date()]
                        data = data[data['date'] <= self.end_date.date()]
                        
                        # Select only date and close columns
                        close_data = data[['date', 'close']].copy()
                        close_data.columns = ['date', 'close']
                        
                        # Sort by date
                        close_data = close_data.sort_values('date').reset_index(drop=True)
                        
                        logger.info(f"[SUCCESS] Fetched {len(close_data)} records from TradingView")
                        logger.info(f"Date range: {close_data['date'].min()} to {close_data['date'].max()}")
                        return close_data
                        
                except Exception as e:
                    logger.warning(f"Failed with symbol {tv_symbol}: {e}")
                    continue
            
            logger.warning("TradingView fetch failed for all symbol formats")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching from TradingView: {e}")
            return pd.DataFrame()
    
    def save_to_csv(self, data: pd.DataFrame) -> str:
        """Save data to CSV file"""
        if data.empty:
            logger.error("No data to save")
            return ""
        
        try:
            # Ensure we're in the ASI_prediction directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, self.csv_file)
            
            # Save to CSV
            data.to_csv(csv_path, index=False)
            logger.info(f"[SUCCESS] Saved data to {csv_path}")
            
            # Print summary
            logger.info("="*60)
            logger.info("DATA SUMMARY")
            logger.info("="*60)
            logger.info(f"Total records: {len(data)}")
            logger.info(f"Date range: {data['date'].min()} to {data['date'].max()}")
            logger.info(f"Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
            logger.info(f"First close price: {data['close'].iloc[0]:.2f}")
            logger.info(f"Last close price: {data['close'].iloc[-1]:.2f}")
            
            # Calculate price change
            price_change = data['close'].iloc[-1] - data['close'].iloc[0]
            price_change_pct = (price_change / data['close'].iloc[0]) * 100
            logger.info(f"Total price change: {price_change:.2f} ({price_change_pct:+.2f}%)")
            
            # Calculate annualized return
            years = (data['date'].max() - data['date'].min()).days / 365.25
            if years > 0:
                annualized_return = ((data['close'].iloc[-1] / data['close'].iloc[0]) ** (1/years) - 1) * 100
                logger.info(f"Annualized return: {annualized_return:.2f}%")
            
            return csv_path
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            return ""
    
    def run(self) -> str:
        """Run the complete fetching process"""
        logger.info("="*60)
        logger.info(f"Fetching ASI close prices for last 15 years")
        logger.info(f"Date range: {self.start_date_str} to {self.end_date_str}")
        logger.info("="*60)
        
        data = self.fetch_from_tradingview()
        
        if not data.empty:
            csv_file = self.save_to_csv(data)
            if csv_file:
                logger.info("="*60)
                logger.info("[SUCCESS] DATA FETCHING COMPLETED SUCCESSFULLY!")
                logger.info("="*60)
                return csv_file
        
        logger.error("[FAILED] DATA FETCHING FAILED")
        return ""


def main():
    """Main function"""
    print("\n" + "="*60)
    print("ASI (All Share Index) Close Prices Fetcher - Last 15 Years")
    print("="*60 + "\n")
    
    fetcher = ASIDataFetcher()
    csv_file = fetcher.run()
    
    if csv_file:
        print(f"\n[SUCCESS] Data extraction completed!")
        print(f"Data saved to: {csv_file}")
        print(f"Check 'asi_fetcher.log' for detailed information\n")
    else:
        print(f"\n[FAILED] Data extraction failed!")
        print(f"Check 'asi_fetcher.log' for error details\n")
        print("Note: Make sure tvDatafeed is installed: pip install tvDatafeed")


if __name__ == "__main__":
    main()

