import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import argparse
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/download.log'),
        logging.StreamHandler()
    ]
)

def download_data(ticker, start_date, end_date, retries=3, delay=5):
    """Download historical data with retry mechanism"""
    logging.info(f"Downloading historical data for {ticker} from {start_date} to {end_date}...")
    
    for attempt in range(retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=False)
            if data is not None and not data.empty:
                logging.info(f"Successfully downloaded {len(data)} rows for {ticker}")
                return data
            logging.warning(f"Attempt {attempt+1}/{retries}: Empty data received for {ticker}")
        except Exception as e:
            logging.error(f"Attempt {attempt+1}/{retries}: Error downloading data for {ticker}: {str(e)}")
        
        if attempt < retries - 1:
            logging.info(f"Waiting {delay} seconds before retrying...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    
    return None

def save_data(data, filename):
    """Save data to CSV with proper directory structure"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data.to_csv(filename, index=False)
    logging.info(f"Data saved to {filename}")

def analyze_data(data):
    """Perform comprehensive data analysis"""
    analysis = {}
    
    # Basic statistics
    analysis['basic_stats'] = data['Close'].describe().to_dict()
    
    # Volatility measures
    data['daily_range'] = data['High'] - data['Low']
    data['daily_return'] = data['Close'].pct_change()
    analysis['avg_daily_range'] = data['daily_range'].mean()
    analysis['volatility'] = data['daily_return'].std() * (252 ** 0.5)  # Annualized
    
    # Trend indicators
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # Calculate monthly returns
    monthly_returns = data['Close'].resample('M').last().pct_change()
    analysis['monthly_returns'] = monthly_returns.describe().to_dict()
    
    # Calculate drawdowns
    data['peak'] = data['Close'].cummax()
    data['drawdown'] = (data['Close'] - data['peak']) / data['peak']
    analysis['max_drawdown'] = data['drawdown'].min()
    
    return data, analysis

def main():
    parser = argparse.ArgumentParser(description='Download and analyze historical market data')
    parser.add_argument('--ticker', default="XAUUSD=X", help='Primary ticker symbol')
    parser.add_argument('--alt-ticker', default="GC=F", help='Alternative ticker symbol')
    parser.add_argument('--start', default="2000-01-01", help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=None, help='End date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--output', default="data/historical_data.csv", help='Output file path')
    parser.add_argument('--analysis-output', default="data/historical_data_analysis.csv", help='Analysis output file path')
    parser.add_argument('--json-output', default="data/analysis_summary.json", help='JSON analysis summary path')
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Set end date to today if not specified
    end_date = args.end if args.end else datetime.today().strftime('%Y-%m-%d')
    
    # Try primary ticker
    data = download_data(args.ticker, args.start, end_date)
    
    # If primary fails, try alternative ticker
    if data is None or data.empty:
        logging.warning(f"No data found for {args.ticker}. Trying alternative {args.alt_ticker}...")
        data = download_data(args.alt_ticker, args.start, end_date)
    
    # Check if we have data
    if data is None or data.empty:
        logging.error("Failed to download data using both tickers. Please check your internet connection or the ticker symbols.")
        return 1
    
    # Process the data
    data.reset_index(inplace=True)
    
    # Convert index column name to lowercase if it's "Date"
    if "Date" in data.columns:
        data.rename(columns={"Date": "date"}, inplace=True)
    
    # Ensure datetime format
    if "date" in data.columns:
        if not pd.api.types.is_datetime64_any_dtype(data["date"]):
            data["date"] = pd.to_datetime(data["date"])
    
    # Save raw data
    save_data(data, args.output)
    
    # Analyze and save enhanced data
    analyzed_data, analysis_summary = analyze_data(data)
    save_data(analyzed_data, args.analysis_output)
    
    # Save analysis summary as JSON
    with open(args.json_output, 'w') as f:
        json.dump(analysis_summary, f, indent=4)
    
    # Print summary
    logging.info("\nAnalysis Summary:")
    logging.info(f"Period: {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
    logging.info(f"Data points: {len(data)}")
    logging.info(f"Average daily range: {analysis_summary['avg_daily_range']:.2f}")
    logging.info(f"Annualized volatility: {analysis_summary['volatility']*100:.2f}%")
    logging.info(f"Maximum drawdown: {analysis_summary['max_drawdown']*100:.2f}%")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        exit(1)