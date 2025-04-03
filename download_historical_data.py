import yfinance as yf
import pandas as pd
from datetime import datetime

# Function to attempt downloading data for a given ticker
def download_data(ticker, start_date, end_date):
    print(f"Downloading historical data for {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=False)
        return data
    except Exception as e:
        print(f"Error downloading data for {ticker}: {str(e)}")
        return None

# Set primary and alternative ticker symbols
primary_ticker = "XAUUSD=X"   # Primary ticker (may fail on some systems)
alternative_ticker = "GC=F"     # Alternative ticker for Gold Futures

start_date = "2000-01-01"  # Adjust as needed (covers 5-10+ years)
end_date = datetime.today().strftime('%Y-%m-%d')

# Attempt to download with the primary ticker
data = download_data(primary_ticker, start_date, end_date)

# If no data was downloaded, try the alternative ticker
if data is None or data.empty:
    print(f"No data found for ticker {primary_ticker}. Trying alternative ticker {alternative_ticker}...")
    data = download_data(alternative_ticker, start_date, end_date)

# Check if data is still empty and exit if so
if data is None or data.empty:
    print("Error: No data was downloaded for either ticker. Please check your internet connection or the ticker symbols.")
    exit(1)

# Reset index so that 'Date' becomes a column
data.reset_index(inplace=True)

# Save raw historical data to CSV
raw_csv = "historical_data.csv"
data.to_csv(raw_csv, index=False)
print(f"Historical data saved to {raw_csv}")

# --- Basic Analysis ---
print("\nPerforming basic data analysis...")

# Display descriptive statistics
analysis_summary = data.describe()
print("Descriptive Statistics:")
print(analysis_summary)

# Calculate moving averages (50-day and 200-day)
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

# Calculate daily returns
data['Daily_Return'] = data['Close'].pct_change()

# Save the data with analysis (moving averages, returns) to a new CSV file
analysis_csv = "historical_data_analysis.csv"
data.to_csv(analysis_csv, index=False)
print(f"Analysis complete. Data with moving averages and daily returns saved to {analysis_csv}")
