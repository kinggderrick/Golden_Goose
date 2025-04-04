import pandas as pd
import numpy as np
import datetime

def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Computes the Sharpe ratio based on daily returns.
    Assumes 252 trading days per year.
    """
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return np.nan
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe

def compute_max_drawdown(equity_curve):
    """
    Computes the maximum drawdown given an equity curve.
    """
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def analyze_trades(trade_journal_path):
    """
    Analyzes the trade journal CSV file and computes performance metrics.
    The CSV file should have columns: time, symbol, direction, price, lots.
    """
    # Load the trade journal (CSV file with no header)
    try:
        df = pd.read_csv(trade_journal_path, names=['time', 'symbol', 'direction', 'price', 'lots'])
    except Exception as e:
        print(f"Error reading trade journal: {e}")
        return {}

    # Convert time column to datetime
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df.sort_values('time', inplace=True)
    
    if df.empty:
        print("No trades found in the journal.")
        return {}

    # For demonstration purposes, we assume a simplified P&L calculation.
    # NOTE: Replace this with your actual P&L calculation logic.
    # Here, we simulate a P&L by computing the difference in price multiplied by lot size,
    # assuming sequential trades (this is a placeholder).
    df['pnl'] = df['price'].diff() * df['lots']
    df['pnl'].fillna(0, inplace=True)
    df['cumulative_pnl'] = df['pnl'].cumsum()

    # Calculate daily returns based on changes in cumulative P&L.
    # Here, we assume that each trade represents a discrete return.
    df['return'] = df['cumulative_pnl'].pct_change().fillna(0)
    returns = df['return'].replace([np.inf, -np.inf], 0)

    sharpe_ratio = compute_sharpe_ratio(returns)
    max_drawdown = compute_max_drawdown(df['cumulative_pnl'])
    win_rate = (df['pnl'] > 0).mean() * 100
    total_trades = len(df)
    total_pnl = df['cumulative_pnl'].iloc[-1]

    report = {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_pnl': total_pnl
    }
    return report

if __name__ == "__main__":
    journal_path = "backups/trade_journal.csv"  # Ensure this path is correct relative to your repository
    report = analyze_trades(journal_path)
    if report:
        print("Performance Report:")
        for metric, value in report.items():
            print(f"{metric}: {value}")
    else:
        print("No performance data available.")
