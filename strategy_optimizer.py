import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute

# Placeholder for your trading strategy function
def trading_strategy(data, param1, param2):
    """
    Simulates a trading strategy with given parameters.
    Replace this function with your actual strategy logic.

    :param data: DataFrame with historical price data.
    :param param1: First parameter to optimize.
    :param param2: Second parameter to optimize.
    :return: Cumulative returns from the strategy.
    """
    # Example strategy logic (to be replaced with your own)
    data['signal'] = np.where(data['price'].rolling(window=int(param1)).mean() > data['price'].rolling(window=int(param2)).mean(), 1, -1)
    data['returns'] = data['price'].pct_change() * data['signal'].shift(1)
    cumulative_returns = (1 + data['returns']).cumprod()
    return cumulative_returns

def objective_function(params, data):
    """
    Objective function for optimization.

    :param params: Tuple of parameters (param1, param2).
    :param data: DataFrame with historical price data.
    :return: Negative of final cumulative return (since we minimize).
    """
    param1, param2 = params
    cumulative_returns = trading_strategy(data, param1, param2)
    return -cumulative_returns.iloc[-1]  # Negative for minimization

def optimize_strategy(data):
    """
    Optimizes the trading strategy parameters using brute-force search.

    :param data: DataFrame with historical price data.
    :return: Optimal parameters and corresponding performance.
    """
    # Define parameter ranges
    param_ranges = (slice(5, 50, 5), slice(10, 100, 10))  # Example ranges for param1 and param2

    # Perform brute-force optimization
    optimal_params = brute(objective_function, param_ranges, args=(data,), finish=None)

    # Evaluate performance with optimal parameters
    optimal_performance = -objective_function(optimal_params, data)

    return optimal_params, optimal_performance

if __name__ == "__main__":
    # Load historical data
    data = pd.read_csv('historical_data.csv')  # Ensure this file exists in your directory
    data['price'] = data['Close']  # Adjust based on your data's column names

    # Optimize strategy
    best_params, best_performance = optimize_strategy(data)
    print(f"Optimal Parameters: {best_params}")
    print(f"Best Performance (Cumulative Return): {best_performance}")

    # Visualize performance with optimal parameters
    optimal_returns = trading_strategy(data, *best_params)
    plt.figure(figsize=(10, 6))
    plt.plot(optimal_returns, label='Optimized Strategy')
    plt.title('Strategy Performance with Optimal Parameters')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()
