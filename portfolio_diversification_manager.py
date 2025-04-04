import pandas as pd
import numpy as np
import cvxpy as cp

class PortfolioDiversificationManager:
    def __init__(self, asset_data, risk_tolerance=0.05):
        """
        Initializes the portfolio diversification manager with asset data and risk tolerance.

        :param asset_data: DataFrame containing historical price data for each asset.
        :param risk_tolerance: Maximum allowable portfolio volatility.
        """
        self.asset_data = asset_data
        self.risk_tolerance = risk_tolerance
        self.asset_returns = self.calculate_asset_returns()
        self.cov_matrix = self.asset_returns.cov()

    def calculate_asset_returns(self):
        """
        Calculates the historical returns for each asset.

        :return: DataFrame of asset returns.
        """
        return self.asset_data.pct_change().dropna()

    def optimize_portfolio(self):
        """
        Optimizes the portfolio allocation to maximize diversification while adhering to risk tolerance.

        :return: Dictionary with optimal asset allocations.
        """
        num_assets = len(self.asset_data.columns)
        weights = cp.Variable(num_assets)
        expected_returns = self.asset_returns.mean().values
        portfolio_return = expected_returns @ weights
        portfolio_volatility = cp.quad_form(weights, self.cov_matrix.values)

        objective = cp.Maximize(portfolio_return)
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            portfolio_volatility <= self.risk_tolerance
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        optimal_weights = weights.value
        allocation = dict(zip(self.asset_data.columns, optimal_weights))

        return allocation

    def rebalance_portfolio(self, current_allocations, new_allocations):
        """
        Determines the trades needed to rebalance the portfolio to new allocations.

        :param current_allocations: Dictionary of current asset allocations.
        :param new_allocations: Dictionary of new asset allocations.
        :return: Dictionary of trades to execute.
        """
        trades = {}
        for asset in new_allocations:
            trades[asset] = new_allocations[asset] - current_allocations.get(asset, 0)
        return trades
