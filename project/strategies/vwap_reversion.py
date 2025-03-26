import logging

import numpy as np
import pandas as pd

from strategies.base import StrategyBase

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class VWAPReversionStrategy(StrategyBase):
    """
    VWAP Reversion Strategy.

    This strategy generates signals based on the deviation of the price from the VWAP
    (Volume Weighted Average Price). When the price moves sufficiently below the VWAP,
    it signals a 'buy', and when the price moves above, it signals a 'sell'.

    Attributes:
        price_data (pd.DataFrame): A DataFrame containing 'close' and 'volume' columns.
        threshold (float): Threshold for deviation from VWAP. Default is 2% (0.02).
    """

    def __init__(self, price_data: pd.DataFrame, threshold: float = 0.02):
        """
        Initialize the VWAP Reversion Strategy.

        Parameters:
            price_data (pd.DataFrame): A DataFrame containing columns ['close', 'volume'].
            threshold (float): Threshold for deviation from VWAP. Default is 2% (0.02).
        """
        super().__init__(price_data)
        self.threshold = threshold

    def _calculate_vwap(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP (Volume Weighted Average Price).

        VWAP is calculated using the cumulative typical price * volume divided
        by the cumulative volume.

        Parameters:
            data (pd.DataFrame): A DataFrame containing 'close' and 'volume'.

        Returns:
            pd.DataFrame: The input DataFrame with an additional 'vwap' column.
        """
        data = data.copy()
        data["tpv"] = data["close"] * data["volume"]
        data["cum_tpv"] = data["tpv"].cumsum()
        data["cum_volume"] = data["volume"].cumsum()
        data["vwap"] = data["cum_tpv"] / data["cum_volume"]
        logging.info("VWAP calculated and added to the DataFrame.")
        return data

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on deviation from VWAP.

        Uses the computed VWAP and the deviation threshold to generate buy/sell signals:
        - Buy when the price is sufficiently below the VWAP (negative deviation).
        - Sell when the price is sufficiently above the VWAP (positive deviation).

        Returns:
            pd.DataFrame: The input DataFrame with the additional columns:
                          - 'deviation': Deviation from VWAP.
                          - 'dynamic_threshold': Adjusted threshold based on volatility.
                          - 'signal': Generated trading signals.
        """
        logging.info("Generating signals based on VWAP deviation...")
        data = self._calculate_vwap(self.price_data)

        data["deviation"] = (data["close"] - data["vwap"]) / data["vwap"]

        data["volatility"] = data["close"].rolling(window=14).std()
        data["dynamic_threshold"] = self.threshold * (
            data["volatility"] / data["volatility"].mean()
        )

        # Generate buy/sell signals
        data["signal"] = 0
        data.loc[
            data["deviation"] < -data["dynamic_threshold"], "signal"
        ] = 1  # Buy
        data.loc[
            data["deviation"] > data["dynamic_threshold"], "signal"
        ] = -1  # Sell

        data["signal"] = data["signal"].shift()
        logging.info(
            "Signals generated. Last 5 rows of signals:\n%s",
            data[["close", "vwap", "deviation", "signal"]].tail(),
        )

        return data

    def run_backtest(
        self, initial_balance: float = 10000.0, transaction_cost: float = 0.001
    ) -> pd.DataFrame:
        """
        Run backtest for the strategy, including trading costs.

        Parameters:
            initial_balance (float): The starting balance for the strategy. Default is 10,000.
            transaction_cost (float): Trading cost per trade as a percentage. Default is 0.001.

        Returns:
            pd.DataFrame: A DataFrame with backtest results and additional columns:
                          - 'strategy_return': Returns of the strategy.
                          - 'balance': The simulated account balance over time.
        """
        logging.info(
            "Running backtest with initial_balance=%.2f and transaction_cost=%.4f...",
            initial_balance,
            transaction_cost,
        )
        data = self.generate_signals()

        data["return"] = data["close"].pct_change()
        data["strategy_return"] = data["signal"] * data["return"]

        trade_mask = data["signal"] != 0
        data.loc[trade_mask, "strategy_return"] -= transaction_cost

        data["balance"] = (
            initial_balance * (1 + data["strategy_return"]).cumprod()
        )

        logging.info(
            "Backtest finished. Final balance: %.2f", data["balance"].iloc[-1]
        )
        logging.info(
            "Last 5 rows of backtest:\n%s",
            data[["signal", "balance", "strategy_return"]].tail(),
        )

        return data

    def get_metrics(self) -> dict:
        """
        Calculate and return key performance metrics for the strategy.

        Metrics include:
        - Total return
        - Maximum drawdown
        - Sharpe ratio
        - Final balance

        Returns:
            dict: A dictionary containing the calculated performance metrics.
        """
        logging.info("Calculating strategy performance metrics...")
        backtest = self.run_backtest()
        total_return = (
            backtest["balance"].iloc[-1] / backtest["balance"].iloc[0]
        ) - 1
        max_drawdown = (
            backtest["balance"] / backtest["balance"].cummax() - 1
        ).min()
        sharpe_ratio = (
            np.mean(backtest["strategy_return"])
            / np.std(backtest["strategy_return"])
            * np.sqrt(252)
            if np.std(backtest["strategy_return"]) > 0
            else 0
        )

        # Log metrics
        logging.info(
            "Metrics calculated: Total Return = %.2f, Sharpe Ratio = %.2f, Max Drawdown = %.2f, Final Balance = %.2f",
            total_return,
            sharpe_ratio,
            max_drawdown,
            backtest["balance"].iloc[-1],
        )

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "final_balance": backtest["balance"].iloc[-1],
        }
