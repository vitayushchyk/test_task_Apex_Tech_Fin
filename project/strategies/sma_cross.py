import logging

import numpy as np
import pandas as pd

from project.strategies.base import StrategyBase

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SmaCrossoverStrategy(StrategyBase):
    """
    A trading strategy based on the Simple Moving Average (SMA) crossover method.

    This strategy generates a buy signal when the short-term SMA crosses above
    the long-term SMA and closes the position when the reverse occurs.

    Attributes:
        price_data (pd.DataFrame): A DataFrame containing the price data with a 'close' column.
        short_window (int): The period for the short-term SMA.
        long_window (int): The period for the long-term SMA.
    """

    def __init__(
        self,
        price_data: pd.DataFrame,
        short_window: int = 10,
        long_window: int = 50,
    ):
        """
        Initialize the SmaCrossoverStrategy class with moving average parameters.

        Parameters:
            price_data (pd.DataFrame): The price data, must include a 'close' column.
            short_window (int): The period for the short-term SMA. Default is 10.
            long_window (int): The period for the long-term SMA. Default is 50.
        """
        super().__init__(price_data)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on SMA crossover.

        The method calculates short-term and long-term moving averages and
        generates a trading signal ('signal' column) where:
        - 1: Indicates a long/buy signal (entry into position).
        - 0: Indicates closing a position.

        Returns:
            pd.DataFrame: The DataFrame containing the original data with additional
                          SMA columns and a 'signal' column.
        """
        logging.info("Generating signals using SMA Crossover Strategy...")
        data = self.price_data.copy()

        data["SMA_Short"] = (
            data["close"].rolling(window=self.short_window).mean()
        )
        data["SMA_Long"] = (
            data["close"].rolling(window=self.long_window).mean()
        )

        data["signal"] = np.where(data["SMA_Short"] > data["SMA_Long"], 1, 0)
        data["signal"] = data["signal"].diff()

        logging.info(
            f"Signals generated. Last 5 rows:\n{data[['SMA_Short', 'SMA_Long', 'signal']].tail()}"
        )
        return data

    def run_backtest(self, initial_balance: float = 10000.0) -> pd.DataFrame:
        """
        Runs a backtest simulation using the generated signals.

        This method calculates the strategy's returns based on price movements
        triggered by the SMA crossover signals.

        Parameters:
            initial_balance (float): The starting balance for the backtest. Default is 10,000.

        Returns:
            pd.DataFrame: A DataFrame containing backtest results with:
                          - 'strategy_return': The compounded returns of the strategy.
                          - 'balance': Simulated account balance over time.
        """
        logging.info("Running backtest for SMA Crossover Strategy...")
        data = self.generate_signals()

        data["strategy_return"] = (
            data["signal"].shift() * data["close"].pct_change()
        )

        data["balance"] = (
            initial_balance * (1 + data["strategy_return"]).cumprod()
        )

        logging.info(
            f"Backtest completed. Final balance: {data['balance'].iloc[-1]:.2f}"
        )
        logging.info(
            f"Last 5 rows of backtest:\n{data[['signal', 'balance']].tail()}"
        )

        return data

    def get_metrics(self) -> dict:
        """
        Calculate and return performance metrics for the strategy.

        This method runs the backtest and computes metrics such as total return
        and final account balance.

        Returns:
            dict: A dictionary containing:
                  - 'total_return': The total return of the strategy in monetary terms.
                  - 'final_balance': The final balance at the end of the backtest.
        """
        logging.info("Calculating strategy metrics...")
        backtest = self.run_backtest()

        total_return = (
            backtest["balance"].iloc[-1] - backtest["balance"].iloc[0]
        )
        final_balance = backtest["balance"].iloc[-1]

        return {"total_return": total_return, "final_balance": final_balance}
