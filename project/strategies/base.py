from abc import ABC, abstractmethod

import pandas as pd


class StrategyBase(ABC):
    """
    An abstract base class for trading strategies.

    This class serves as a blueprint for all trading strategies. Each custom strategy
    class should inherit from `StrategyBase` and implement all abstract methods.

    Attributes:
        price_data (pd.DataFrame): A DataFrame containing price and potentially other
                                   market data needed for the strategy.
    """

    def __init__(self, price_data: pd.DataFrame):
        """
        Initializes the base strategy with price data.

        Parameters:
            price_data (pd.DataFrame): The market data required for the strategy,
                                       typically with columns like 'close', 'volume', etc.
        """
        self.price_data = price_data

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        Generates trading signals based on the strategy logic.

        This method must be implemented in subclasses and should return a DataFrame
        that contains the trading signals generated by the strategy.

        Output:
            pd.DataFrame: A DataFrame including the signals for executing trades.
                          Typically, a 'signal' column is added:
                          - 1 indicates a buy signal
                          - -1 indicates a sell signal
                          - 0 indicates no action
        """
        pass

    @abstractmethod
    def run_backtest(
        self,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        slippage_percent: float = 0.0,
        execution_lag: int = 0,
    ) -> pd.DataFrame:
        """
        Executes the backtest for the given strategy.

        This method must be implemented in subclasses and will simulate trading
        based on the generated signals. Common trading constraints such as transaction
        costs, slippage, and execution lag should be considered.

        Parameters:
            initial_balance (float, optional): The starting capital for the backtest. Default is 10000.0.
            transaction_cost (float, optional): The transaction cost as a percentage of trade value. Default is 0.001.
            slippage_percent (float, optional): The slippage as a percentage of price. Default is 0.0.
            execution_lag (int, optional): The number of periods to delay signal execution. Default is 0.

        Output:
            pd.DataFrame: A DataFrame containing the backtest results, typically with columns like:
                          - 'signal': Trading signals
                          - 'balance': Account balance over time
                          - 'returns': Strategy returns over time
        """
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        """
        Fetches the key performance metrics of the strategy.

        This method must be implemented in subclasses and should return the final metrics
        that summarize the strategy's performance, such as total returns or Sharpe ratio.

        Output:
            dict: A dictionary of performance metrics, for example:
                  {
                      "total_return": 12.4,
                      "final_balance": 11240.0,
                      "sharpe_ratio": 1.5
                  }
        """
        pass
