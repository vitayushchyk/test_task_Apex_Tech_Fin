import logging

import pandas as pd

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RsiBollingerStrategy:
    def __init__(
        self,
        price_data: pd.DataFrame,
        rsi_period: int = 14,
        bb_period: int = 20,
    ):
        """
        Initialize the RsiBollingerStrategy.

        :param price_data: DataFrame containing pricing data (must include a 'close' column).
        :param rsi_period: RSI calculation period.
        :param bb_period: Bollinger Bands calculation period.
        """
        self.price_data = price_data
        self.rsi_period = rsi_period
        self.bb_period = bb_period

    def generate_signals(self) -> pd.DataFrame:
        """
        Generates buy/sell signals based on RSI and Bollinger Bands.
        :return: DataFrame with additional 'RSI', 'BB_Upper', 'BB_Lower', and 'signal' columns.
        """
        data = self.price_data.copy()

        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.rsi_period).mean()
        loss = -delta.where(delta < 0, 0.0).rolling(self.rsi_period).mean()
        rs = gain / (loss + 1e-9)
        data["RSI"] = 100 - (100 / (1.0 + rs))

        data["BB_Mid"] = data["close"].rolling(window=self.bb_period).mean()
        data["BB_Upper"] = (
            data["BB_Mid"]
            + 2 * data["close"].rolling(window=self.bb_period).std()
        )
        data["BB_Lower"] = (
            data["BB_Mid"]
            - 2 * data["close"].rolling(window=self.bb_period).std()
        )

        data["signal"] = 0
        data.loc[
            (data["RSI"] < 40) & (data["close"] < data["BB_Lower"]), "signal"
        ] = 1
        data.loc[
            (data["RSI"] > 60) & (data["close"] > data["BB_Upper"]), "signal"
        ] = -1
        signal_summary = (
            f"Generated signals: Buy = {sum(data['signal'] == 1)}, Sell = {sum(data['signal'] == -1)}, "
            f"Total Entries = {data['signal'].sum()}"
        )
        logger.info(signal_summary)

        return data

    def run_backtest(
        self,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.0001,
        slippage_percent: float = 0.0001,
        execution_lag: int = 1,
    ) -> pd.DataFrame:
        """
        Runs backtesting on generated signals.

        :param initial_balance: Starting balance.
        :param transaction_cost: Transaction fee.
        :param slippage_percent: Price slippage percentage.
        :param execution_lag: Number of intervals delay for signal execution.
        :return: DataFrame with backtesting results.
        """
        data = self.generate_signals()

        if data["signal"].sum() == 0:
            logger.warning("No signals were generated during the backtest.")
            data["balance"] = initial_balance
            return data

        data["signal"] = data["signal"].shift(execution_lag)

        data["price_with_slippage"] = data["close"] * (
            1 + slippage_percent * data["signal"]
        )

        data["return_with_slippage"] = (
            data["signal"].shift() * data["price_with_slippage"].pct_change()
        )

        trade_mask = data["signal"] != 0
        data.loc[trade_mask, "return_with_slippage"] -= transaction_cost

        data["balance"] = (
            initial_balance
            * (1 + data["return_with_slippage"].fillna(0)).cumprod()
        )

        final_balance = data["balance"].iloc[-1]
        logger.info(
            f"Backtest complete: Final Balance = {final_balance:.2f}, Initial Balance = {initial_balance:.2f}"
        )

        return data

    def get_metrics(self) -> dict:
        """
        Calculate key strategy performance metrics.
        :return: Dictionary containing 'total_return' and 'final_balance'.
        """
        backtest_result = self.run_backtest()

        initial_balance = backtest_result["balance"].iloc[0]
        final_balance = backtest_result["balance"].iloc[-1]

        if initial_balance == 0 or final_balance == 0:
            total_return = 0
        else:
            total_return = (final_balance / initial_balance) - 1

        logger.info(
            f"Metrics calculated: Total Return = {total_return:.4f}, Final Balance = {final_balance:.2f}, "
            f"Initial Balance = {initial_balance:.2f}"
        )

        return {
            "total_return": total_return,
            "final_balance": final_balance,
        }
