import os

import matplotlib.pyplot as plt
import pandas as pd


class Backtester:
    """
    The Backtester class for testing trading strategies.

    This class combines data loading, strategy execution, results processing,
    and saving metrics, charts, and detailed backtest results into files.

    Attributes:
        strategy_class (class): The trading strategy class to be tested.
        data_loader (object): The object responsible for loading price data.
        output_dir (str): Directory where the backtest results are stored.
        results (list): A list of metrics from all executed strategies.
    """

    def __init__(self, strategy_class, data_loader, output_dir="results/"):
        """
        Initializes the Backtester.

        Parameters:
            strategy_class (class): The trading strategy class.
            data_loader (object): The data loading object.
            output_dir (str, optional): The directory to store results.
                                        Default is "results/".
        """
        self.strategy_class = strategy_class
        self.data_loader = data_loader
        self.output_dir = output_dir
        self.results = []

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(
            os.path.join(self.output_dir, "screenshots"), exist_ok=True
        )

    def run(
        self,
        pair,
        transaction_cost=0.001,
        slippage_percent=0.0005,
        execution_lag=1,
        **strategy_params,
    ):
        """
        Runs the strategy with the specified parameters.

        Parameters:
            pair (str): The trading pair (e.g., "BTC/USD").
            transaction_cost (float, optional): Transaction costs in percentage (default 0.001).
            slippage_percent (float, optional): Slippage percentage (default 0.0005).
            execution_lag (int, optional): Execution delay (default 1 period).
            **strategy_params: Additional parameters passed to the strategy.

        Output:
            Metrics and detailed backtest results are saved into files.
        """

        price_data = self.data_loader.load_cached_data(pair)

        strategy = self.strategy_class(price_data, **strategy_params)
        backtest_results = strategy.run_backtest()
        metrics = strategy.get_metrics()

        metrics["transaction_cost"] = transaction_cost
        metrics["slippage_percent"] = slippage_percent
        metrics["execution_lag"] = execution_lag

        self.calculate_metrics(metrics, strategy_params)
        self._save_equity_curve(backtest_results, strategy_params)
        self._save_results(backtest_results, strategy_params)

    def calculate_metrics(self, metrics, strategy_params):
        """
        Adds the strategy metrics to the overall results and saves them to `metrics.csv`.

        Parameters:
            metrics (dict): The strategy's performance metrics, including execution parameters.
            strategy_params (dict): The strategy's parameters used during execution.

        Output:
            Metrics are appended to the results list and saved as a CSV file.
        """
        metrics["strategy"] = self.strategy_class.__name__
        metrics.update(strategy_params)

        self.results.append(metrics)

        results_df = pd.DataFrame(self.results)
        results_df.to_csv(
            os.path.join(self.output_dir, "metrics.csv"), index=False
        )

    def _save_equity_curve(self, backtest_results, strategy_params):
        """
        Saves the equity curve chart (balance curve) as a PNG file.

        Parameters:
            backtest_results (pd.DataFrame): The backtest results containing the balance column.
            strategy_params (dict): The strategy parameters used during execution.

        Output:
            A PNG chart of the equity curve is saved in the `screenshots` subdirectory.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(backtest_results["balance"], label="Equity Curve")
        plt.title(
            f"Equity Curve: {self.strategy_class.__name__} - {strategy_params}"
        )
        plt.xlabel("Time")
        plt.ylabel("Balance")
        plt.legend()
        plt.grid()

        file_name = (
            f"{self.strategy_class.__name__}_{strategy_params}.png".replace(
                " ", "_"
            ).replace(":", "-")
        )
        plt.savefig(os.path.join(self.output_dir, "screenshots", file_name))
        plt.close()

    def _save_results(self, backtest_results, strategy_params):
        """
        Saves detailed backtest results as a CSV file.

        Parameters:
            backtest_results (pd.DataFrame): The full backtest results.
            strategy_params (dict): The strategy parameters used during execution.

        Output:
            A CSV file containing detailed results is saved in the main results directory.
        """

        file_name = f"{self.strategy_class.__name__}_results.csv".replace(
            " ", "_"
        )
        backtest_results.to_csv(
            os.path.join(self.output_dir, file_name), index=False
        )
