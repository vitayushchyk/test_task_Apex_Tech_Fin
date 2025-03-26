import logging

from core.backtester import Backtester
from core.data_loader import DataLoader
from strategies.rsi_bb import RsiBollingerStrategy
from strategies.sma_cross import SmaCrossoverStrategy
from strategies.vwap_reversion import VWAPReversionStrategy

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler(),
    ],
)

if __name__ == "__main__":
    data_loader = DataLoader(cache_dir="data")
    logging.info("DataLoader initialized.")

    top_pairs = data_loader.get_top_liquid_pairs(
        base_currency="BTC", limit=100
    )

    if len(top_pairs) < 100:
        logging.warning(
            "Could not retrieve exactly 100 pairs. Processing available pairs."
        )
    else:
        logging.info("Top 100 pairs by liquidity retrieved successfully.")

    for pair in top_pairs:
        logging.info(f"Processing pair: {pair}")

        cached_data = data_loader.load_cached_data(pair)
        if cached_data.empty:
            logging.warning(f"No data available for {pair}. Skipping...")
            continue

        cached_data = cached_data[cached_data["volume"] > 0]

        cached_data = cached_data[
            ~(
                (cached_data["open"] == cached_data["high"])
                & (cached_data["high"] == cached_data["low"])
                & (cached_data["low"] == cached_data["close"])
            )
        ]

        if cached_data.empty:
            logging.warning(f"Cleaned data for {pair} is empty. Skipping...")
            continue

        logging.info(f"Cleaned data prepared for {pair}.")

        strategies = [
            {
                "name": "SMA Crossover",
                "class": SmaCrossoverStrategy,
                "params": {"short_window": 10, "long_window": 50},
            },
            {
                "name": "RSI + Bollinger Bands",
                "class": RsiBollingerStrategy,
                "params": {"rsi_period": 14, "bb_period": 20},
            },
            {
                "name": "VWAP Reversion",
                "class": VWAPReversionStrategy,
                "params": {"threshold": 0.02},
            },
        ]

        for strategy in strategies:
            try:
                logging.info(f"Running {strategy['name']} for {pair}...")

                backtester = Backtester(
                    strategy_class=strategy["class"],
                    data_loader=data_loader,
                    output_dir=f"results/{strategy['class'].__name__}/{pair}",
                )

                backtester.run(pair=pair, **strategy["params"])

                logging.info(
                    f"Backtest complete for {strategy['name']} on pair {pair}."
                )

            except Exception as e:
                logging.error(
                    f"Error while executing {strategy['name']} for {pair}: {e}"
                )

    logging.info("All strategies tested successfully!")
