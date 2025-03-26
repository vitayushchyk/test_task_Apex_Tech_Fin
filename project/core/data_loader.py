import logging
import os
from typing import List

import ccxt
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data_loader.log"), logging.StreamHandler()],
)


class DataLoader:
    """
    A utility class for loading and caching Binance OHLCV data in the specified directory.
    Supports data fetching, caching in Parquet format, and retrieval of top liquid trading pairs.
    """

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        cache_dir: str = "data",
    ):
        """
        Initializes the DataLoader instance with the Binance API and sets up a caching directory for data.

        :param api_key: Optional Binance API key for private endpoints. Defaults to None.
        :param api_secret: Optional Binance API secret for private endpoints. Defaults to None.
        :param cache_dir: Directory for caching or saving data. Defaults to "data".
        """

        self.exchange = ccxt.binance({"apiKey": api_key, "secret": api_secret})
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        logging.info(
            f"Initialized DataLoader with cache directory: {self.cache_dir}"
        )

    def get_top_liquid_pairs(
        self, base_currency: str = "BTC", limit: int = 100
    ) -> List[str]:
        """
        Retrieves the top liquid trading pairs for a specified base currency, sorted by 24-hour trading volume.

        :param base_currency: The base currency to search pairs for (e.g., 'BTC'). Defaults to "BTC".
        :param limit: Maximum number of trading pairs to return. Defaults to 100.
        :return: A list of trading pair strings (symbols).
        """

        try:
            markets = self.exchange.fetch_tickers()
            filtered = [
                market
                for market in markets.values()
                if market["symbol"].endswith(f"/{base_currency}")
            ]
            sorted_markets = sorted(
                filtered, key=lambda x: x["quoteVolume"], reverse=True
            )
            top_pairs = [m["symbol"] for m in sorted_markets[:limit]]

            if len(top_pairs) < limit:
                logging.warning(
                    f"Found only {len(top_pairs)} pairs to {base_currency} (expected: {limit})."
                )
            else:
                logging.info(
                    f"Successfully retrieved {len(top_pairs)} pairs to {base_currency}."
                )

            return top_pairs
        except Exception as e:
            logging.error(
                f"Error retrieving top pairs for {base_currency}: {e}"
            )
            return []

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """
        Fetches OHLCV (Open, High, Low, Close, Volume) data for a specified trading pair, timeframe, and date range.

        :param symbol: The trading pair symbol (e.g., 'ETH/BTC').
        :param start_date: Start date in 'YYYY-MM-DD' format.
        :param end_date: End date in 'YYYY-MM-DD' format.
        :param timeframe: The timeframe for OHLCV data (e.g., '1m', '1h', '1d'). Defaults to '1m'.
        :return: A DataFrame containing OHLCV data filtered by the specified date range. Returns an empty DataFrame if no data is found.
        """

        since = self.exchange.parse8601(f"{start_date}T00:00:00Z")
        end_ts = self.exchange.parse8601(f"{end_date}T23:59:59Z")
        all_data = []

        try:
            while since < end_ts:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=1000
                )
                if not ohlcv:
                    logging.warning(
                        f"No more data available for {symbol} from {start_date} to {end_date}."
                    )
                    break

                data = pd.DataFrame(
                    ohlcv,
                    columns=[
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ],
                )

                data["timestamp"] = pd.to_datetime(
                    data["timestamp"], unit="ms"
                )

                data = data[
                    (data["timestamp"] >= start_date)
                    & (data["timestamp"] <= end_date)
                ]

                if not data.empty:
                    all_data.append(data)

                since = ohlcv[-1][0] + 1

            if all_data:
                df = pd.concat(all_data)
                df.index = df["timestamp"]
                logging.info(
                    f"Fetched OHLCV data for {symbol} from {start_date} to {end_date}."
                )
                return df
            else:
                logging.warning(
                    f"No OHLCV data found for {symbol} in the provided date range."
                )
                return pd.DataFrame()

        except Exception as e:
            logging.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return pd.DataFrame()

    def save_to_parquet(self, df: pd.DataFrame, symbol: str):
        """
        Saves a pandas DataFrame as a compressed Parquet file in the cache directory.
        If the file for the trading pair symbol already exists, it skips saving.

        :param df: The DataFrame to save.
        :param symbol: The trading pair symbol (e.g., 'ETH/BTC') used to name the file.
        """
        try:
            file_path = os.path.join(
                self.cache_dir, f"{symbol.replace('/', '_')}.parquet"
            )

            if os.path.exists(file_path):
                logging.info(
                    f"File already exists for {symbol}. Skipping save."
                )
                return

            df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

            df.to_parquet(file_path, compression="snappy")
            logging.info(f"Saved {symbol} data to {file_path}.")
        except Exception as e:
            logging.error(f"Error saving data for {symbol} to Parquet: {e}")

    def load_cached_data(self, symbol: str) -> pd.DataFrame:
        """
        Loads OHLCV data for a specified trading pair from a cached Parquet file.

        :param symbol: The trading pair symbol (e.g., 'ETH/BTC').
        :return: A DataFrame containing the cached data. Returns an empty DataFrame if the file does not exist.
        """

        try:
            file_path = os.path.join(
                self.cache_dir, f"{symbol.replace('/', '_')}.parquet"
            )
            if os.path.exists(file_path):
                logging.info(
                    f"Loaded cached data for {symbol} from {file_path}."
                )
                return pd.read_parquet(file_path)
            else:
                logging.warning(f"No cached data found for {symbol}.")
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading cached data for {symbol}: {e}")
            return pd.DataFrame()
