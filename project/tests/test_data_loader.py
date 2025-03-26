import os
import unittest
from unittest.mock import MagicMock

import pandas as pd

from core.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.cache_dir = "test_data"
        self.data_loader = DataLoader(
            api_key="test_key",
            api_secret="test_secret",
            cache_dir=self.cache_dir,
        )

        self.data_loader.exchange = MagicMock()

    def tearDown(self):
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
            os.rmdir(self.cache_dir)

    def test_initialization(self):
        self.assertEqual(self.data_loader.cache_dir, self.cache_dir)
        self.assertTrue(os.path.exists(self.cache_dir))

    def test_get_top_liquid_pairs(self):
        self.data_loader.exchange.fetch_tickers.return_value = {
            "BTC/USDT": {"symbol": "BTC/USDT", "quoteVolume": 1000},
            "ETH/USDT": {"symbol": "ETH/USDT", "quoteVolume": 2000},
            "LTC/BTC": {"symbol": "LTC/BTC", "quoteVolume": 500},
        }

        pairs = self.data_loader.get_top_liquid_pairs(
            base_currency="USDT", limit=2
        )
        self.assertEqual(pairs, ["ETH/USDT", "BTC/USDT"])

    def test_fetch_ohlcv_no_data(self):
        self.data_loader.exchange.fetch_ohlcv.return_value = []
        result = self.data_loader.fetch_ohlcv(
            "BTC/USDT", "2023-01-01", "2023-01-02"
        )
        self.assertTrue(result.empty)

    def test_save_to_parquet_and_load_cache(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2023-01-01 00:00:00", "2023-01-01 00:01:00"]
                ),
                "open": [100, 101],
                "high": [110, 111],
                "low": [90, 91],
                "close": [105, 106],
                "volume": [1000, 2000],
            }
        )
        symbol = "BTC/USDT"

        self.data_loader.save_to_parquet(df, symbol)
        file_path = os.path.join(self.cache_dir, "BTC_USDT.parquet")
        self.assertTrue(os.path.exists(file_path))

        loaded_df = self.data_loader.load_cached_data(symbol)
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_load_cached_data_no_file(self):
        result = self.data_loader.load_cached_data("NON_EXISTENT_PAIR")
        self.assertTrue(result.empty)
