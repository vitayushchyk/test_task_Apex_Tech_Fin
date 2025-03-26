import logging

from core.data_loader import DataLoader

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

    top_pairs = data_loader.get_top_liquid_pairs(
        base_currency="BTC", limit=100
    )

    if len(top_pairs) < 100:
        logging.warning(
            "Could not retrieve exactly 100 pairs. Processing available pairs."
        )
    else:
        logging.info("Top 100 pairs to BTC retrieved successfully.")

    for pair in top_pairs:
        cached_data = data_loader.load_cached_data(pair)
        if not cached_data.empty:
            logging.info(f"Cached data found for {pair}. Skipping fetching.")
            continue

        logging.info(f"Fetching data for pair: {pair}")

        df = data_loader.fetch_ohlcv(
            pair, start_date="2025-02-01", end_date="2025-02-28"
        )

        if not df.empty:
            df = df[(df.index >= "2025-02-01") & (df.index <= "2025-02-28")]

            if not df.empty:
                data_loader.save_to_parquet(df, pair)
            else:
                logging.warning(
                    f"No valid data in the specified date range for {pair}."
                )
        else:
            logging.warning(f"No data found for {pair}.")
