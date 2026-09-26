
import logging
import time

import pandas as pd
import yfinance as yf

from src.config import (
    STOCKS,
    START_DATE,
    END_DATE,
    INTERVAL,
    RAW_DATA_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def fetch_stock_data(
    company_name: str,
    ticker_symbol: str,
) -> pd.DataFrame:
    """
    Fetch historical daily stock data for one stock.

    Returns a DataFrame containing:
    Open, High, Low, Close, Volume.
    """

    logger.info(
        "Fetching data for %s (%s)",
        company_name,
        ticker_symbol,
    )

    try:
        ticker = yf.Ticker(ticker_symbol)

        data = ticker.history(
            start=START_DATE,
            end=END_DATE,
            interval=INTERVAL,
            auto_adjust=True,
            actions=False,
        )

        if data.empty:
            logger.warning(
                "No data returned for %s",
                ticker_symbol,
            )
            return pd.DataFrame()

        # Keep only the required OHLCV columns
        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        missing_columns = [
            col for col in required_columns
            if col not in data.columns
        ]

        if missing_columns:
            logger.warning(
                "Missing columns for %s: %s",
                ticker_symbol,
                missing_columns,
            )
            return pd.DataFrame()

        data = data[required_columns].copy()

        # Convert the date index into a column
        data.index.name = "Date"
        data = data.reset_index()

        # Remove duplicate dates and sort chronologically
        data = data.drop_duplicates(subset=["Date"])
        data = data.sort_values("Date")

        # Remove rows with missing essential values
        data = data.dropna(
            subset=["Open", "High", "Low", "Close", "Volume"]
        )

        # Save the stock dataset
        output_file = RAW_DATA_DIR / f"{company_name}.csv"

        data.to_csv(output_file, index=False)

        logger.info(
            "Saved %s rows to %s",
            len(data),
            output_file,
        )

        return data

    except Exception as error:
        logger.error(
            "Failed to fetch %s: %s",
            ticker_symbol,
            error,
        )
        return pd.DataFrame()


def collect_all_stocks() -> dict:
    """
    Download historical data for all configured stocks.
    """

    results = {}

    for company_name, ticker_symbol in STOCKS.items():

        data = fetch_stock_data(
            company_name,
            ticker_symbol,
        )

        if not data.empty:
            results[company_name] = data

        # Short delay between requests
        time.sleep(1)

    return results


if __name__ == "__main__":

    stock_data = collect_all_stocks()

    print("\n========== DATA COLLECTION SUMMARY ==========")

    total_rows = 0

    for company, data in stock_data.items():
        print(f"{company}: {len(data)} records")
        total_rows += len(data)

    print("---------------------------------------------")
    print(f"Successful stocks: {len(stock_data)}")
    print(f"Total records: {total_rows}")
    print(f"Output directory: {RAW_DATA_DIR}")