
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Create directories automatically
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Indian stock market tickers
# NSE tickers use the .NS suffix in yfinance

STOCKS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "HDFC_BANK": "HDFCBANK.NS",
    "ICICI_BANK": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "ITC": "ITC.NS",
    "LARSEN_TOUBRO": "LT.NS",
    "BHARTI_AIRTEL": "BHARTIARTL.NS",
        "TATA_MOTORS_PV": "TMPV.NS",
    "TATA_MOTORS_CV": "TMCV.NS",
}

# Historical data settings
START_DATE = "2021-01-01"

# yfinance end date is exclusive.
# None means fetch data up to the latest available date.
END_DATE = None

INTERVAL = "1d"