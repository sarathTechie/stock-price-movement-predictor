
import logging

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, STOCKS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# Prediction horizons in trading days
HORIZONS = [1, 5, 30]

# Minimum future return required to classify as UP or DOWN.
# These are initial project settings and can be tuned later.
THRESHOLDS = {
    1: 0.005,   # 0.5%
    5: 0.01,    # 1.0%
    30: 0.02,   # 2.0%
}


def calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate the Relative Strength Index."""

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    # RSI is 100 when there are gains but no losses
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)

    # RSI is 50 when both average gain and loss are zero
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)

    return rsi


def calculate_atr(
    data: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """Calculate Average True Range."""

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    return atr


def add_technical_indicators(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add technical indicators and historical features."""

    df = data.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # --------------------------------------------------
    # 1. DAILY RETURNS AND MOMENTUM
    # --------------------------------------------------

    df["Daily_Return"] = close.pct_change()

    df["ROC_5"] = close.pct_change(periods=5)
    df["ROC_10"] = close.pct_change(periods=10)
    df["ROC_20"] = close.pct_change(periods=20)

    # --------------------------------------------------
    # 2. SIMPLE MOVING AVERAGES
    # --------------------------------------------------

    for window in [5, 10, 20, 50, 100, 200]:

        sma = close.rolling(
            window=window,
            min_periods=window,
        ).mean()

        df[f"SMA_{window}"] = sma

        # Price relative to moving average
        df[f"Close_SMA_Ratio_{window}"] = close / sma - 1

    # --------------------------------------------------
    # 3. EXPONENTIAL MOVING AVERAGES
    # --------------------------------------------------

    for span in [12, 26, 50]:

        df[f"EMA_{span}"] = close.ewm(
            span=span,
            adjust=False,
        ).mean()

    df["EMA_12_26_Diff"] = (
        df["EMA_12"] / df["EMA_26"] - 1
    )

    # --------------------------------------------------
    # 4. RSI
    # --------------------------------------------------

    df["RSI_14"] = calculate_rsi(close, period=14)

    # --------------------------------------------------
    # 5. MACD
    # --------------------------------------------------

    macd = df["EMA_12"] - df["EMA_26"]

    macd_signal = macd.ewm(
        span=9,
        adjust=False,
    ).mean()

    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd - macd_signal

    # --------------------------------------------------
    # 6. BOLLINGER BANDS
    # --------------------------------------------------

    middle_band = close.rolling(20).mean()

    rolling_std = close.rolling(20).std()

    upper_band = middle_band + (2 * rolling_std)
    lower_band = middle_band - (2 * rolling_std)

    df["BB_Upper"] = upper_band
    df["BB_Lower"] = lower_band

    df["BB_Width"] = (
        (upper_band - lower_band) / middle_band
    )

    df["BB_Position"] = (
        (close - lower_band) /
        (upper_band - lower_band).replace(0, np.nan)
    )

    # --------------------------------------------------
    # 7. ATR AND VOLATILITY
    # --------------------------------------------------

    df["ATR_14"] = calculate_atr(df, period=14)

    df["ATR_Ratio"] = df["ATR_14"] / close

    df["Volatility_20"] = (
        df["Daily_Return"]
        .rolling(20)
        .std()
    )

    # --------------------------------------------------
    # 8. STOCHASTIC OSCILLATOR
    # --------------------------------------------------

    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()

    denominator = (highest_high - lowest_low).replace(
        0, np.nan
    )

    df["Stochastic_K"] = (
        100 * (close - lowest_low) / denominator
    )

    df["Stochastic_D"] = (
        df["Stochastic_K"].rolling(3).mean()
    )

    # --------------------------------------------------
    # 9. VOLUME FEATURES
    # --------------------------------------------------

    df["Volume_Change"] = volume.pct_change()

    volume_sma = volume.rolling(20).mean()

    df["Volume_SMA_20"] = volume_sma

    df["Volume_Ratio"] = volume / volume_sma

    # On-Balance Volume
    price_direction = np.sign(close.diff()).fillna(0)

    df["OBV"] = (
        price_direction * volume
    ).cumsum()

    # --------------------------------------------------
    # 10. LAGGED FEATURES
    # --------------------------------------------------

    for lag in [1, 5, 10, 20]:

        df[f"Return_Lag_{lag}"] = (
            df["Daily_Return"].shift(lag)
        )

    # --------------------------------------------------
    # 11. TARGET LABELS
    # --------------------------------------------------

    for horizon in HORIZONS:

        future_return = (
            close.shift(-horizon) / close - 1
        )

        threshold = THRESHOLDS[horizon]

        target = pd.Series(
            np.nan,
            index=df.index,
            dtype="object",
        )

        target.loc[future_return > threshold] = "UP"

        target.loc[future_return < -threshold] = "DOWN"

        target.loc[
            future_return.abs() <= threshold
        ] = "STAY"

        df[f"Target_{horizon}D"] = target

    return df


def process_stock(
    company_name: str,
    ticker_symbol: str,
) -> pd.DataFrame:
    """Process one stock's raw data."""

    input_file = RAW_DATA_DIR / f"{company_name}.csv"

    if not input_file.exists():
        logger.warning("Missing raw file: %s", input_file)
        return pd.DataFrame()

    logger.info("Processing %s", company_name)

    data = pd.read_csv(input_file, parse_dates=["Date"])

    data = data.sort_values("Date")
    data = data.drop_duplicates(subset=["Date"])

    data = data.reset_index(drop=True)

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(col in data.columns for col in required_columns):
        logger.error("Invalid columns in %s", input_file)
        return pd.DataFrame()

    # Remove invalid OHLCV rows
    data = data.dropna(subset=required_columns)

    data = add_technical_indicators(data)

    # Remove rows where indicators or targets are unavailable
    feature_columns = [
        col for col in data.columns
        if col not in [
            "Date",
            "Target_1D",
            "Target_5D",
            "Target_30D",
        ]
    ]

    target_columns = [
        "Target_1D",
        "Target_5D",
        "Target_30D",
    ]

    data = data.dropna(
        subset=feature_columns + target_columns
    )

    data["Ticker"] = ticker_symbol

    output_file = (
        PROCESSED_DATA_DIR /
        f"{company_name}_features.csv"
    )

    data.to_csv(output_file, index=False)

    logger.info(
        "%s: saved %s processed rows",
        company_name,
        len(data),
    )

    return data


def process_all_stocks() -> pd.DataFrame:
    """Process every configured stock."""

    processed_data = []

    for company_name, ticker_symbol in STOCKS.items():

        try:
            data = process_stock(
                company_name,
                ticker_symbol,
            )

            if not data.empty:
                processed_data.append(data)

        except Exception as error:
            logger.exception(
                "Failed to process %s: %s",
                company_name,
                error,
            )

    if not processed_data:
        logger.error("No stocks were processed.")
        return pd.DataFrame()

    combined = pd.concat(
        processed_data,
        ignore_index=True,
    )

    combined_file = (
        PROCESSED_DATA_DIR /
        "all_stocks_features.csv"
    )

    combined.to_csv(combined_file, index=False)

    logger.info(
        "Saved combined dataset with %s rows",
        len(combined),
    )

    return combined


if __name__ == "__main__":

    result = process_all_stocks()

    print("\n========== FEATURE ENGINEERING SUMMARY ==========")

    if not result.empty:
        print(f"Processed rows: {len(result)}")
        print(f"Total columns: {len(result.columns)}")

        print("\nTarget distribution (1-day):")
        print(result["Target_1D"].value_counts())

        print("\nTarget distribution (5-day):")
        print(result["Target_5D"].value_counts())

        print("\nTarget distribution (30-day):")
        print(result["Target_30D"].value_counts())

        print("\nProcessed files saved to:", PROCESSED_DATA_DIR)