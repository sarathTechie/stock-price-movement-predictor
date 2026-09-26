
from src.data_collection import collect_all_stocks


def main():
    print("=" * 55)
    print(" STOCK PRICE MOVEMENT PREDICTOR")
    print(" Indian Stock Market - NSE")
    print("=" * 55)

    print("\nStarting historical data collection...\n")

    stock_data = collect_all_stocks()

    if not stock_data:
        print("\nNo stock data was downloaded.")
        print("Check your internet connection and ticker symbols.")
        return

    print("\nDATA COLLECTION COMPLETED")
    print("-" * 55)

    total_records = 0

    for company, data in stock_data.items():
        records = len(data)
        total_records += records

        print(f"{company:<20} {records:>8} rows")

    print("-" * 55)
    print(f"Stocks downloaded: {len(stock_data)}")
    print(f"Total data records: {total_records}")
    print("\nCheck the data/raw folder for CSV files.")


if __name__ == "__main__":
    main()