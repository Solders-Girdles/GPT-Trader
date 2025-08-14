#!/usr/bin/env python3
"""
Simple Data Download Demo - DEMO-003
GPT-Trader Emergency Recovery

This script demonstrates that the data download system is working.
Downloads data for AAPL, MSFT, GOOGL using YFinanceSource and saves to data/historical/.

Usage:
    poetry run python demos/download_data.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path so we can import bot modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from bot.dataflow.sources.yfinance_source import YFinanceSource
    from bot.logging import get_logger
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you're running with 'poetry run python demos/download_data.py'")
    sys.exit(1)

# Set up logging
logger = get_logger("demo")

# Configuration
SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
DAYS_BACK = 90  # Download last 90 days
DATA_DIR = project_root / "data" / "historical"


def main():
    """Main demo function"""
    print("=" * 60)
    print("🚀 GPT-Trader Data Download Demo")
    print("=" * 60)

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Data directory: {DATA_DIR}")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_BACK)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print(f"📅 Fetching data from {start_str} to {end_str}")
    print(f"🎯 Symbols: {', '.join(SYMBOLS)}")
    print()

    # Initialize YFinance source
    yf_source = YFinanceSource()

    results = {}

    for symbol in SYMBOLS:
        print(f"📊 Downloading {symbol}...")
        try:
            # Download data
            df = yf_source.get_daily_bars(symbol, start=start_str, end=end_str)

            if df.empty:
                print(f"  ❌ No data returned for {symbol}")
                results[symbol] = {"status": "empty", "rows": 0}
                continue

            # Save to CSV
            output_file = DATA_DIR / f"{symbol}_data.csv"
            df.to_csv(output_file)

            # Get info about the data
            first_date = df.index.min().strftime("%Y-%m-%d")
            last_date = df.index.max().strftime("%Y-%m-%d")
            rows = len(df)
            file_size = output_file.stat().st_size

            print(f"  ✅ Downloaded {rows} rows from {first_date} to {last_date}")
            print(f"  💾 Saved to: {output_file.name} ({file_size:,} bytes)")

            # Show sample data
            print(f"  📈 Latest Close: ${df['Close'].iloc[-1]:.2f}")
            print(f"  📊 Average Volume: {df['Volume'].mean():,.0f}")
            print()

            results[symbol] = {
                "status": "success",
                "rows": rows,
                "file": str(output_file),
                "first_date": first_date,
                "last_date": last_date,
                "latest_close": df["Close"].iloc[-1],
            }

        except Exception as e:
            print(f"  ❌ Failed to download {symbol}: {e}")
            results[symbol] = {"status": "error", "error": str(e)}
            continue

    # Summary
    print("=" * 60)
    print("📋 DOWNLOAD SUMMARY")
    print("=" * 60)

    successful = 0
    total_rows = 0

    for symbol, result in results.items():
        status = result["status"]
        if status == "success":
            successful += 1
            total_rows += result["rows"]
            print(f"✅ {symbol}: {result['rows']} rows, latest ${result['latest_close']:.2f}")
        elif status == "empty":
            print(f"⚠️  {symbol}: No data available")
        else:
            print(f"❌ {symbol}: {result.get('error', 'Unknown error')}")

    print()
    print(f"🎯 Success Rate: {successful}/{len(SYMBOLS)} symbols")
    print(f"📊 Total Rows Downloaded: {total_rows:,}")
    print(f"📁 Files saved to: {DATA_DIR}")

    # List files in data directory
    print("\n📂 Files in data/historical/:")
    for file in sorted(DATA_DIR.glob("*.csv")):
        file_size = file.stat().st_size
        print(f"  - {file.name} ({file_size:,} bytes)")

    if successful > 0:
        print("\n🎉 DATA DOWNLOAD DEMO COMPLETED SUCCESSFULLY!")
        print(f"✅ {successful} symbols downloaded successfully")
        print("✅ Data saved to CSV files")
        print("✅ System is working and can fetch market data")
        return 0
    else:
        print("\n❌ No data was successfully downloaded")
        print("Check your internet connection or try again later")
        return 1


if __name__ == "__main__":
    exit(main())
