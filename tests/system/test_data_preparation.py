#!/usr/bin/env python3
"""
Test script for Data Preparation Pipeline

Tests the integration of Historical Data Manager and Data Quality Framework
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
import sys

sys.path.insert(0, "src")

from bot.dataflow.historical_data_manager import create_historical_data_manager, DataFrequency
from bot.dataflow.data_quality_framework import create_data_quality_framework

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_data_preparation():
    """Test the complete data preparation pipeline"""

    print("🚀 Testing Data Preparation Pipeline")
    print("=" * 50)

    # Test configuration
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "META"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    output_dir = Path("data/test_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Test Symbols: {test_symbols}")
    print(f"Date Range: {start_date.date()} to {end_date.date()}")
    print(f"Output Directory: {output_dir}")

    try:
        # Step 1: Initialize Historical Data Manager
        print("\n📡 Step 1: Initializing Historical Data Manager...")
        data_manager = create_historical_data_manager(
            min_quality_score=0.70,  # Lower threshold for testing
            cache_dir=str(output_dir / "cache"),
            max_concurrent_downloads=3,
        )
        print("✅ Historical Data Manager initialized")

        # Step 2: Initialize Data Quality Framework
        print("\n🧹 Step 2: Initializing Data Quality Framework...")
        quality_framework = create_data_quality_framework(
            min_quality_score=70.0, outlier_method="iqr", missing_data_method="forward"
        )
        print("✅ Data Quality Framework initialized")

        # Step 3: Download Historical Data
        print("\n📊 Step 3: Downloading Historical Data...")
        datasets, metadata = data_manager.get_training_dataset(
            symbols=test_symbols,
            start_date=start_date,
            end_date=end_date,
            frequency=DataFrequency.DAILY,
            force_refresh=False,
        )

        print(f"✅ Downloaded data for {len(datasets)}/{len(test_symbols)} symbols")
        for symbol, data in datasets.items():
            print(f"   • {symbol}: {len(data)} records")

        # Step 4: Quality Assessment and Cleaning
        print("\n🔍 Step 4: Quality Assessment and Cleaning...")
        cleaned_datasets = {}
        quality_reports = {}

        for symbol, raw_data in datasets.items():
            print(f"\n   Processing {symbol}...")

            # Assess quality
            quality_report = quality_framework.assess_quality(raw_data, symbol)
            print(f"   • Initial Quality Score: {quality_report.quality_score:.1f}/100")

            if quality_report.issues:
                print(f"   • Issues Found: {len(quality_report.issues)}")
                for issue in quality_report.issues[:3]:  # Show first 3 issues
                    print(f"     - {issue.severity.value.upper()}: {issue.description}")

            # Clean data
            cleaned_data, final_report = quality_framework.clean_and_validate(raw_data, symbol)

            cleaned_datasets[symbol] = cleaned_data
            quality_reports[symbol] = final_report

            print(f"   • Final Quality Score: {final_report.quality_score:.1f}/100")
            if final_report.cleaning_applied:
                print(f"   • Cleaning Actions: {len(final_report.cleaning_applied)}")
                for action in final_report.cleaning_applied[:2]:  # Show first 2 actions
                    print(f"     - {action}")

            print(f"   • Usable: {'✅ Yes' if final_report.is_usable else '❌ No'}")

        # Step 5: Generate Summary
        print("\n📈 Step 5: Generating Summary...")
        usable_datasets = [symbol for symbol, report in quality_reports.items() if report.is_usable]
        quality_scores = [report.quality_score for report in quality_reports.values()]

        print(f"\n🎯 RESULTS SUMMARY")
        print(f"   Symbols Requested: {len(test_symbols)}")
        print(f"   Data Downloaded: {len(datasets)}")
        print(f"   Usable Datasets: {len(usable_datasets)}")
        print(f"   Success Rate: {(len(usable_datasets)/len(test_symbols)*100):.1f}%")
        print(f"   Average Quality Score: {sum(quality_scores)/len(quality_scores):.1f}/100")

        # Step 6: Save Test Results
        print(f"\n💾 Step 6: Saving Test Results...")
        results_dir = output_dir / "test_results"
        results_dir.mkdir(exist_ok=True)

        for symbol, data in cleaned_datasets.items():
            filename = f"{symbol}_test_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = results_dir / filename
            data.to_csv(filepath)
            print(f"   • Saved {symbol}: {len(data)} records to {filename}")

        # Cache info
        cache_info = data_manager.get_cache_info()
        print(f"\n💾 Cache Information:")
        print(f"   • Cached Datasets: {cache_info['cached_datasets']}")
        print(f"   • Cache Size: {cache_info['total_cache_size_mb']:.1f} MB")

        print(f"\n✅ TEST COMPLETED SUCCESSFULLY!")
        print(f"   Ready to prepare datasets for {len(usable_datasets)} symbols")
        print(f"   Next step: Use prepare-datasets CLI for full universe")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        logger.exception("Full error details:")
        return False


if __name__ == "__main__":
    success = test_data_preparation()
    sys.exit(0 if success else 1)
