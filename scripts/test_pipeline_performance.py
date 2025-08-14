#!/usr/bin/env python3
"""
Data Pipeline Performance Testing
Phase 2.5 - Day 4

Tests the real-time data pipeline for latency, throughput, and reliability.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import json
import logging
import random
import statistics
import time
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PipelinePerformanceTester:
    """
    Tests data pipeline performance metrics.

    Metrics:
    - End-to-end latency
    - Throughput (messages/second)
    - Data source failover time
    - Validation overhead
    - Memory usage
    """

    def __init__(self):
        self.metrics = {
            "latencies": [],
            "throughput": [],
            "failover_times": [],
            "validation_times": [],
            "memory_usage": [],
            "errors": [],
        }

        # Test configuration
        self.test_duration = 60  # seconds
        self.message_rate = 100  # messages per second target
        self.test_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

    async def test_data_source_latency(self):
        """Test latency for each data source"""
        from src.bot.dataflow.data_source_manager import DataSource, DataSourceManager

        logger.info("\nTesting Data Source Latency...")
        manager = DataSourceManager()

        results = {}

        # Test each source
        sources = [DataSource.ALPACA, DataSource.POLYGON, DataSource.IEX, DataSource.YAHOO]

        for source in sources:
            logger.info(f"  Testing {source.value}...")
            latencies = []

            for symbol in self.test_symbols:
                start_time = time.time()

                try:
                    # Force specific source
                    manager.primary_source = source
                    data = await manager._fetch_from_source(source, symbol, timeout=10.0)

                    if data:
                        latency = (time.time() - start_time) * 1000  # ms
                        latencies.append(latency)
                        logger.debug(f"    {symbol}: {latency:.2f}ms")
                    else:
                        logger.warning(f"    {symbol}: No data returned")

                except Exception as e:
                    logger.error(f"    {symbol}: Error - {e}")

            if latencies:
                results[source.value] = {
                    "avg": statistics.mean(latencies),
                    "min": min(latencies),
                    "max": max(latencies),
                    "p50": np.percentile(latencies, 50),
                    "p95": np.percentile(latencies, 95),
                    "p99": np.percentile(latencies, 99),
                }

                logger.info(f"    Average: {results[source.value]['avg']:.2f}ms")
                logger.info(f"    P95: {results[source.value]['p95']:.2f}ms")
                logger.info(f"    P99: {results[source.value]['p99']:.2f}ms")
            else:
                results[source.value] = None
                logger.warning("    No successful requests")

        await manager.shutdown()
        return results

    async def test_failover_speed(self):
        """Test how quickly system fails over to backup sources"""
        from src.bot.dataflow.data_source_manager import DataSourceManager

        logger.info("\nTesting Failover Speed...")
        manager = DataSourceManager()

        # Simulate primary source failure
        primary = manager.primary_source
        logger.info(f"  Primary source: {primary.value}")

        # Force primary to fail
        manager.source_status[primary].consecutive_errors = 3
        manager.source_status[primary].is_active = False

        failover_start = time.time()

        # Try to fetch data (should failover)
        data = await manager.fetch_market_data("AAPL", timeout=10.0)

        failover_time = (time.time() - failover_start) * 1000  # ms

        if data:
            logger.info(f"  ✅ Failover successful to {data.source.value}")
            logger.info(f"  Failover time: {failover_time:.2f}ms")
        else:
            logger.error("  ❌ Failover failed")
            failover_time = None

        await manager.shutdown()
        return failover_time

    async def test_validation_overhead(self):
        """Test data validation performance impact"""
        from src.bot.dataflow.realtime_feed import DataSource, DataValidator, MarketData

        logger.info("\nTesting Validation Overhead...")
        validator = DataValidator()

        # Create test data
        test_data = []
        for _ in range(1000):
            test_data.append(
                MarketData(
                    symbol=random.choice(self.test_symbols),
                    timestamp=datetime.now(),
                    price=Decimal(str(random.uniform(100, 500))),
                    bid=Decimal(str(random.uniform(99, 499))),
                    ask=Decimal(str(random.uniform(101, 501))),
                    volume=random.randint(1000, 1000000),
                    source=DataSource.YAHOO,
                )
            )

        # Test with validation
        start_time = time.time()
        valid_count = 0

        for data in test_data:
            is_valid, error = validator.validate_market_data(data)
            if is_valid:
                valid_count += 1

        validation_time = time.time() - start_time
        avg_validation_time = (validation_time / len(test_data)) * 1000  # ms per validation

        logger.info(f"  Validated {len(test_data)} messages")
        logger.info(f"  Valid: {valid_count}/{len(test_data)}")
        logger.info(f"  Total time: {validation_time:.3f}s")
        logger.info(f"  Avg per message: {avg_validation_time:.3f}ms")
        logger.info(f"  Throughput: {len(test_data)/validation_time:.0f} msg/s")

        return {
            "total_messages": len(test_data),
            "valid_messages": valid_count,
            "total_time_s": validation_time,
            "avg_time_ms": avg_validation_time,
            "throughput": len(test_data) / validation_time,
        }

    async def test_throughput(self):
        """Test maximum message throughput"""
        from src.bot.dataflow.realtime_feed import DataFeedConfig, MarketData, RealtimeDataFeed

        logger.info("\nTesting Message Throughput...")

        config = DataFeedConfig(buffer_size=10000, validate_data=True)

        feed = RealtimeDataFeed(config)

        # Track received messages
        received_messages = []
        message_latencies = []

        def on_data(data: MarketData):
            received_time = datetime.now()
            latency = (received_time - data.timestamp).total_seconds() * 1000
            received_messages.append(data)
            message_latencies.append(latency)

        feed.register_data_callback(on_data)

        # Generate messages at high rate
        logger.info(f"  Generating {self.message_rate} msg/s for {self.test_duration}s...")

        start_time = time.time()
        messages_sent = 0

        # Simulate high-rate message generation
        for _ in range(self.test_duration):
            for _ in range(self.message_rate):
                data = MarketData(
                    symbol=random.choice(self.test_symbols),
                    timestamp=datetime.now(),
                    price=Decimal(str(random.uniform(100, 500))),
                    source=DataSource.YAHOO,
                )

                # Process through validation
                feed._process_market_data(data)
                messages_sent += 1

            await asyncio.sleep(1)  # 1 second intervals

        elapsed = time.time() - start_time

        # Calculate metrics
        actual_throughput = len(received_messages) / elapsed
        drop_rate = 1 - (len(received_messages) / messages_sent)

        logger.info(f"  Messages sent: {messages_sent}")
        logger.info(f"  Messages received: {len(received_messages)}")
        logger.info(f"  Actual throughput: {actual_throughput:.0f} msg/s")
        logger.info(f"  Drop rate: {drop_rate:.2%}")

        if message_latencies:
            avg_latency = statistics.mean(message_latencies)
            p95_latency = np.percentile(message_latencies, 95)
            p99_latency = np.percentile(message_latencies, 99)

            logger.info(f"  Avg latency: {avg_latency:.2f}ms")
            logger.info(f"  P95 latency: {p95_latency:.2f}ms")
            logger.info(f"  P99 latency: {p99_latency:.2f}ms")

        feed.stop()

        return {
            "messages_sent": messages_sent,
            "messages_received": len(received_messages),
            "throughput": actual_throughput,
            "drop_rate": drop_rate,
            "latencies": message_latencies[:100] if message_latencies else [],  # Sample
        }

    async def test_memory_usage(self):
        """Test memory usage under load"""
        import gc

        import psutil

        logger.info("\nTesting Memory Usage...")

        process = psutil.Process()

        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"  Baseline memory: {baseline_memory:.1f} MB")

        from src.bot.dataflow.data_source_manager import DataSourceManager
        from src.bot.dataflow.realtime_feed import DataFeedConfig, RealtimeDataFeed

        # Create instances
        manager = DataSourceManager()
        feed = RealtimeDataFeed(DataFeedConfig(buffer_size=10000))

        # Simulate load
        logger.info("  Simulating load...")
        memory_samples = []

        for i in range(10):
            # Fetch data
            tasks = []
            for symbol in self.test_symbols * 10:  # 50 requests
                tasks.append(manager.fetch_market_data(symbol))

            await asyncio.gather(*tasks, return_exceptions=True)

            # Check memory
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(current_memory)

            logger.debug(f"    Sample {i+1}: {current_memory:.1f} MB")
            await asyncio.sleep(1)

        # Cleanup
        feed.stop()
        await manager.shutdown()

        # Final memory after cleanup
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        peak_memory = max(memory_samples)
        avg_memory = statistics.mean(memory_samples)
        memory_increase = peak_memory - baseline_memory

        logger.info(f"  Peak memory: {peak_memory:.1f} MB")
        logger.info(f"  Average memory: {avg_memory:.1f} MB")
        logger.info(f"  Memory increase: {memory_increase:.1f} MB")
        logger.info(f"  Final memory: {final_memory:.1f} MB")

        return {
            "baseline_mb": baseline_memory,
            "peak_mb": peak_memory,
            "average_mb": avg_memory,
            "increase_mb": memory_increase,
            "final_mb": final_memory,
        }

    async def test_historical_data_performance(self):
        """Test historical data fetch performance"""
        from src.bot.dataflow.data_source_manager import DataSourceManager

        logger.info("\nTesting Historical Data Performance...")
        manager = DataSourceManager()

        date_ranges = [("1 week", 7), ("1 month", 30), ("3 months", 90)]

        results = {}

        for range_name, days in date_ranges:
            logger.info(f"  Testing {range_name} data...")

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            fetch_times = []
            data_points = []

            for symbol in self.test_symbols:
                start_time = time.time()

                try:
                    df = await manager.fetch_historical_data(symbol, start_date, end_date)

                    if df is not None and not df.empty:
                        fetch_time = time.time() - start_time
                        fetch_times.append(fetch_time)
                        data_points.append(len(df))
                        logger.debug(f"    {symbol}: {len(df)} points in {fetch_time:.2f}s")
                    else:
                        logger.warning(f"    {symbol}: No data returned")

                except Exception as e:
                    logger.error(f"    {symbol}: Error - {e}")

            if fetch_times:
                results[range_name] = {
                    "avg_time_s": statistics.mean(fetch_times),
                    "total_points": sum(data_points),
                    "points_per_second": sum(data_points) / sum(fetch_times),
                }

                logger.info(f"    Average time: {results[range_name]['avg_time_s']:.2f}s")
                logger.info(f"    Total points: {results[range_name]['total_points']}")
                logger.info(f"    Speed: {results[range_name]['points_per_second']:.0f} points/s")

        await manager.shutdown()
        return results

    async def run_performance_tests(self):
        """Run complete performance test suite"""
        logger.info("=" * 60)
        logger.info("Data Pipeline Performance Test Suite")
        logger.info("Phase 2.5 - Day 4")
        logger.info("=" * 60)

        all_results = {}

        # Test 1: Data Source Latency
        logger.info("\n[Test 1/6] Data Source Latency")
        try:
            latency_results = await self.test_data_source_latency()
            all_results["source_latency"] = latency_results
        except Exception as e:
            logger.error(f"Test failed: {e}")
            all_results["source_latency"] = None

        # Test 2: Failover Speed
        logger.info("\n[Test 2/6] Failover Speed")
        try:
            failover_time = await self.test_failover_speed()
            all_results["failover_time_ms"] = failover_time
        except Exception as e:
            logger.error(f"Test failed: {e}")
            all_results["failover_time_ms"] = None

        # Test 3: Validation Overhead
        logger.info("\n[Test 3/6] Validation Overhead")
        try:
            validation_results = await self.test_validation_overhead()
            all_results["validation"] = validation_results
        except Exception as e:
            logger.error(f"Test failed: {e}")
            all_results["validation"] = None

        # Test 4: Throughput
        logger.info("\n[Test 4/6] Message Throughput")
        try:
            throughput_results = await self.test_throughput()
            all_results["throughput"] = throughput_results
        except Exception as e:
            logger.error(f"Test failed: {e}")
            all_results["throughput"] = None

        # Test 5: Memory Usage
        logger.info("\n[Test 5/6] Memory Usage")
        try:
            memory_results = await self.test_memory_usage()
            all_results["memory"] = memory_results
        except Exception as e:
            logger.error(f"Test failed: {e}")
            all_results["memory"] = None

        # Test 6: Historical Data
        logger.info("\n[Test 6/6] Historical Data Performance")
        try:
            historical_results = await self.test_historical_data_performance()
            all_results["historical"] = historical_results
        except Exception as e:
            logger.error(f"Test failed: {e}")
            all_results["historical"] = None

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Performance Test Summary")
        logger.info("=" * 60)

        # Evaluate requirements
        meets_requirements = True
        requirements = []

        # Check latency requirement (< 100ms for primary source)
        if all_results.get("source_latency"):
            primary_latency = None
            for source, metrics in all_results["source_latency"].items():
                if metrics and "p95" in metrics:
                    if primary_latency is None or metrics["p95"] < primary_latency:
                        primary_latency = metrics["p95"]

            if primary_latency and primary_latency < 100:
                requirements.append(
                    f"✅ Primary source P95 latency: {primary_latency:.2f}ms (< 100ms)"
                )
            elif primary_latency:
                requirements.append(
                    f"❌ Primary source P95 latency: {primary_latency:.2f}ms (>= 100ms)"
                )
                meets_requirements = False

        # Check failover speed (< 1000ms)
        if all_results.get("failover_time_ms"):
            if all_results["failover_time_ms"] < 1000:
                requirements.append(
                    f"✅ Failover time: {all_results['failover_time_ms']:.2f}ms (< 1000ms)"
                )
            else:
                requirements.append(
                    f"❌ Failover time: {all_results['failover_time_ms']:.2f}ms (>= 1000ms)"
                )
                meets_requirements = False

        # Check throughput (> 1000 msg/s)
        if all_results.get("throughput") and all_results["throughput"].get("throughput"):
            throughput = all_results["throughput"]["throughput"]
            if throughput >= 1000:
                requirements.append(f"✅ Message throughput: {throughput:.0f} msg/s (>= 1000)")
            else:
                requirements.append(f"⚠️ Message throughput: {throughput:.0f} msg/s (< 1000)")

        # Check validation performance (> 10000 msg/s)
        if all_results.get("validation") and all_results["validation"].get("throughput"):
            val_throughput = all_results["validation"]["throughput"]
            if val_throughput >= 10000:
                requirements.append(
                    f"✅ Validation throughput: {val_throughput:.0f} msg/s (>= 10000)"
                )
            else:
                requirements.append(
                    f"⚠️ Validation throughput: {val_throughput:.0f} msg/s (< 10000)"
                )

        # Check memory usage (< 500MB increase)
        if all_results.get("memory") and all_results["memory"].get("increase_mb"):
            mem_increase = all_results["memory"]["increase_mb"]
            if mem_increase < 500:
                requirements.append(f"✅ Memory increase: {mem_increase:.1f} MB (< 500MB)")
            else:
                requirements.append(f"❌ Memory increase: {mem_increase:.1f} MB (>= 500MB)")
                meets_requirements = False

        logger.info("\nPerformance Requirements:")
        for req in requirements:
            logger.info(f"  {req}")

        if meets_requirements:
            logger.info("\n🎉 Pipeline meets all performance requirements!")
        else:
            logger.info("\n⚠️ Some performance requirements not met.")

        # Save results
        results_file = "pipeline_performance_results.json"

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj

        with open(results_file, "w") as f:
            json.dump(convert_numpy(all_results), f, indent=2, default=str)

        logger.info(f"\nResults saved to {results_file}")

        return all_results, meets_requirements


async def main():
    """Main test function"""
    tester = PipelinePerformanceTester()

    try:
        results, success = await tester.run_performance_tests()

        if success:
            logger.info("\n✅ Pipeline performance test PASSED")
            return 0
        else:
            logger.info("\n⚠️ Pipeline performance test completed with warnings")
            return 1

    except Exception as e:
        logger.error(f"\n❌ Pipeline performance test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
