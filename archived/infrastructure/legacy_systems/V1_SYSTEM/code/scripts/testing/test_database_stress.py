#!/usr/bin/env python3
"""
Database Stress Testing
Phase 2.5 - Day 4

Tests PostgreSQL performance under high load to verify 1000+ QPS capability.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseStressTester:
    """
    Stress tests PostgreSQL database with various workloads.

    Tests:
    - Concurrent reads/writes
    - Connection pool saturation
    - Bulk operations
    - Cache performance
    - Time-series queries
    """

    def __init__(self):
        from src.bot.database.manager import get_db_manager
        from src.bot.database.models import (
            Alert,
            FeatureValue,
            Model,
            Order,
            Position,
            Prediction,
            SystemMetric,
            Trade,
        )

        self.db = get_db_manager()
        self.models = {
            "Position": Position,
            "Order": Order,
            "Trade": Trade,
            "FeatureValue": FeatureValue,
            "Model": Model,
            "Prediction": Prediction,
            "SystemMetric": SystemMetric,
            "Alert": Alert,
        }

        # Test parameters
        self.num_threads = 20
        self.num_processes = 4
        self.test_duration = 60  # seconds
        self.batch_size = 100

        # Metrics
        self.metrics = {
            "total_reads": 0,
            "total_writes": 0,
            "read_errors": 0,
            "write_errors": 0,
            "read_latencies": [],
            "write_latencies": [],
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def generate_test_data(self, count: int) -> list[dict]:
        """Generate random test data"""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "AMD"]
        data = []

        for _ in range(count):
            data.append(
                {
                    "symbol": random.choice(symbols),
                    "quantity": Decimal(str(random.randint(1, 1000))),
                    "entry_price": Decimal(str(round(random.uniform(10, 1000), 2))),
                    "current_price": Decimal(str(round(random.uniform(10, 1000), 2))),
                    "status": random.choice(["open", "closed", "pending"]),
                    "position_id": str(uuid.uuid4()),
                }
            )

        return data

    def test_concurrent_writes(self, duration: int = 10):
        """Test concurrent write performance"""
        logger.info(f"\nTesting concurrent writes for {duration} seconds...")

        start_time = time.time()
        end_time = start_time + duration
        write_count = 0
        errors = 0
        latencies = []

        def write_position():
            try:
                write_start = time.time()
                data = self.generate_test_data(1)[0]

                position = self.db.create(self.models["Position"], **data)

                latency = (time.time() - write_start) * 1000  # ms
                return True, latency
            except Exception as e:
                logger.debug(f"Write error: {e}")
                return False, 0

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []

            while time.time() < end_time:
                futures.append(executor.submit(write_position))

                # Process completed futures
                done_futures = [f for f in futures if f.done()]
                for future in done_futures:
                    success, latency = future.result()
                    if success:
                        write_count += 1
                        latencies.append(latency)
                    else:
                        errors += 1
                    futures.remove(future)

        # Wait for remaining futures
        for future in as_completed(futures):
            success, latency = future.result()
            if success:
                write_count += 1
                latencies.append(latency)
            else:
                errors += 1

        elapsed = time.time() - start_time
        wps = write_count / elapsed

        logger.info(f"  Writes: {write_count}")
        logger.info(f"  Errors: {errors}")
        logger.info(f"  WPS (Writes Per Second): {wps:.1f}")
        if latencies:
            logger.info(f"  Avg Latency: {np.mean(latencies):.2f}ms")
            logger.info(f"  P95 Latency: {np.percentile(latencies, 95):.2f}ms")
            logger.info(f"  P99 Latency: {np.percentile(latencies, 99):.2f}ms")

        return {"writes": write_count, "errors": errors, "wps": wps, "latencies": latencies}

    def test_concurrent_reads(self, duration: int = 10):
        """Test concurrent read performance"""
        logger.info(f"\nTesting concurrent reads for {duration} seconds...")

        # Create test data first
        logger.info("  Creating test data...")
        test_data = self.generate_test_data(100)
        for data in test_data:
            try:
                self.db.create(self.models["Position"], **data)
            except:
                pass

        start_time = time.time()
        end_time = start_time + duration
        read_count = 0
        errors = 0
        latencies = []
        cache_hits = 0

        def read_positions():
            try:
                read_start = time.time()

                # Random query type
                query_type = random.choice(["all", "symbol", "status"])

                if query_type == "all":
                    positions = self.db.get_many(self.models["Position"], limit=10)
                elif query_type == "symbol":
                    symbol = random.choice(["AAPL", "GOOGL", "MSFT"])
                    positions = self.db.get_many(self.models["Position"], symbol=symbol)
                else:
                    positions = self.db.get_many(self.models["Position"], status="open")

                latency = (time.time() - read_start) * 1000  # ms

                # Check if cached (simplified check)
                is_cached = latency < 1.0  # Assume < 1ms means cache hit

                return True, latency, is_cached
            except Exception as e:
                logger.debug(f"Read error: {e}")
                return False, 0, False

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []

            while time.time() < end_time:
                futures.append(executor.submit(read_positions))

                # Process completed futures
                done_futures = [f for f in futures if f.done()]
                for future in done_futures:
                    success, latency, cached = future.result()
                    if success:
                        read_count += 1
                        latencies.append(latency)
                        if cached:
                            cache_hits += 1
                    else:
                        errors += 1
                    futures.remove(future)

        # Wait for remaining futures
        for future in as_completed(futures):
            success, latency, cached = future.result()
            if success:
                read_count += 1
                latencies.append(latency)
                if cached:
                    cache_hits += 1
            else:
                errors += 1

        elapsed = time.time() - start_time
        qps = read_count / elapsed
        cache_hit_rate = cache_hits / max(1, read_count)

        logger.info(f"  Reads: {read_count}")
        logger.info(f"  Errors: {errors}")
        logger.info(f"  QPS (Queries Per Second): {qps:.1f}")
        logger.info(f"  Cache Hit Rate: {cache_hit_rate:.2%}")
        if latencies:
            logger.info(f"  Avg Latency: {np.mean(latencies):.2f}ms")
            logger.info(f"  P95 Latency: {np.percentile(latencies, 95):.2f}ms")
            logger.info(f"  P99 Latency: {np.percentile(latencies, 99):.2f}ms")

        return {
            "reads": read_count,
            "errors": errors,
            "qps": qps,
            "cache_hit_rate": cache_hit_rate,
            "latencies": latencies,
        }

    def test_bulk_operations(self):
        """Test bulk insert performance"""
        logger.info("\nTesting bulk operations...")

        # Generate large dataset
        batch_sizes = [100, 500, 1000, 5000]
        results = {}

        for batch_size in batch_sizes:
            logger.info(f"  Testing batch size: {batch_size}")

            # Generate data
            test_data = []
            for _ in range(batch_size):
                test_data.append(
                    {
                        "timestamp": datetime.utcnow(),
                        "metric_name": f"test_metric_{random.randint(1, 100)}",
                        "metric_value": Decimal(str(random.uniform(0, 1000))),
                        "component": "stress_test",
                    }
                )

            # Measure bulk insert
            start_time = time.time()
            try:
                self.db.bulk_insert(self.models["SystemMetric"], test_data)
                elapsed = time.time() - start_time
                records_per_second = batch_size / elapsed

                logger.info(f"    Time: {elapsed:.3f}s")
                logger.info(f"    Speed: {records_per_second:.0f} records/second")

                results[batch_size] = {"time": elapsed, "speed": records_per_second}
            except Exception as e:
                logger.error(f"    Bulk insert failed: {e}")
                results[batch_size] = None

        return results

    def test_connection_pool(self):
        """Test connection pool limits"""
        logger.info("\nTesting connection pool saturation...")

        pool_status_start = self.db.get_pool_status()
        logger.info(f"  Initial pool: {pool_status_start}")

        def long_running_query():
            try:
                # Simulate long-running query
                import time as t

                t.sleep(random.uniform(0.5, 2.0))

                # Get pool status during execution
                status = self.db.get_pool_status()

                # Perform actual query
                self.db.get_many(self.models["Position"], limit=100)

                return True, status
            except Exception as e:
                return False, str(e)

        # Try to saturate the pool
        max_connections = 60  # Pool size + overflow
        with ThreadPoolExecutor(max_workers=max_connections + 10) as executor:
            futures = [executor.submit(long_running_query) for _ in range(max_connections + 10)]

            success_count = 0
            timeout_count = 0
            max_checked_out = 0

            for future in as_completed(futures):
                success, result = future.result()
                if success:
                    success_count += 1
                    if isinstance(result, dict) and "checked_out" in result:
                        max_checked_out = max(max_checked_out, result["checked_out"])
                else:
                    if "timeout" in str(result).lower():
                        timeout_count += 1

        pool_status_end = self.db.get_pool_status()

        logger.info(f"  Successful connections: {success_count}")
        logger.info(f"  Timeouts: {timeout_count}")
        logger.info(f"  Max connections used: {max_checked_out}")
        logger.info(f"  Final pool: {pool_status_end}")

        return {
            "success": success_count,
            "timeouts": timeout_count,
            "max_connections": max_checked_out,
        }

    def test_time_series_queries(self):
        """Test TimescaleDB performance"""
        logger.info("\nTesting time-series queries...")

        # Insert time-series data
        logger.info("  Inserting time-series data...")

        symbols = ["AAPL", "GOOGL", "MSFT"]
        timestamps = [datetime.utcnow() - timedelta(minutes=i) for i in range(1000)]

        for symbol in symbols:
            predictions = []
            for ts in timestamps:
                predictions.append(
                    {
                        "model_id": 1,  # Assume model exists
                        "symbol": symbol,
                        "timestamp": ts,
                        "prediction_type": "price",
                        "prediction_value": Decimal(str(random.uniform(100, 200))),
                        "confidence": Decimal(str(random.uniform(0.5, 1.0))),
                        "prediction_data": {"features": {}},
                    }
                )

            try:
                self.db.bulk_insert(self.models["Prediction"], predictions)
            except Exception as e:
                logger.warning(f"    Insert failed for {symbol}: {e}")

        # Test various time-series queries
        queries = [
            ("1 hour range", timedelta(hours=1)),
            ("1 day range", timedelta(days=1)),
            ("1 week range", timedelta(days=7)),
        ]

        results = {}
        for query_name, time_range in queries:
            logger.info(f"  Testing {query_name}...")

            start_time = time.time()
            end_date = datetime.utcnow()
            start_date = end_date - time_range

            try:
                # Query using time range
                predictions = (
                    self.db.session.query(self.models["Prediction"])
                    .filter(
                        self.models["Prediction"].timestamp >= start_date,
                        self.models["Prediction"].timestamp <= end_date,
                    )
                    .all()
                )

                elapsed = time.time() - start_time
                logger.info(f"    Records: {len(predictions)}")
                logger.info(f"    Time: {elapsed*1000:.2f}ms")

                results[query_name] = {"records": len(predictions), "time_ms": elapsed * 1000}
            except Exception as e:
                logger.error(f"    Query failed: {e}")
                results[query_name] = None

        return results

    def run_stress_test(self):
        """Run complete stress test suite"""
        logger.info("=" * 60)
        logger.info("Database Stress Test Suite")
        logger.info("Phase 2.5 - Day 4")
        logger.info("=" * 60)

        all_results = {}

        # Test 1: Concurrent Writes
        logger.info("\n[Test 1/5] Concurrent Write Performance")
        write_results = self.test_concurrent_writes(duration=30)
        all_results["concurrent_writes"] = write_results

        # Test 2: Concurrent Reads
        logger.info("\n[Test 2/5] Concurrent Read Performance")
        read_results = self.test_concurrent_reads(duration=30)
        all_results["concurrent_reads"] = read_results

        # Test 3: Bulk Operations
        logger.info("\n[Test 3/5] Bulk Operation Performance")
        bulk_results = self.test_bulk_operations()
        all_results["bulk_operations"] = bulk_results

        # Test 4: Connection Pool
        logger.info("\n[Test 4/5] Connection Pool Limits")
        pool_results = self.test_connection_pool()
        all_results["connection_pool"] = pool_results

        # Test 5: Time-series Queries
        logger.info("\n[Test 5/5] Time-series Query Performance")
        ts_results = self.test_time_series_queries()
        all_results["time_series"] = ts_results

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Stress Test Summary")
        logger.info("=" * 60)

        # Check if we meet requirements
        meets_requirements = True
        requirements = []

        # Check QPS requirement (1000+)
        if "concurrent_reads" in all_results and all_results["concurrent_reads"]:
            qps = all_results["concurrent_reads"].get("qps", 0)
            if qps >= 1000:
                requirements.append(f"✅ Read QPS: {qps:.0f} (>= 1000)")
            else:
                requirements.append(f"❌ Read QPS: {qps:.0f} (< 1000)")
                meets_requirements = False

        # Check WPS requirement (100+)
        if "concurrent_writes" in all_results and all_results["concurrent_writes"]:
            wps = all_results["concurrent_writes"].get("wps", 0)
            if wps >= 100:
                requirements.append(f"✅ Write WPS: {wps:.0f} (>= 100)")
            else:
                requirements.append(f"❌ Write WPS: {wps:.0f} (< 100)")
                meets_requirements = False

        # Check latency requirement (P99 < 100ms)
        if "concurrent_reads" in all_results and all_results["concurrent_reads"]:
            latencies = all_results["concurrent_reads"].get("latencies", [])
            if latencies:
                p99 = np.percentile(latencies, 99)
                if p99 < 100:
                    requirements.append(f"✅ P99 Read Latency: {p99:.2f}ms (< 100ms)")
                else:
                    requirements.append(f"❌ P99 Read Latency: {p99:.2f}ms (>= 100ms)")
                    meets_requirements = False

        # Check cache performance
        if "concurrent_reads" in all_results and all_results["concurrent_reads"]:
            cache_rate = all_results["concurrent_reads"].get("cache_hit_rate", 0)
            if cache_rate >= 0.5:
                requirements.append(f"✅ Cache Hit Rate: {cache_rate:.1%} (>= 50%)")
            else:
                requirements.append(f"⚠️ Cache Hit Rate: {cache_rate:.1%} (< 50%)")

        # Check connection pool
        if "connection_pool" in all_results and all_results["connection_pool"]:
            max_conn = all_results["connection_pool"].get("max_connections", 0)
            requirements.append(f"ℹ️ Max Connections Used: {max_conn}/60")

        logger.info("\nPerformance Requirements:")
        for req in requirements:
            logger.info(f"  {req}")

        if meets_requirements:
            logger.info("\n🎉 Database meets all performance requirements!")
        else:
            logger.info("\n⚠️ Some performance requirements not met. Consider optimization.")

        # Save results
        results_file = "stress_test_results.json"
        import json

        # Convert numpy arrays to lists for JSON serialization
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(i) for i in obj]
            else:
                return obj

        with open(results_file, "w") as f:
            json.dump(convert_arrays(all_results), f, indent=2, default=str)

        logger.info(f"\nResults saved to {results_file}")

        return all_results, meets_requirements


def main():
    """Main test function"""
    tester = DatabaseStressTester()

    try:
        results, success = tester.run_stress_test()

        if success:
            logger.info("\n✅ Database stress test PASSED")
            return 0
        else:
            logger.info("\n⚠️ Database stress test completed with warnings")
            return 1

    except Exception as e:
        logger.error(f"\n❌ Database stress test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
