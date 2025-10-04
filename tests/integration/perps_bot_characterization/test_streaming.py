"""
Characterization Tests for PerpsBot Streaming Lock Sharing

Tests documenting shared lock behavior between update_marks and streaming.
"""

import pytest
import threading
from decimal import Decimal

from bot_v2.orchestration.perps_bot import PerpsBot


@pytest.mark.integration
@pytest.mark.characterization
class TestPerpsBotStreamingLockSharing:
    """Characterize shared lock between update_marks and streaming"""

    def test_mark_lock_is_reentrant_lock(self, monkeypatch, tmp_path, minimal_config):
        """Document: _mark_lock must be RLock for reentrant access"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # RLock is _thread.RLock type, check by name
        assert type(bot._mark_lock).__name__ == "RLock"

    @pytest.mark.asyncio
    async def test_update_mark_window_is_thread_safe(self, monkeypatch, tmp_path, minimal_config):
        """Document: _update_mark_window must use _mark_lock"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        # Instrument threading.RLock to track acquire calls
        import threading

        acquire_count = {"count": 0}
        original_rlock = threading.RLock

        class InstrumentedRLock:
            """Wrapper that tracks acquire calls to verify lock usage"""

            def __init__(self):
                self._lock = original_rlock()

            def acquire(self, blocking=True, timeout=-1):
                acquire_count["count"] += 1
                if timeout == -1:
                    return self._lock.acquire(blocking)
                return self._lock.acquire(blocking, timeout)

            def release(self):
                return self._lock.release()

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *args):
                self.release()

        # Patch threading.RLock before bot construction
        monkeypatch.setattr(threading, "RLock", InstrumentedRLock)

        bot = PerpsBot(minimal_config)

        # Reset count after initialization (bot.__init__ may acquire lock)
        acquire_count["count"] = 0

        # Run concurrent updates
        def concurrent_update():
            bot._update_mark_window("BTC-USD", Decimal("50000"))

        thread = threading.Thread(target=concurrent_update)
        thread.start()
        bot._update_mark_window("BTC-USD", Decimal("50100"))
        thread.join(timeout=1.0)

        # CRITICAL: Verify lock was actually acquired (fails if lock removed)
        # Without this assertion, GIL makes list appends safe, hiding lock removal
        assert (
            acquire_count["count"] >= 2
        ), f"Lock acquired {acquire_count['count']} times, expected >= 2"

        # Verify no corruption
        assert len(bot.mark_windows["BTC-USD"]) == 2
        assert all(isinstance(m, Decimal) for m in bot.mark_windows["BTC-USD"])
        assert Decimal("50000") in bot.mark_windows["BTC-USD"]
        assert Decimal("50100") in bot.mark_windows["BTC-USD"]

    def test_streaming_service_shares_mark_lock(self, monkeypatch, tmp_path, minimal_config):
        """Document: StreamingService must use same _mark_lock as PerpsBot"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))

        bot = PerpsBot(minimal_config)

        # Verify lock sharing via MarketDataService (service always created)
        assert bot._streaming_service is not None
        assert bot._streaming_service.market_data_service._mark_lock is bot._mark_lock

    def test_concurrent_update_mark_window_no_race(self, monkeypatch, tmp_path, minimal_config):
        """Document: Concurrent update_mark_window calls must not corrupt mark_windows"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        # Set large MA values to prevent trimming during test
        minimal_config.short_ma = 100
        minimal_config.long_ma = 200
        bot = PerpsBot(minimal_config)

        # Stress test with many concurrent updates
        num_threads = 20
        updates_per_thread = 10

        def worker(thread_id):
            for i in range(updates_per_thread):
                mark = Decimal(f"{50000 + thread_id * 100 + i}")
                bot._update_mark_window("BTC-USD", mark)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        # Verify no corruption
        marks = bot.mark_windows["BTC-USD"]
        assert (
            len(marks) == num_threads * updates_per_thread
        ), "Lost updates indicate race condition"
        assert all(
            isinstance(m, Decimal) for m in marks
        ), "Type corruption indicates race condition"

        # Verify all values are unique (no duplicates from races)
        assert len(set(marks)) == len(marks), "Duplicate values indicate race condition"

    def test_mark_trimming_is_atomic(self, monkeypatch, tmp_path, minimal_config):
        """Document: Mark window trimming must be atomic under concurrent access"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        # Use small MA values to trigger frequent trimming
        minimal_config.short_ma = 5
        minimal_config.long_ma = 10
        bot = PerpsBot(minimal_config)

        # max_size will be max(5, 10) + 5 = 15
        expected_max_size = 15

        # Concurrent updates that will trigger trimming
        num_threads = 10
        updates_per_thread = 20  # Total 200 updates, will trigger trimming

        def worker(thread_id):
            for i in range(updates_per_thread):
                mark = Decimal(f"{thread_id * 1000 + i}")
                bot._update_mark_window("BTC-USD", mark)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        # Verify trimming worked correctly
        marks = bot.mark_windows["BTC-USD"]

        # CRITICAL: Window must not exceed max_size (trimming is working)
        assert (
            len(marks) <= expected_max_size
        ), f"Window size {len(marks)} exceeds max {expected_max_size} - trimming failed"

        # Verify window contains exactly max_size elements (trimming happened)
        assert (
            len(marks) == expected_max_size
        ), f"Expected trimmed window of {expected_max_size}, got {len(marks)}"

        # Verify no corruption (all Decimals)
        assert all(isinstance(m, Decimal) for m in marks), "Type corruption in trimmed window"
