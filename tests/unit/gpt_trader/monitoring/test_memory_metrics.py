"""Unit tests for memory metrics and event store cache tracking."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestMemoryGaugeEmission:
    """Tests for memory gauge emission."""

    def test_event_store_cache_gauges_emitted(self) -> None:
        """Test that event store cache gauges are emitted via _collect_event_store_metrics."""
        from gpt_trader.features.live_trade.engines.system_maintenance import (
            SystemMaintenanceService,
        )

        mock_status_reporter = MagicMock()

        # Create mock event store with cache methods
        mock_event_store = MagicMock()
        mock_event_store.get_cache_size.return_value = 500
        mock_event_store.get_cache_fill_ratio.return_value = 0.05

        service = SystemMaintenanceService(
            status_reporter=mock_status_reporter,
            event_store=mock_event_store,
        )

        with patch(
            "gpt_trader.features.live_trade.engines.system_maintenance.record_gauge"
        ) as mock_gauge:
            # Call the method directly to test it
            service._collect_event_store_metrics()

            # Check event store gauges were recorded
            gauge_calls = mock_gauge.call_args_list
            gauge_names = [c[0][0] for c in gauge_calls]

            assert "gpt_trader_event_store_cache_size" in gauge_names
            assert "gpt_trader_deque_cache_fill_ratio" in gauge_names

            # Check values
            cache_size_call = next(
                c for c in gauge_calls if c[0][0] == "gpt_trader_event_store_cache_size"
            )
            assert cache_size_call[0][1] == 500.0

            fill_ratio_call = next(
                c for c in gauge_calls if c[0][0] == "gpt_trader_deque_cache_fill_ratio"
            )
            assert fill_ratio_call[0][1] == 0.05

    def test_event_store_metrics_handles_none_store(self) -> None:
        """Test that _collect_event_store_metrics handles None event store."""
        from gpt_trader.features.live_trade.engines.system_maintenance import (
            SystemMaintenanceService,
        )

        mock_status_reporter = MagicMock()
        service = SystemMaintenanceService(
            status_reporter=mock_status_reporter,
            event_store=None,
        )

        with patch(
            "gpt_trader.features.live_trade.engines.system_maintenance.record_gauge"
        ) as mock_gauge:
            # Should not raise
            service._collect_event_store_metrics()

            # Should not record any gauges
            assert not mock_gauge.called

    def test_event_store_metrics_handles_missing_methods(self) -> None:
        """Test graceful handling when event store lacks cache methods."""
        from gpt_trader.features.live_trade.engines.system_maintenance import (
            SystemMaintenanceService,
        )

        mock_status_reporter = MagicMock()

        # Create a mock event store WITHOUT cache methods
        mock_event_store = MagicMock(spec=[])  # Empty spec = no methods

        service = SystemMaintenanceService(
            status_reporter=mock_status_reporter,
            event_store=mock_event_store,
        )

        with patch(
            "gpt_trader.features.live_trade.engines.system_maintenance.record_gauge"
        ) as mock_gauge:
            # Should not raise
            service._collect_event_store_metrics()

            # Should not record any gauges (methods don't exist)
            assert not mock_gauge.called


class TestEventStoreCacheMetrics:
    """Tests for EventStore cache metric methods."""

    def test_get_cache_size(self) -> None:
        """Test get_cache_size returns correct count."""
        from gpt_trader.persistence.event_store import EventStore

        store = EventStore(root=None, max_cache_size=100)

        assert store.get_cache_size() == 0

        store.append("test", {"key": "value"})
        assert store.get_cache_size() == 1

        for i in range(5):
            store.append("test", {"i": i})
        assert store.get_cache_size() == 6

    def test_get_cache_max_size(self) -> None:
        """Test get_cache_max_size returns configured size."""
        from gpt_trader.persistence.event_store import EventStore

        store = EventStore(root=None, max_cache_size=500)
        assert store.get_cache_max_size() == 500

    def test_get_cache_fill_ratio(self) -> None:
        """Test get_cache_fill_ratio calculation."""
        from gpt_trader.persistence.event_store import EventStore

        store = EventStore(root=None, max_cache_size=100)

        assert store.get_cache_fill_ratio() == 0.0

        for i in range(50):
            store.append("test", {"i": i})
        assert store.get_cache_fill_ratio() == 0.5

        for i in range(50):
            store.append("test", {"i": i + 50})
        assert store.get_cache_fill_ratio() == 1.0

    def test_get_cache_fill_ratio_handles_zero_max(self) -> None:
        """Test fill ratio handles zero max_cache_size gracefully."""
        from gpt_trader.persistence.event_store import EventStore

        store = EventStore(root=None, max_cache_size=0)
        assert store.get_cache_fill_ratio() == 0.0
