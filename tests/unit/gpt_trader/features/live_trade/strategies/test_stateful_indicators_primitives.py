"""
Tests for stateful indicator primitives (WelfordState, RollingWindow).
"""

from decimal import Decimal

from gpt_trader.features.live_trade.stateful_indicators import RollingWindow, WelfordState


class TestWelfordState:
    """Test Welford's online algorithm for mean/variance."""

    def test_empty_state(self) -> None:
        """Test initial state values."""
        state = WelfordState()
        assert state.count == 0
        assert state.mean == Decimal("0")
        assert state.variance == Decimal("0")

    def test_single_value(self) -> None:
        """Test state after single update."""
        state = WelfordState()
        state.update(Decimal("10"))
        assert state.count == 1
        assert state.mean == Decimal("10")
        assert state.variance == Decimal("0")  # Need at least 2 for variance

    def test_multiple_values_mean(self) -> None:
        """Test mean calculation with multiple values."""
        state = WelfordState()
        values = [Decimal("10"), Decimal("20"), Decimal("30")]
        for v in values:
            state.update(v)

        assert state.count == 3
        assert state.mean == Decimal("20")  # (10 + 20 + 30) / 3

    def test_variance_calculation(self) -> None:
        """Test variance matches expected value."""
        state = WelfordState()
        # Values: 2, 4, 4, 4, 5, 5, 7, 9 (classic example)
        values = [Decimal(v) for v in [2, 4, 4, 4, 5, 5, 7, 9]]
        for v in values:
            state.update(v)

        # Mean = 5, Variance = 4 (population)
        assert state.mean == Decimal("5")
        # Allow for small floating point differences
        assert abs(state.variance - Decimal("4")) < Decimal("0.0001")

    def test_serialize_deserialize_roundtrip(self) -> None:
        """Test serialization preserves state."""
        state = WelfordState()
        for v in [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]:
            state.update(v)

        serialized = state.serialize()
        restored = WelfordState.deserialize(serialized)

        assert restored.count == state.count
        assert restored.mean == state.mean
        assert restored.m2 == state.m2


class TestRollingWindow:
    """Test circular buffer rolling window."""

    def test_add_below_capacity(self) -> None:
        """Test adding values before window is full."""
        window = RollingWindow(max_size=3)
        assert window.add(Decimal("1")) is None
        assert window.add(Decimal("2")) is None
        assert len(window) == 2
        assert not window.is_full

    def test_add_at_capacity_evicts(self) -> None:
        """Test that adding at capacity evicts oldest value."""
        window = RollingWindow(max_size=3)
        window.add(Decimal("1"))
        window.add(Decimal("2"))
        window.add(Decimal("3"))
        assert window.is_full

        evicted = window.add(Decimal("4"))
        assert evicted == Decimal("1")
        assert list(window.values) == [Decimal("2"), Decimal("3"), Decimal("4")]

    def test_mean_calculation(self) -> None:
        """Test O(1) mean calculation."""
        window = RollingWindow(max_size=3)
        window.add(Decimal("10"))
        window.add(Decimal("20"))
        window.add(Decimal("30"))

        assert window.mean == Decimal("20")

    def test_running_sum_accuracy(self) -> None:
        """Test running sum stays accurate through evictions."""
        window = RollingWindow(max_size=3)
        for i in range(1, 11):
            window.add(Decimal(str(i)))

        # Window should contain [8, 9, 10]
        assert window.running_sum == Decimal("27")
        assert window.mean == Decimal("9")

    def test_serialize_deserialize_roundtrip(self) -> None:
        """Test serialization preserves window state."""
        window = RollingWindow(max_size=5)
        for i in range(1, 4):
            window.add(Decimal(str(i)))

        serialized = window.serialize()
        restored = RollingWindow.deserialize(serialized)

        assert restored.max_size == window.max_size
        assert list(restored.values) == list(window.values)
        assert restored.running_sum == window.running_sum
