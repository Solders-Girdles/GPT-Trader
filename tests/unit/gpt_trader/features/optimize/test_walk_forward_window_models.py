"""Unit tests for walk-forward window models."""

from __future__ import annotations

from datetime import datetime

from gpt_trader.features.optimize.walk_forward import WalkForwardWindow, WindowResult


class TestWalkForwardWindow:
    """Tests for WalkForwardWindow dataclass."""

    def test_window_creation(self) -> None:
        """Test basic window creation."""
        window = WalkForwardWindow(
            window_id=0,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 4, 1),
            test_start=datetime(2024, 4, 1),
            test_end=datetime(2024, 5, 1),
        )
        assert window.window_id == 0
        assert window.train_start == datetime(2024, 1, 1)
        assert window.train_end == datetime(2024, 4, 1)
        assert window.test_start == datetime(2024, 4, 1)
        assert window.test_end == datetime(2024, 5, 1)

    def test_train_days_property(self) -> None:
        """Test train_days property calculation."""
        window = WalkForwardWindow(
            window_id=0,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 4, 1),  # Jan + Feb + Mar (leap-year Feb)
            test_start=datetime(2024, 4, 1),
            test_end=datetime(2024, 5, 1),
        )
        assert window.train_days == 91

    def test_test_days_property(self) -> None:
        """Test test_days property calculation."""
        window = WalkForwardWindow(
            window_id=0,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 4, 1),
            test_start=datetime(2024, 4, 1),
            test_end=datetime(2024, 5, 1),  # 30 days in April
        )
        assert window.test_days == 30


class TestWindowResult:
    """Tests for WindowResult dataclass."""

    def test_valid_result(self) -> None:
        """Test creating a valid window result."""
        window = WalkForwardWindow(
            window_id=0,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 4, 1),
            test_start=datetime(2024, 4, 1),
            test_end=datetime(2024, 5, 1),
        )

        result = WindowResult(
            window=window,
            best_parameters={"rsi_period": 14, "ma_period": 20},
            optimization_trials=50,
            best_train_objective=1.5,
            test_objective_value=1.2,
            is_valid=True,
        )

        assert result.is_valid is True
        assert result.best_parameters["rsi_period"] == 14
        assert result.test_objective_value == 1.2

    def test_invalid_result_with_errors(self) -> None:
        """Test creating an invalid window result with errors."""
        window = WalkForwardWindow(
            window_id=0,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 4, 1),
            test_start=datetime(2024, 4, 1),
            test_end=datetime(2024, 5, 1),
        )

        result = WindowResult(
            window=window,
            best_parameters={},
            optimization_trials=50,
            best_train_objective=float("-inf"),
            is_valid=False,
            validation_errors=["No valid optimization trials completed"],
        )

        assert result.is_valid is False
        assert len(result.validation_errors) == 1
