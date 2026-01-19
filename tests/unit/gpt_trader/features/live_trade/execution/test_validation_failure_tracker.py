"""Tests for ValidationFailureTracker."""


class TestValidationFailureTracker:
    def test_initial_failure_count_is_zero(self) -> None:
        from gpt_trader.features.live_trade.execution.validation import ValidationFailureTracker

        tracker = ValidationFailureTracker()
        assert tracker.get_failure_count("mark_staleness") == 0

    def test_record_failure_increments_count(self) -> None:
        from gpt_trader.features.live_trade.execution.validation import ValidationFailureTracker

        tracker = ValidationFailureTracker()
        tracker.record_failure("mark_staleness")
        assert tracker.get_failure_count("mark_staleness") == 1

        tracker.record_failure("mark_staleness")
        assert tracker.get_failure_count("mark_staleness") == 2

    def test_record_success_resets_count(self) -> None:
        from gpt_trader.features.live_trade.execution.validation import ValidationFailureTracker

        tracker = ValidationFailureTracker()
        tracker.record_failure("mark_staleness")
        tracker.record_failure("mark_staleness")
        assert tracker.get_failure_count("mark_staleness") == 2

        tracker.record_success("mark_staleness")
        assert tracker.get_failure_count("mark_staleness") == 0

    def test_escalation_triggered_at_threshold(self) -> None:
        from gpt_trader.features.live_trade.execution.validation import ValidationFailureTracker

        escalation_called = []
        tracker = ValidationFailureTracker(
            escalation_threshold=3,
            escalation_callback=lambda: escalation_called.append(True),
        )

        assert tracker.record_failure("mark_staleness") is False
        assert tracker.record_failure("mark_staleness") is False
        assert len(escalation_called) == 0

        assert tracker.record_failure("mark_staleness") is True
        assert len(escalation_called) == 1

    def test_escalation_without_callback(self) -> None:
        from gpt_trader.features.live_trade.execution.validation import ValidationFailureTracker

        tracker = ValidationFailureTracker(
            escalation_threshold=2,
            escalation_callback=None,
        )

        tracker.record_failure("test")
        escalated = tracker.record_failure("test")
        assert escalated is True

    def test_separate_tracking_per_check_type(self) -> None:
        from gpt_trader.features.live_trade.execution.validation import ValidationFailureTracker

        tracker = ValidationFailureTracker()
        tracker.record_failure("mark_staleness")
        tracker.record_failure("mark_staleness")
        tracker.record_failure("slippage_guard")

        assert tracker.get_failure_count("mark_staleness") == 2
        assert tracker.get_failure_count("slippage_guard") == 1
