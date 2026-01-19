"""Tests for get_validation_metrics helper."""


class TestGetValidationMetrics:
    def test_returns_empty_failures_initially(self) -> None:
        from gpt_trader.app.config import BotConfig
        from gpt_trader.app.container import (
            ApplicationContainer,
            clear_application_container,
            set_application_container,
        )
        from gpt_trader.features.live_trade.execution.validation import get_validation_metrics

        container = ApplicationContainer(BotConfig())
        set_application_container(container)
        try:
            metrics = get_validation_metrics()

            assert "failures" in metrics
            assert "escalation_threshold" in metrics
            assert "any_escalated" in metrics
            assert metrics["failures"] == {}
            assert metrics["any_escalated"] is False
        finally:
            clear_application_container()

    def test_returns_failure_counts(self) -> None:
        from gpt_trader.app.config import BotConfig
        from gpt_trader.app.container import (
            ApplicationContainer,
            clear_application_container,
            set_application_container,
        )
        from gpt_trader.features.live_trade.execution.validation import get_validation_metrics

        container = ApplicationContainer(BotConfig())
        set_application_container(container)
        try:
            tracker = container.validation_failure_tracker
            tracker.record_failure("mark_staleness")
            tracker.record_failure("mark_staleness")
            tracker.record_failure("slippage_guard")

            metrics = get_validation_metrics()

            assert metrics["failures"]["mark_staleness"] == 2
            assert metrics["failures"]["slippage_guard"] == 1
            assert metrics["any_escalated"] is False
        finally:
            clear_application_container()

    def test_reports_escalation_status(self) -> None:
        from gpt_trader.app.config import BotConfig
        from gpt_trader.app.container import (
            ApplicationContainer,
            clear_application_container,
            set_application_container,
        )
        from gpt_trader.features.live_trade.execution.validation import get_validation_metrics

        container = ApplicationContainer(BotConfig())
        set_application_container(container)
        try:
            tracker = container.validation_failure_tracker
            tracker.escalation_threshold = 3
            tracker.record_failure("mark_staleness")
            tracker.record_failure("mark_staleness")
            tracker.record_failure("mark_staleness")

            metrics = get_validation_metrics()

            assert metrics["any_escalated"] is True
            assert metrics["escalation_threshold"] == 3
        finally:
            clear_application_container()
