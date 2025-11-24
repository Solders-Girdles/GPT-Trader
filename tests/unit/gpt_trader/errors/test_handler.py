from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from gpt_trader.errors import NetworkError, TradingError
from gpt_trader.errors.handler import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    ErrorHandler,
    RecoveryStrategy,
    RetryConfig,
    get_error_handler,
    set_error_handler,
    with_error_handling,
)


@pytest.fixture(autouse=True)
def patch_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gpt_trader.errors.handler.log_error", lambda error: None)


class TestCircuitBreaker:
    def test_closed_to_open_transition(self) -> None:
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(config)

        breaker.record_failure(NetworkError("boom", url="u"))
        breaker.record_failure(NetworkError("boom", url="u"))

        assert breaker.state is CircuitBreakerState.OPEN
        assert breaker.failure_count == 2

    def test_open_to_half_open_after_timeout(self) -> None:
        config = CircuitBreakerConfig(recovery_timeout=5)
        breaker = CircuitBreaker(config)
        breaker.state = CircuitBreakerState.OPEN
        breaker.last_failure_time = datetime.now() - timedelta(seconds=10)

        attempt = breaker.should_attempt_call()

        assert attempt is True
        assert breaker.state is CircuitBreakerState.HALF_OPEN
        assert breaker.success_count == 0

    def test_half_open_to_closed_recovery(self) -> None:
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker(config)
        breaker.state = CircuitBreakerState.HALF_OPEN
        breaker.success_count = 2

        breaker.record_success()

        assert breaker.state is CircuitBreakerState.CLOSED
        assert breaker.success_count == 0
        assert breaker.failure_count == 0

    def test_half_open_failure_reopens(self) -> None:
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)
        breaker.state = CircuitBreakerState.HALF_OPEN

        breaker.record_failure(NetworkError("boom", url="u"))

        assert breaker.state is CircuitBreakerState.OPEN

    def test_non_targeted_exceptions_ignored(self) -> None:
        config = CircuitBreakerConfig(expected_exception_types=(NetworkError,))
        breaker = CircuitBreaker(config)

        breaker.record_failure(ValueError("ignore me"))

        assert breaker.state is CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0


class TestRetryLogic:
    def test_exponential_backoff_with_jitter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = ErrorHandler(
            retry_config=RetryConfig(max_attempts=3, initial_delay=1.0, exponential_base=2.0)
        )
        sleep_calls: list[float] = []
        monkeypatch.setattr("time.sleep", sleep_calls.append)
        monkeypatch.setattr("random.random", lambda: 0.5)

        attempts: list[int] = []

        def flaky(_: int) -> str:
            if len(attempts) < 2:
                attempts.append(len(attempts))
                raise NetworkError("retry", url="u")
            return "ok"

        result = handler.with_retry(flaky, 1)

        assert result == "ok"
        assert sleep_calls == pytest.approx([1.0, 2.0], rel=1e-6)

    def test_max_attempts_enforced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = ErrorHandler(retry_config=RetryConfig(max_attempts=2, jitter=False))
        monkeypatch.setattr("time.sleep", lambda _: None)

        with pytest.raises(TradingError) as exc:
            handler.with_retry(lambda: (_ for _ in ()).throw(NetworkError("boom", url="u")))

        assert exc.value.error_code == "RETRY_EXHAUSTED"

    def test_non_recoverable_errors_fail_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = ErrorHandler()
        monkeypatch.setattr("time.sleep", lambda _: None)

        def bad() -> None:
            raise TradingError("fatal", recoverable=False)

        with pytest.raises(TradingError) as exc:
            handler.with_retry(bad)

        assert exc.value.recoverable is False

    def test_fallback_handler_invoked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fallback_calls: list[tuple[Any, ...]] = []

        def fallback(*args: Any, **kwargs: Any) -> str:
            fallback_calls.append(args)
            return "fallback"

        handler = ErrorHandler(
            retry_config=RetryConfig(max_attempts=1, jitter=False),
            fallback_handlers={NetworkError: fallback},
        )
        monkeypatch.setattr("time.sleep", lambda _: None)

        def bad(arg: str) -> None:
            raise NetworkError("fail", url="u")

        result = handler.with_retry(bad, "param", recovery_strategy=RecoveryStrategy.FALLBACK)

        assert result == "fallback"
        assert fallback_calls == [("param",)]


class TestRecoveryStrategies:
    def test_retry_strategy_eventually_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = ErrorHandler(retry_config=RetryConfig(max_attempts=2, jitter=False))
        monkeypatch.setattr("time.sleep", lambda _: None)

        calls = {"count": 0}

        def sometimes() -> str:
            if calls["count"] == 0:
                calls["count"] += 1
                raise NetworkError("temp", url="u")
            return "result"

        assert handler.with_retry(sometimes, recovery_strategy=RecoveryStrategy.RETRY) == "result"

    def test_fail_fast_strategy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = ErrorHandler(retry_config=RetryConfig(max_attempts=3, jitter=False))
        monkeypatch.setattr("time.sleep", lambda _: None)

        def bad() -> None:
            raise NetworkError("no retry", url="u")

        with pytest.raises(TradingError) as exc:
            handler.with_retry(bad, recovery_strategy=RecoveryStrategy.FAIL_FAST)

        assert exc.value.error_code == "NETWORK_ERROR"

    def test_degrade_strategy_returns_none_and_records_error(self) -> None:
        handler = ErrorHandler()

        result = handler.handle_error(
            NetworkError("warn", url="u"), recovery_strategy=RecoveryStrategy.DEGRADE
        )

        assert result is None
        assert handler.error_history[-1].error_code == "NETWORK_ERROR"


class TestErrorHistory:
    def test_error_recording(self) -> None:
        handler = ErrorHandler()

        handler.handle_error(NetworkError("one", url="u"))
        handler.handle_error(NetworkError("two", url="u"))

        assert len(handler.error_history) == 2
        assert handler.error_history[-1].message == "two"

    def test_rolling_window_max_100(self) -> None:
        handler = ErrorHandler()

        for idx in range(105):
            handler.handle_error(NetworkError(f"err{idx}", url="u"))

        assert len(handler.error_history) == handler.max_history == 100
        assert handler.error_history[0].message == "err5"

    def test_get_error_stats(self) -> None:
        handler = ErrorHandler()
        handler.handle_error(NetworkError("err", url="u"))
        handler.handle_error(NetworkError("err", url="u"))

        stats = handler.get_error_stats()

        assert stats["total_errors"] == 2
        assert stats["error_types"]["NETWORK_ERROR"] == 2
        assert stats["circuit_breaker_state"] == CircuitBreakerState.CLOSED.value
        assert "last_error" in stats


class TestDecorator:
    def test_with_error_handling_decorator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = get_error_handler()
        handler = ErrorHandler(
            retry_config=RetryConfig(max_attempts=2, jitter=False),
            fallback_handlers={Exception: lambda: "fallback"},
        )
        set_error_handler(handler)
        monkeypatch.setattr("time.sleep", lambda _: None)

        calls = {"count": 0}

        @with_error_handling(recovery_strategy=RecoveryStrategy.FALLBACK)
        def unstable() -> str:
            if calls["count"] == 0:
                calls["count"] += 1
                raise NetworkError("decorated", url="u")
            return "ok"

        try:
            assert unstable() == "fallback"
        finally:
            set_error_handler(original)
