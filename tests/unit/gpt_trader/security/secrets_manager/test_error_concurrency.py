"""Concurrency and resource pressure error tests for SecretsManager."""

from __future__ import annotations

from typing import Any


class TestConcurrencyErrors:
    """Handle concurrent writes and resource exhaustion gracefully."""

    def test_concurrent_operation_error(
        self, secrets_manager_with_fallback: Any, fake_clock
    ) -> None:
        import threading

        manager = secrets_manager_with_fallback
        errors: list[Exception] = []

        def failing_operation() -> None:
            try:
                for index in range(10):
                    manager.store_secret(f"test/concurrent/{index}", {"key": "value"})
                    fake_clock(0.001)
            except Exception as exc:  # pragma: no cover - diagnostic capture
                errors.append(exc)

        threads = [threading.Thread(target=failing_operation) for _ in range(5)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive()

        assert errors == []

    def test_memory_error_handling(
        self, secrets_manager_with_fallback: Any, monkeypatch: Any
    ) -> None:
        manager = secrets_manager_with_fallback
        original_encrypt = manager._require_cipher().encrypt

        def failing_encrypt(data):
            if len(data) > 100:
                raise MemoryError("Out of memory")
            return original_encrypt(data)

        monkeypatch.setattr(manager._require_cipher(), "encrypt", failing_encrypt)

        assert manager.store_secret("test/small", {"key": "v"}) is True

        large_data = {"key": "v" * 1000}
        assert manager.store_secret("test/large", large_data) is False
