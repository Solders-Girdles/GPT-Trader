"""Concurrent file operation tests for SecretsManager."""

from __future__ import annotations

from typing import Any


class TestFileConcurrency:
    """Ensure repeated writes from multiple threads succeed."""

    def test_file_concurrent_access(
        self, secrets_manager_with_fallback: Any, sample_secrets: dict[str, dict[str, Any]]
    ) -> None:
        import threading

        manager = secrets_manager_with_fallback
        secret_path = "test/concurrent"
        secret_data = sample_secrets["brokers/coinbase"]

        results: list[bool] = []

        def store_secret() -> None:
            results.append(manager.store_secret(secret_path, secret_data))

        threads = [threading.Thread(target=store_secret) for _ in range(5)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert all(results)
        assert manager.get_secret(secret_path) == secret_data
