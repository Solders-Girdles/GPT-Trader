"""Payload validation errors for SecretsManager."""

from __future__ import annotations

from typing import Any


class TestPayloadValidationErrors:
    """Verify error handling for invalid payload inputs."""

    def test_invalid_secret_payload_type(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback

        for payload in ["string_payload", 123, ["list", "payload"], None]:
            assert manager.store_secret("test/invalid", payload) is False

    def test_as_secret_dict_validation(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback

        valid_dict = {"key": "value"}
        assert manager._as_secret_dict(valid_dict) == valid_dict

        for value in ["string", 123, ["list"], None]:
            assert manager._as_secret_dict(value) is None

    def test_secret_path_validation(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback

        valid_paths = [
            "simple",
            "nested/path",
            "path-with-dashes",
            "path_with_underscores",
            "path.with.dots",
            "very/deep/nested/path/structure",
        ]

        for path in valid_paths:
            assert manager.store_secret(path, {"path": path}) is True

        assert manager.store_secret("", {"empty": True}) is True
        assert manager.store_secret(None, {"none": True}) is False  # type: ignore[arg-type]
