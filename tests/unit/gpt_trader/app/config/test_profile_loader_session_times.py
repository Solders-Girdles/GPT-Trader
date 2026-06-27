"""Session-time validation tests for profile loading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gpt_trader.app.config.profile_loader import (
    ProfileLoader,
    ProfileSchema,
    ProfileValidationError,
)
from gpt_trader.config.types import Profile


def test_loader_invalid_session_time_re_raises_with_field_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_logger = MagicMock()
    monkeypatch.setattr("gpt_trader.app.config.profile_loader.logger", mock_logger)

    yaml_path = tmp_path / "dev.yaml"
    yaml_path.write_text(
        """
profile_name: "dev"
session:
  start_time: "09:00"
  end_time: "2026-06-27 15:00"
"""
    )

    loader = ProfileLoader(profiles_dir=tmp_path)
    with pytest.raises(ProfileValidationError, match=r"session\.end_time"):
        loader.load(Profile.DEV)

    assert mock_logger.warning.call_count == 1
    logged_kwargs = mock_logger.warning.call_args.kwargs
    details = logged_kwargs.get("details", {})
    assert "session.end_time" in details["reason"]
    assert details["path"].endswith("dev.yaml")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("start_time", "not-a-time"),
        ("end_time", "2026-06-27 15:00"),
    ],
)
def test_from_yaml_rejects_invalid_session_times(field_name: str, value: str) -> None:
    data = {
        "profile_name": "invalid-session",
        "session": {
            "start_time": "09:00",
            "end_time": "17:00",
            field_name: value,
        },
    }

    with pytest.raises(
        ProfileValidationError,
        match=rf"session\.{field_name}.*HH:MM",
    ):
        ProfileSchema.from_yaml(data, "invalid-session")


def test_from_yaml_preserves_explicit_null_session_times() -> None:
    data = {
        "profile_name": "null-session",
        "session": {
            "start_time": None,
            "end_time": None,
        },
    }

    schema = ProfileSchema.from_yaml(data, "null-session")

    assert schema.session.start_time is None
    assert schema.session.end_time is None
