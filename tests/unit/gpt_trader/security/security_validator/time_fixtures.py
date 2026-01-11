from __future__ import annotations

import time
from typing import Any

import pytest
from freezegun import freeze_time


@pytest.fixture
def frozen_time() -> Any:
    """Freeze time for deterministic rate limiting tests."""
    with freeze_time("2024-01-01 12:00:00") as frozen:
        yield frozen


@pytest.fixture
def rate_limiter_time_control() -> Any:
    """Time control fixture for rate limiter testing."""

    class TimeControl:
        def __init__(self):
            self.current_time = time.time()
            self.selfincrements = 0

        def advance(self, seconds: int) -> None:
            """Advance time by specified seconds."""
            self.current_time += seconds
            self.selfincrements += 1

        def get_time(self) -> float:
            """Get current time."""
            return self.current_time

    return TimeControl()
