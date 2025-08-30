"""Performance test fixtures."""

import time

import pytest

# Import shared fixtures using absolute import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fixtures.conftest import *


@pytest.fixture
def performance_timer():
    """Timer for performance measurements."""

    class Timer:
        def __init__(self):
            self.start_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def elapsed(self):
            if self.start_time is None:
                return 0
            return time.perf_counter() - self.start_time

    return Timer()
