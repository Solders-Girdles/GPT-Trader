import pytest


@pytest.fixture(autouse=True)
def fast_retry_sleep(fake_clock):
    """Auto-use deterministic clock so retry loops advance instantly."""
    return fake_clock
