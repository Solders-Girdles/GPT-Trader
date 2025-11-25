"""
Minimal Conftest.
"""

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def reset_correlation_context():
    """Reset correlation context before and after each test to prevent pollution."""
    from gpt_trader.logging.correlation import set_correlation_id, set_domain_context

    # Reset before test
    set_correlation_id("")
    set_domain_context({})

    yield

    # Reset after test
    set_correlation_id("")
    set_domain_context({})
