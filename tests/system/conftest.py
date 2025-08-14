"""System test fixtures."""

import pytest
from ..fixtures.conftest import *  # Import shared fixtures


@pytest.fixture
def isolated_environment():
    """Isolated environment for system tests."""
    # TODO: Implement isolated environment setup
    pass


@pytest.fixture
def test_config():
    """Test configuration for system tests."""
    return {"environment": "test", "debug": True, "logging_level": "DEBUG"}
