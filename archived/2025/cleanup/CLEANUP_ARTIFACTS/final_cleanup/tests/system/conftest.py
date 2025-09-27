"""System test fixtures."""

import pytest

# Import shared fixtures using absolute import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fixtures.conftest import *


@pytest.fixture
def isolated_environment():
    """Isolated environment for system tests."""
    # TODO: Implement isolated environment setup
    pass


@pytest.fixture
def test_config():
    """Test configuration for system tests."""
    return {"environment": "test", "debug": True, "logging_level": "DEBUG"}
