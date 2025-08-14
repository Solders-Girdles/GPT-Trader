"""Integration test fixtures."""

import pytest

from ..fixtures.conftest import *  # Import shared fixtures


@pytest.fixture
def mock_data_source():
    """Mock data source for integration tests."""
    from unittest.mock import Mock

    return Mock()


@pytest.fixture
def temp_database():
    """Temporary database for integration tests."""
    # TODO: Implement temporary database setup
    pass
