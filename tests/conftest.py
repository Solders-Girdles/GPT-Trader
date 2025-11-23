"""
Minimal Conftest.
"""
import pytest
import asyncio

@pytest.fixture
def anyio_backend():
    return 'asyncio'
