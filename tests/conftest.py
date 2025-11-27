"""
Minimal Conftest.
"""

import subprocess
from pathlib import Path

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


def pytest_sessionfinish(session, exitstatus):
    """Clean up cache files after test session completes."""
    repo_root = Path(__file__).resolve().parents[1]
    cleanup_script = repo_root / "scripts" / "maintenance" / "cleanup_workspace.py"

    if cleanup_script.exists():
        subprocess.run(
            ["python", str(cleanup_script), "--apply", "--quiet", "--preserve-hypothesis"],
            cwd=repo_root,
            capture_output=True,
        )
