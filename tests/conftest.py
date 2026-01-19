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
def strict_container_mode(monkeypatch):
    """Enable strict container mode for all tests.

    This ensures tests fail fast if they use get_failure_tracker()
    without properly setting up an application container.
    """
    monkeypatch.setenv("GPT_TRADER_STRICT_CONTAINER", "1")


@pytest.fixture
def application_container(mock_config):
    """Provide an ApplicationContainer for tests that need it.

    This fixture sets up a container with the mock_config and registers
    it globally. It automatically clears the container after the test.

    Requires a mock_config fixture to be defined in the test module.
    """
    from gpt_trader.app.container import (
        ApplicationContainer,
        clear_application_container,
        set_application_container,
    )

    container = ApplicationContainer(mock_config)
    set_application_container(container)
    yield container
    clear_application_container()


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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply path-based markers for consistent selection.

    Allows running suites via `pytest -m unit|integration|contract|real_api` without
    requiring every file to declare a redundant marker.
    """
    repo_root = Path(__file__).resolve().parents[1]

    for item in items:
        try:
            rel = Path(str(item.fspath)).resolve().relative_to(repo_root).as_posix()
        except Exception:
            continue

        if rel.startswith("tests/unit/"):
            item.add_marker(pytest.mark.unit)
        elif rel.startswith("tests/integration/"):
            item.add_marker(pytest.mark.integration)
        elif rel.startswith("tests/property/"):
            item.add_marker(pytest.mark.property)
        elif rel.startswith("tests/contract/"):
            item.add_marker(pytest.mark.contract)
        elif rel.startswith("tests/real_api/"):
            item.add_marker(pytest.mark.real_api)
