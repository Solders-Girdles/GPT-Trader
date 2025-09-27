"""End-to-end smoke test using MockBroker and --dev-fast.

Ensures the CLI path initializes, runs one cycle, writes health status,
and shuts down cleanly without real network calls.
"""

import pytest
import importlib
import json
from pathlib import Path

pytestmark = pytest.mark.integration


def test_full_cycle_mock_broker(tmp_path, monkeypatch):
    # Isolate all file writes under a temp cwd
    monkeypatch.chdir(tmp_path)
    # Force mock broker and avoid background threads
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))

    cli = importlib.import_module("bot_v2.cli")

    # Prevent background WS threads for determinism
    from bot_v2.orchestration.perps_bot import PerpsBot
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    # Run CLI: --profile dev with --dev-fast so it runs a single cycle and exits
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--profile",
            "dev",
            "--dry-run",
            "--dev-fast",
        ],
    )

    exit_code = cli.main()
    assert exit_code == 0

    # Health file should be written in ./data/perps_bot/dev/health.json
    health = Path("data/perps_bot/dev/health.json")
    assert health.exists(), "Expected health.json written by perps bot"
    payload = json.loads(health.read_text())
    assert payload.get("ok") is True
