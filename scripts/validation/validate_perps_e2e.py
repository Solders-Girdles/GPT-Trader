#!/usr/bin/env python3
"""
End-to-End (E2E) checks for perps runner, CLI, and quantization.

This script performs non-destructive checks:
 - Verifies presence of the runner, docs, and CLI command
 - Imports and constructs the PerpsBot (dev profile) and runs a minimal cycle
 - Validates enforce_perp_rules quantization on a sample product
"""

from __future__ import annotations

import os
import sys
from decimal import Decimal
from pathlib import Path
import tempfile

# Add repo root to sys.path BEFORE any project imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Set temp dir for EventStore in tests
if "EVENT_STORE_ROOT" not in os.environ:
    os.environ["EVENT_STORE_ROOT"] = tempfile.mkdtemp()


def ok(msg: str):
    print(f"✓ {msg}")


def fail(msg: str):
    print(f"✗ {msg}")
    sys.exit(1)


def check_files():
    cli_entry = REPO_ROOT / "src" / "bot_v2" / "cli" / "__init__.py"
    runbook = REPO_ROOT / "docs" / "ops" / "operations_runbook.md"
    if not cli_entry.exists():
        fail(f"CLI entry point not found: {cli_entry}")
    if not runbook.exists():
        fail(f"Operations runbook not found: {runbook}")
    ok("CLI entry point and operations runbook present")


def check_cli_command():
    from subprocess import run

    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    p = run([sys.executable, "-m", "bot_v2.cli", "--help"], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        fail(f"CLI help failed: {p.stderr}")
    if "--profile" not in p.stdout:
        fail("CLI help missing --profile flag")
    ok("CLI exposes help output")


def check_quantization_helper():
    try:
        from bot_v2.features.brokerages.core.interfaces import Product, MarketType
        from bot_v2.features.brokerages.coinbase.utilities import enforce_perp_rules
    except ImportError as e:
        fail(f"Import failed: {e}\nInstall project deps and run from repo root: pip install -e .")

    product = Product(
        symbol="BTC-PERP",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        step_size=Decimal("0.001"),
        min_size=Decimal("0.001"),
        price_increment=Decimal("0.01"),
        min_notional=Decimal("10"),
    )

    quantity, price = enforce_perp_rules(product, Decimal("1.23456789"), Decimal("50123.4567"))
    assert quantity == Decimal("1.234"), f"Quantity quantization mismatch: {quantity}"
    assert price == Decimal("50123.45"), f"Price quantization mismatch: {price}"
    ok("enforce_perp_rules quantizes quantity/price correctly")


def run_minimal_cycle():
    """Instantiate PerpsBot (dev) and run a small cycle to ensure wiring works."""
    try:
        import asyncio
        from bot_v2.orchestration.bootstrap import build_bot
        from bot_v2.orchestration.configuration import BotConfig
    except ImportError as e:
        fail(f"Import failed: {e}\nEnsure orchestration modules are importable")

    # Dev profile, but disable dry_run to exercise more code paths (still mocked broker)
    config = BotConfig.from_profile("dev", dry_run=False, symbols=["BTC-PERP"], update_interval=1)
    bot, _registry = build_bot(config)
    symbol = bot.config.symbols[0]

    async def one_cycle():
        # Prime marks
        await bot.update_marks()
        # Process the symbol once (may hold if no signal, but should not error)
        await bot.process_symbol(symbol)
        # We expect a decision entry to exist
        assert symbol in bot.last_decisions, f"No decision recorded for {symbol}"

    asyncio.run(one_cycle())
    ok("PerpsBot runs a minimal cycle in dev mode")


def main():
    print("\n=== E2E Validation ===")
    check_files()
    check_cli_command()
    check_quantization_helper()
    run_minimal_cycle()
    print("\n✅ E2E validation completed successfully")


if __name__ == "__main__":
    main()
