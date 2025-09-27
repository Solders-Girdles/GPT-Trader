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
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

# Set temp dir for EventStore in tests
if 'EVENT_STORE_ROOT' not in os.environ:
    os.environ['EVENT_STORE_ROOT'] = tempfile.mkdtemp()


def ok(msg: str):
    print(f"✓ {msg}")


def fail(msg: str):
    print(f"✗ {msg}")
    sys.exit(1)


def check_files():
    repo_root = Path(__file__).resolve().parents[1]
    runner = repo_root / 'scripts' / 'run_perps_bot.py'
    runbook = repo_root / 'docs' / 'RUNBOOK_PERPS.md'
    if not runner.exists():
        fail(f"Runner not found: {runner}")
    if not runbook.exists():
        fail(f"Runbook not found: {runbook}")
    ok("Runner and runbook present")


def check_cli_command():
    from subprocess import run, PIPE
    repo_root = Path(__file__).resolve().parents[1]
    cli = repo_root / 'src' / 'bot_v2' / 'simple_cli.py'
    if not cli.exists():
        fail("simple_cli.py not found")
    p = run([sys.executable, str(cli), '--help'], stdout=PIPE, stderr=PIPE, text=True)
    if p.returncode != 0:
        fail(f"CLI help failed: {p.stderr}")
    if 'perps' not in p.stdout:
        fail("CLI missing 'perps' command")
    ok("CLI exposes 'perps' command")


def check_quantization_helper():
    try:
        from bot_v2.features.brokerages.core.interfaces import Product, MarketType
        from bot_v2.features.brokerages.coinbase.utils import enforce_perp_rules
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
        min_notional=Decimal("10")
    )

    qty, price = enforce_perp_rules(product, Decimal("1.23456789"), Decimal("50123.4567"))
    assert qty == Decimal("1.234"), f"Qty quantization mismatch: {qty}"
    assert price == Decimal("50123.45"), f"Price quantization mismatch: {price}"
    ok("enforce_perp_rules quantizes qty/price correctly")


def run_minimal_cycle():
    """Instantiate PerpsBot (dev) and run a small cycle to ensure wiring works."""
    try:
        import asyncio
        from scripts.run_perps_bot import PerpsBot, BotConfig
    except ImportError as e:
        fail(f"Import failed: {e}\nEnsure scripts/run_perps_bot.py exists")

    # Dev profile, but disable dry_run to exercise more code paths (still mocked broker)
    config = BotConfig.from_profile('dev', dry_run=False, symbols=["BTC-PERP"], update_interval=1)
    bot = PerpsBot(config)

    async def one_cycle():
        # Prime marks
        await bot.update_marks()
        # Process the symbol once (may hold if no signal, but should not error)
        await bot.process_symbol("BTC-PERP")
        # We expect a decision entry to exist
        assert "BTC-PERP" in bot.last_decisions, "No decision recorded for BTC-PERP"

    asyncio.run(one_cycle())
    ok("PerpsBot runs a minimal cycle in dev mode")


def main():
    print("\n=== E2E Validation ===")
    check_files()
    check_cli_command()
    check_quantization_helper()
    run_minimal_cycle()
    print("\n✅ E2E validation completed successfully")


if __name__ == '__main__':
    main()
