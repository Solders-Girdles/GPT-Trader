#!/usr/bin/env python3
"""
Production preflight entry point.

Delegates to the modular gpt_trader.preflight package which encapsulates the
individual validation steps and CLI orchestration.
"""

from __future__ import annotations

from gpt_trader.preflight import PreflightCheck, run_preflight_cli


def main() -> int:
    """Run the preflight CLI pipeline."""
    return run_preflight_cli()


if __name__ == "__main__":
    raise SystemExit(main())
