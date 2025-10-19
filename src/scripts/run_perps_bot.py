"""Compatibility shim for the legacy `scripts/run_perps_bot.py` entry point."""

from __future__ import annotations

from bot_v2.cli import main as cli_main

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Invoke the modern CLI entry point."""
    return cli_main(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
