"""CLI package for GPT-Trader.

Submodules register their own subparsers via add_subparser(subparsers).
The package entrypoint is `bot.cli.__main__.main`.
This module re-exports `main` for compatibility with imports like
`from bot.cli import main`.
"""

from __future__ import annotations

# Don't import from __main__ here to avoid circular import warning
# Users should either:
# 1. Run as: python -m bot.cli
# 2. Import as: from bot.cli.__main__ import main
