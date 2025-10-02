"""Backward-compatible CLI entry module.

The bulk of the CLI implementation now lives in :mod:`bot_v2.cli` (the
package). This thin wrapper exists so historical entry-points that import the
``main`` function from this module continue to work.
"""

from __future__ import annotations

import sys

from bot_v2.cli import main

__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - thin wrapper
    sys.exit(main())
