"""Module executed when running ``python -m bot_v2.cli``."""

from __future__ import annotations

import sys

from bot_v2.cli import main

if __name__ == "__main__":  # pragma: no cover - exercised via CLI smoke tests
    sys.exit(main())
