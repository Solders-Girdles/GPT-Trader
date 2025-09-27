#!/usr/bin/env python3
"""
Stage 3 Runner wrapper.

Convenience entry that forwards to the Perps Bot CLI so existing docs and
preflight messages remain accurate.

Examples:
  python scripts/stage3_runner.py --profile dev --dev-fast
  python scripts/stage3_runner.py --profile canary --dry-run
"""

import sys

def main() -> int:
    from bot_v2.cli import main as perps_main
    return perps_main()


if __name__ == "__main__":
    sys.exit(main())

