#!/usr/bin/env python3
"""
Legacy bundling helper (retired).

The historical experimental modules now live in pre-generated archives under
`var/legacy/`. To rebuild them, check out an earlier commit that still contains
`archived/experimental/**` and `src/gpt_trader/**`, then archive the paths
manually.
"""

from __future__ import annotations

import sys
from textwrap import dedent


def main() -> None:
    message = dedent(
        """
        Legacy bundling helper retired.

        Use the pre-generated archive at var/legacy/legacy_bundle_latest.tar.gz or
        check out a commit that still contains archived/experimental and src/gpt_trader
        before creating a manual tarball. See docs/archive/legacy_recovery.md for details.
        """
    ).strip()
    print(message)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
