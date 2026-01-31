from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Convenience wrapper for running local CI without installing the package.

    Prefer `uv run local-ci` for the supported entrypoint.
    """

    repo_root = Path(__file__).resolve().parents[2]

    # Ensure the `src/` layout is importable.
    sys.path.insert(0, str(repo_root / "src"))

    from gpt_trader.ci.local_ci import main as local_ci_main  # noqa: PLC0415

    return int(local_ci_main())


if __name__ == "__main__":
    raise SystemExit(main())
