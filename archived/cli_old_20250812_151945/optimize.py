from __future__ import annotations

import argparse

from bot.optimization.cli import main as optimization_main


def add_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "optimize",
        help="Optimize trading strategy parameters",
        description="""
        Optimize your trading strategy parameters using advanced optimization techniques.

        Examples:
            # Basic optimization
            gpt-trader optimize --symbol AAPL --start 2023-01-01 --end 2023-12-31

            # Grid search optimization
            gpt-trader optimize --symbol-list universe.csv --start 2023-01-01 --end 2023-12-31 \\
                --method grid --param donchian:20:100:10 --param atr:10:50:5
        """,
    )
    p.add_argument(
        "--features",
        default="",
        help="Comma-separated feature sets to compose for research workflows",
    )
    # We simply reuse bot.optimization.cli:main by forwarding argv when invoked
    p.set_defaults(func=_handle)
    return p


def _handle(args: argparse.Namespace) -> None:
    # If features provided via profile/CLI, expose as env for optimization.cli to read (lightweight handoff)
    import os

    if getattr(args, "features", ""):
        os.environ["GPT_TRADER_FEATURES"] = args.features
    optimization_main()
