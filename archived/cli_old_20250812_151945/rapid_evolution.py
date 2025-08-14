from __future__ import annotations

import argparse

from bot.cli_rapid_evolution import main as rapid_main


def add_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "rapid-evolution",
        help="Rapid evolutionary strategy optimization",
        description="""
        Use rapid evolutionary algorithms to optimize trading strategy parameters.

        This command provides fast, efficient optimization using evolutionary computation
        techniques to find optimal parameter combinations for your trading strategies.
        """,
    )
    p.set_defaults(func=lambda _: rapid_main())
    return p
