"""
Preflight checks package for GPT-Trader.

Provides orchestrated production preflight verification including CLI entry
points and individual validation steps that can be reused or extended.
"""

from .core import PreflightCheck
from .cli import main as run_preflight_cli

__all__ = ["PreflightCheck", "run_preflight_cli"]
