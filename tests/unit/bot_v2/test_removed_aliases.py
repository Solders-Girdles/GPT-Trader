"""Ensure deprecated compatibility modules remain removed."""

from __future__ import annotations

import importlib
import sys

import pytest


REMOVED_MODULES = (
    "bot_v2.logging_setup",
    "bot_v2.system_paths",
    "bot_v2.validate_calculations",
    "bot_v2.features.live_trade.execution_v3",
    "bot_v2.features.live_trade.strategies.perps_baseline_v2",
    "bot_v2.features.live_trade.strategies.week2_filters",
    "bot_v2.features.brokerages.coinbase.utils",
    "bot_v2.features.brokerages.coinbase.market_data_utils",
    "bot_v2.orchestration.mock_broker",
)


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_removed_modules_raise_import_error(module_name: str) -> None:
    """Importing a retired compatibility module should fail."""

    sys.modules.pop(module_name, None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
