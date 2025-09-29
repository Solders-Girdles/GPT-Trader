"""Tests for adaptive portfolio configuration serialization."""

from __future__ import annotations

import json
from pathlib import Path

from bot_v2.features.adaptive_portfolio.config_manager import ConfigManager


def _sample_raw_config() -> dict:
    return {
        "version": "1.0",
        "last_updated": "2025-01-01",
        "description": "Test configuration",
        "tiers": {
            "micro": {
                "name": "Micro",
                "range": [500, 2500],
                "positions": {"min": 1, "max": 3, "target": 2},
                "min_position_size": 250,
                "strategies": ["momentum"],
                "risk": {
                    "daily_limit_pct": 1.0,
                    "quarterly_limit_pct": 5.0,
                    "position_stop_loss_pct": 4.0,
                    "max_sector_exposure_pct": 40.0,
                },
                "trading": {
                    "max_trades_per_week": 5,
                    "account_type": "cash",
                    "settlement_days": 2,
                    "pdt_compliant": True,
                },
            }
        },
        "costs": {
            "commission_per_trade": 0.0,
            "spread_estimate_pct": 0.1,
            "slippage_pct": 0.2,
            "financing_rate_annual_pct": 4.5,
        },
        "market_constraints": {
            "min_share_price": 5.0,
            "max_share_price": 500.0,
            "min_daily_volume": 100000,
            "excluded_sectors": ["OTC"],
            "excluded_symbols": ["PENNY"],
            "market_hours_only": True,
        },
        "validation": {
            "min_account_size": 500,
            "max_positions_any_tier": 10,
        },
        "rebalancing": {
            "frequency_days": 7,
            "threshold_pct": 10.0,
        },
    }


def test_config_round_trip_serialization(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    manager = ConfigManager(config_path=path)
    raw = _sample_raw_config()

    config = manager._parse_config(raw)
    serialized = manager._config_to_dict(config)

    assert serialized == raw

    manager.save_config(config, backup=False)
    written = json.loads(path.read_text())
    assert written == raw


def test_config_serialization_creates_defensive_copies(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    manager = ConfigManager(config_path=path)
    raw = _sample_raw_config()
    config = manager._parse_config(raw)

    serialized = manager._config_to_dict(config)
    serialized["tiers"]["micro"]["strategies"].append("mean_reversion")
    serialized["validation"]["min_account_size"] = 999

    round_two = manager._config_to_dict(config)
    assert round_two == raw

    # Ensure external mutation does not bleed into persisted artifact
    manager.save_config(config, backup=False)
    written = json.loads(path.read_text())
    assert written == raw
