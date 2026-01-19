"""Tests for ProfileLoader schema hydration (strategy/risk config)."""

from __future__ import annotations


class TestProfileStrategyHydration:
    """Tests for profile loading with strategy config."""

    def test_profile_loader_builds_strategy_config(self) -> None:
        """Test ProfileLoader builds PerpsStrategyConfig from schema."""
        from gpt_trader.app.config.profile_loader import ProfileLoader
        from gpt_trader.config.types import Profile

        loader = ProfileLoader()
        schema = loader.load(Profile.DEV)
        kwargs = loader.to_bot_config_kwargs(schema, Profile.DEV)

        # Check strategy config is present and correct type
        assert "strategy" in kwargs
        from gpt_trader.features.live_trade.strategies.perps_baseline import (
            PerpsStrategyConfig,
        )

        assert isinstance(kwargs["strategy"], PerpsStrategyConfig)
        # Check strategy fields are populated
        assert kwargs["strategy"].short_ma_period > 0
        assert kwargs["strategy"].long_ma_period > 0

    def test_profile_loader_builds_risk_config(self) -> None:
        """Test ProfileLoader builds BotRiskConfig from schema."""
        from gpt_trader.app.config.bot_config import BotRiskConfig
        from gpt_trader.app.config.profile_loader import ProfileLoader
        from gpt_trader.config.types import Profile

        loader = ProfileLoader()
        schema = loader.load(Profile.DEV)
        kwargs = loader.to_bot_config_kwargs(schema, Profile.DEV)

        # Check risk config is present and correct type
        assert "risk" in kwargs
        assert isinstance(kwargs["risk"], BotRiskConfig)
        # Check daily_loss_limit_pct is set
        assert kwargs["risk"].daily_loss_limit_pct > 0

    def test_paper_profile_exists(self) -> None:
        """Test PAPER profile is in Profile enum."""
        from gpt_trader.config.types import Profile

        assert hasattr(Profile, "PAPER")
        assert Profile.PAPER.value == "paper"

    def test_paper_profile_loads(self) -> None:
        """Test PAPER profile can be loaded."""
        from gpt_trader.app.config.profile_loader import ProfileLoader
        from gpt_trader.config.types import Profile

        loader = ProfileLoader()
        schema = loader.load(Profile.PAPER)
        # Paper profile should load (may come from YAML or defaults)
        assert schema.profile_name == "paper"
        # Note: mock_broker may be overridden by YAML file if it exists

    def test_profile_daily_loss_limit_pct_from_yaml(self) -> None:
        """Test daily_loss_limit_pct is parsed from profile YAML."""
        from gpt_trader.app.config.profile_loader import ProfileLoader
        from gpt_trader.config.types import Profile

        loader = ProfileLoader()
        # PAPER profile has daily_loss_limit_pct=0.05 in defaults
        schema = loader.load(Profile.PAPER)
        assert schema.risk.daily_loss_limit_pct == 0.05
