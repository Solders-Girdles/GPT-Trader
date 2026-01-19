"""Tests for StrategyOrchestrator class structure."""

from __future__ import annotations

from gpt_trader.features.live_trade.orchestrator.orchestrator import StrategyOrchestrator


class TestStrategyOrchestratorClassStructure:
    """Test StrategyOrchestrator class structure."""

    def test_requires_context_attribute(self) -> None:
        """Test that requires_context is True."""
        assert StrategyOrchestrator.requires_context is True

    def test_inherits_from_mixins(self) -> None:
        """Test that StrategyOrchestrator inherits from expected mixins."""
        from gpt_trader.features.live_trade.orchestrator.context import ContextBuilderMixin
        from gpt_trader.features.live_trade.orchestrator.decision import DecisionEngineMixin
        from gpt_trader.features.live_trade.orchestrator.initialization import (
            StrategyInitializationMixin,
        )
        from gpt_trader.features.live_trade.orchestrator.spot_filters import SpotFiltersMixin

        assert issubclass(StrategyOrchestrator, SpotFiltersMixin)
        assert issubclass(StrategyOrchestrator, DecisionEngineMixin)
        assert issubclass(StrategyOrchestrator, ContextBuilderMixin)
        assert issubclass(StrategyOrchestrator, StrategyInitializationMixin)
