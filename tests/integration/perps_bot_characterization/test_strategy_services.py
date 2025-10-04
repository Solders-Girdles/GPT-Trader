"""
Characterization Tests for StrategyOrchestrator Extracted Services

Tests documenting StrategyOrchestrator extracted services (Phase 1-4 refactor).
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock

from bot_v2.orchestration.perps_bot import PerpsBot


@pytest.mark.integration
@pytest.mark.characterization
class TestStrategyOrchestratorExtractedServices:
    """Characterize StrategyOrchestrator extracted services (Phase 1-4 refactor)"""

    def test_strategy_orchestrator_has_equity_calculator(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: StrategyOrchestrator must have EquityCalculator service"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        assert hasattr(bot.strategy_orchestrator, "equity_calculator")
        assert bot.strategy_orchestrator.equity_calculator is not None

    def test_strategy_orchestrator_has_risk_gate_validator(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: StrategyOrchestrator must have RiskGateValidator service"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # Lazy initialization - access property to trigger creation
        validator = bot.strategy_orchestrator.risk_gate_validator
        assert validator is not None

    def test_strategy_orchestrator_has_strategy_registry(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: StrategyOrchestrator must have StrategyRegistry service"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # Lazy initialization - access property to trigger creation
        registry = bot.strategy_orchestrator.strategy_registry
        assert registry is not None

    def test_strategy_orchestrator_has_strategy_executor(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: StrategyOrchestrator must have StrategyExecutor service"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # Lazy initialization - access property to trigger creation
        executor = bot.strategy_orchestrator.strategy_executor
        assert executor is not None

    @pytest.mark.asyncio
    async def test_process_symbol_uses_extracted_services(
        self, monkeypatch, tmp_path, minimal_config, mock_quote
    ):
        """Document: process_symbol must use extracted services for equity, validation, evaluation"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # Setup test data
        bot.mark_windows["BTC-USD"] = [Decimal("50000"), Decimal("50100")]
        bot.broker.list_balances = Mock(
            return_value=[Mock(asset="USD", total=Decimal("10000"), available=Decimal("10000"))]
        )
        bot.broker.list_positions = Mock(return_value=[])

        # Initialize strategy
        bot.strategy_orchestrator.init_strategy()

        # Mock extracted services to verify they're called
        original_equity_calc = bot.strategy_orchestrator.equity_calculator.calculate
        original_risk_validate = bot.strategy_orchestrator.risk_gate_validator.validate_gates
        original_strategy_eval = bot.strategy_orchestrator.strategy_executor.evaluate
        original_strategy_record = bot.strategy_orchestrator.strategy_executor.record_decision

        bot.strategy_orchestrator.equity_calculator.calculate = Mock(
            side_effect=original_equity_calc
        )
        bot.strategy_orchestrator.risk_gate_validator.validate_gates = Mock(
            side_effect=original_risk_validate
        )
        bot.strategy_orchestrator.strategy_executor.evaluate = Mock(
            side_effect=original_strategy_eval
        )
        bot.strategy_orchestrator.strategy_executor.record_decision = Mock(
            side_effect=original_strategy_record
        )

        # Execute
        await bot.strategy_orchestrator.process_symbol("BTC-USD")

        # Verify extracted services were used
        bot.strategy_orchestrator.equity_calculator.calculate.assert_called_once()
        bot.strategy_orchestrator.risk_gate_validator.validate_gates.assert_called_once()
        bot.strategy_orchestrator.strategy_executor.evaluate.assert_called_once()
        bot.strategy_orchestrator.strategy_executor.record_decision.assert_called_once()
