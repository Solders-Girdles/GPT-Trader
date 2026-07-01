"""Strategy-signal proposal routing (#1033).

Covers the default-off gate that wires live strategy decisions into the
approval-gated trade-idea workflow via ``TradeIdeaService.propose()``:

- disabled gate leaves direct execution untouched (no proposals),
- enabled gate proposes supported buy decisions and submits nothing,
- enabled gate never submits or proposes for non-buy actions,
- the proposal path never approves an idea (no ``ApprovalPolicy`` bypass).
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gpt_trader.features.live_trade.engines.strategy import TradingEngine
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision
from gpt_trader.features.trade_ideas import (
    TradeDirection,
    TradeIdeaState,
    create_trade_idea_service,
)


def _enable_proposals(engine: TradingEngine, tmp_path, monkeypatch) -> None:
    """Turn the gate on and point the trade-idea store at an isolated root."""
    monkeypatch.setenv("GPT_TRADER_IDEAS_ROOT", str(tmp_path))
    engine.context.config.strategy_signal_proposals_enabled = True
    engine._init_strategy_proposal_bridge()


def _ok_result() -> SimpleNamespace:
    return SimpleNamespace(blocked=False, failed=False, reason=None, error=None)


@pytest.mark.asyncio
async def test_disabled_gate_leaves_execution_untouched(
    engine, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPT_TRADER_IDEAS_ROOT", str(tmp_path))
    assert engine._strategy_proposal_adapter is None
    assert engine._trade_idea_service is None

    validate = AsyncMock(return_value=_ok_result())
    monkeypatch.setattr(engine, "_validate_and_place_order", validate)

    await engine._handle_decision(
        symbol="BTC-USD",
        decision=Decision(Action.BUY, "reclaim", 0.82),
        price=Decimal("50000"),
        equity=Decimal("1000"),
        position_state=None,
    )

    validate.assert_awaited_once()
    # No proposal store was ever created while disabled.
    assert create_trade_idea_service().list_views(state=TradeIdeaState.PROPOSED) == []


@pytest.mark.asyncio
async def test_enabled_gate_proposes_buy_and_skips_execution(
    engine, mock_broker, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_proposals(engine, tmp_path, monkeypatch)

    validate = AsyncMock()
    submit = AsyncMock()
    monkeypatch.setattr(engine, "_validate_and_place_order", validate)
    monkeypatch.setattr(engine, "submit_order", submit)
    engine.strategy.active_strategies = "baseline"

    await engine._handle_decision(
        symbol="BTC-USD",
        decision=Decision(Action.BUY, "RSI reclaimed the long MA", 0.82),
        price=Decimal("50000"),
        equity=Decimal("1000"),
        position_state=None,
    )

    # Proposal-only: no execution path was taken, no broker order submitted.
    validate.assert_not_called()
    submit.assert_not_called()
    engine._order_submitter.submit_order_with_result.assert_not_called()
    mock_broker.place_order.assert_not_called()

    service = create_trade_idea_service()
    proposed = service.list_views(state=TradeIdeaState.PROPOSED)
    assert len(proposed) == 1
    view = proposed[0]
    assert view.state is TradeIdeaState.PROPOSED
    assert view.idea.instrument == "BTC-USD"
    assert view.idea.direction is TradeDirection.LONG
    # Enough review context was recorded: strategy, symbol, mark/as-of source.
    assert any(evidence.startswith("live-strategy:decision:") for evidence in view.idea.data_used)
    assert any(evidence.startswith("strategy:baseline:") for evidence in view.idea.data_used)

    # The proposal path never approves — no ApprovalPolicy bypass.
    assert service.list_views(state=TradeIdeaState.APPROVED) == []


@pytest.mark.asyncio
async def test_enabled_gate_refuses_cfm_context_without_submission(
    engine, mock_broker, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_proposals(engine, tmp_path, monkeypatch)
    engine.context.config.trading_modes = ["cfm"]
    engine.context.config.cfm_enabled = True
    engine.context.config.cfm_symbols = ["BTC-USD"]

    validate = AsyncMock()
    submit = AsyncMock()
    monkeypatch.setattr(engine, "_validate_and_place_order", validate)
    monkeypatch.setattr(engine, "submit_order", submit)

    await engine._handle_decision(
        symbol="BTC-USD",
        decision=Decision(Action.BUY, "futures setup", 0.82),
        price=Decimal("50000"),
        equity=Decimal("1000"),
        position_state={"symbol": "BTC-USD", "product_type": "FUTURE"},
    )

    validate.assert_not_called()
    submit.assert_not_called()
    mock_broker.place_order.assert_not_called()
    assert create_trade_idea_service().list_views(state=TradeIdeaState.PROPOSED) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("product_type", ["options", "unknown-contract"])
async def test_enabled_gate_refuses_declared_non_spot_product_type_without_submission(
    engine,
    mock_broker,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    product_type: str,
) -> None:
    _enable_proposals(engine, tmp_path, monkeypatch)

    validate = AsyncMock()
    submit = AsyncMock()
    monkeypatch.setattr(engine, "_validate_and_place_order", validate)
    monkeypatch.setattr(engine, "submit_order", submit)

    await engine._handle_decision(
        symbol="BTC-USD",
        decision=Decision(Action.BUY, "non-spot setup", 0.82),
        price=Decimal("50000"),
        equity=Decimal("1000"),
        position_state={"symbol": "BTC-USD", "product_type": product_type},
    )

    validate.assert_not_called()
    submit.assert_not_called()
    mock_broker.place_order.assert_not_called()
    assert create_trade_idea_service().list_views(state=TradeIdeaState.PROPOSED) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [Action.SELL, Action.CLOSE, Action.HOLD])
async def test_enabled_gate_never_submits_or_proposes_for_non_buy(
    engine, mock_broker, tmp_path, monkeypatch: pytest.MonkeyPatch, action: Action
) -> None:
    _enable_proposals(engine, tmp_path, monkeypatch)

    validate = AsyncMock()
    submit = AsyncMock()
    monkeypatch.setattr(engine, "_validate_and_place_order", validate)
    monkeypatch.setattr(engine, "submit_order", submit)

    await engine._handle_decision(
        symbol="BTC-USD",
        decision=Decision(action, "exit signal", 0.5),
        price=Decimal("50000"),
        equity=Decimal("1000"),
        position_state={"symbol": "BTC-USD", "side": "long", "quantity": Decimal("0.5")},
    )

    validate.assert_not_called()
    submit.assert_not_called()
    mock_broker.place_order.assert_not_called()
    # The adapter only maps buys, so no idea is proposed for these shapes.
    assert create_trade_idea_service().list_views(state=TradeIdeaState.PROPOSED) == []
