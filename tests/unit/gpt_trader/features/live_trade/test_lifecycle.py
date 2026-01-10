from __future__ import annotations

from unittest.mock import MagicMock

from gpt_trader.features.live_trade.lifecycle import (
    TRADING_BOT_TRANSITIONS,
    LifecycleStateMachine,
    TradingBotState,
)


def test_lifecycle_transition_audited() -> None:
    logger = MagicMock()
    machine = LifecycleStateMachine(
        initial_state=TradingBotState.INIT,
        entity="test_bot",
        transitions=TRADING_BOT_TRANSITIONS,
        logger=logger,
    )

    assert machine.transition(TradingBotState.STARTING, reason="test_start")
    assert machine.state == TradingBotState.STARTING
    logger.info.assert_called_once()


def test_lifecycle_transition_rejects_invalid() -> None:
    logger = MagicMock()
    machine = LifecycleStateMachine(
        initial_state=TradingBotState.INIT,
        entity="test_bot",
        transitions=TRADING_BOT_TRANSITIONS,
        logger=logger,
    )

    assert not machine.transition(TradingBotState.RUNNING, reason="invalid_jump")
    assert machine.state == TradingBotState.INIT
    logger.warning.assert_called_once()


def test_lifecycle_transition_force_override() -> None:
    logger = MagicMock()
    machine = LifecycleStateMachine(
        initial_state=TradingBotState.INIT,
        entity="test_bot",
        transitions=TRADING_BOT_TRANSITIONS,
        logger=logger,
    )

    assert machine.transition(
        TradingBotState.RUNNING,
        reason="manual_override",
        force=True,
    )
    assert machine.state == TradingBotState.RUNNING
