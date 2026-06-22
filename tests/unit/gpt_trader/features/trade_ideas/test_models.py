from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea

from gpt_trader.features.trade_ideas import (
    BrokerTicket,
    MaxLoss,
    TicketStatus,
    TicketVenue,
    TimeHorizon,
    TradeIdea,
)


def test_round_trip_preserves_record(trade_idea: TradeIdea) -> None:
    restored = TradeIdea.from_dict(trade_idea.to_dict())

    assert restored == trade_idea


def test_to_dict_serializes_decimals_as_strings(trade_idea: TradeIdea) -> None:
    payload = trade_idea.to_dict()

    assert payload["max_loss"]["amount"] == "250"
    assert payload["entry_zone"]["lower"] == "60000"
    assert payload["sizing_recommendation"]["quantity"] == "0.1"


def test_from_dict_restores_decimal_types(trade_idea: TradeIdea) -> None:
    restored = TradeIdea.from_dict(trade_idea.to_dict())

    assert restored.max_loss.amount == Decimal("250")
    assert isinstance(restored.max_loss.amount, Decimal)
    assert restored.time_horizon.expires_at == trade_idea.time_horizon.expires_at


def test_time_horizon_rejects_timezone_naive_expiry(trade_idea: TradeIdea) -> None:
    with pytest.raises(ValueError, match="time_horizon.expires_at must include a timezone"):
        TimeHorizon(expires_at=datetime(2035, 6, 19, 16, 0))

    payload = trade_idea.to_dict()
    payload["time_horizon"]["expires_at"] = "2035-06-19T16:00:00"

    with pytest.raises(ValueError, match="time_horizon.expires_at must include a timezone"):
        TradeIdea.from_dict(payload)


def test_broker_ticket_defaults_to_no_venue(trade_idea: TradeIdea) -> None:
    assert trade_idea.broker_ticket == BrokerTicket(
        venue=TicketVenue.NONE, status=TicketStatus.NOT_CREATED
    )


def test_record_hash_is_stable(trade_idea: TradeIdea) -> None:
    assert trade_idea.record_hash() == trade_idea.record_hash()
    assert trade_idea.record_hash() == build_trade_idea().record_hash()


def test_record_hash_changes_when_content_changes(trade_idea: TradeIdea) -> None:
    amended = build_trade_idea(max_loss=MaxLoss(amount=Decimal("300")))

    assert amended.record_hash() != trade_idea.record_hash()


def test_optional_value_objects_round_trip_when_empty() -> None:
    idea = build_trade_idea(max_loss=MaxLoss(), do_not_trade_if=())
    restored = TradeIdea.from_dict(idea.to_dict())

    assert restored.max_loss.amount is None
    assert restored.max_loss.percent_of_account is None
    assert restored.do_not_trade_if == ()
