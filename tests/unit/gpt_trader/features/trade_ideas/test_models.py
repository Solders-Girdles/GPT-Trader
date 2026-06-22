from __future__ import annotations

import re
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


@pytest.mark.parametrize(
    ("field_path", "malformed_value", "message"),
    [
        ("data_used", "coinbase:candles:BTC-USD", "data_used must be a JSON array of strings"),
        ("data_used", 42, "data_used must be a JSON array of strings"),
        ("data_used", ["coinbase:candles:BTC-USD", 42], "data_used[1] must be a string"),
        (
            "do_not_trade_if",
            "FOMC announcement within 24 hours",
            "do_not_trade_if must be a JSON array of strings",
        ),
        (
            "do_not_trade_if",
            42,
            "do_not_trade_if must be a JSON array of strings",
        ),
        (
            "do_not_trade_if",
            ["FOMC announcement within 24 hours", 42],
            "do_not_trade_if[1] must be a string",
        ),
        (
            "max_loss.assumptions",
            "No slippage beyond 10 bps",
            "max_loss.assumptions must be a JSON array of strings",
        ),
        (
            "max_loss.assumptions",
            42,
            "max_loss.assumptions must be a JSON array of strings",
        ),
        (
            "max_loss.assumptions",
            ["No slippage beyond 10 bps", 42],
            "max_loss.assumptions[1] must be a string",
        ),
    ],
)
def test_from_dict_rejects_malformed_string_sequences(
    trade_idea: TradeIdea,
    field_path: str,
    malformed_value: object,
    message: str,
) -> None:
    payload = trade_idea.to_dict()
    if field_path == "max_loss.assumptions":
        payload["max_loss"]["assumptions"] = malformed_value
    else:
        payload[field_path] = malformed_value

    with pytest.raises(ValueError, match=re.escape(message)):
        TradeIdea.from_dict(payload)


@pytest.mark.parametrize(
    ("field_path", "malformed_value", "message"),
    [
        ("thesis", 42, "thesis must be a string"),
        ("instrument", 42, "instrument must be a string"),
        ("invalidation", 42, "invalidation must be a string"),
        ("target_exit", 42, "target_exit must be a string"),
        ("failure_mode", 42, "failure_mode must be a string"),
        ("entry_zone.trigger", 42, "entry_zone.trigger must be a string"),
        (
            "sizing_recommendation.rationale",
            42,
            "sizing_recommendation.rationale must be a string",
        ),
        ("time_horizon.expected_hold", 42, "time_horizon.expected_hold must be a string"),
        ("confidence.rationale", 42, "confidence.rationale must be a string"),
    ],
)
def test_from_dict_rejects_malformed_scalar_strings(
    trade_idea: TradeIdea,
    field_path: str,
    malformed_value: object,
    message: str,
) -> None:
    payload = trade_idea.to_dict()
    target = payload
    parts = field_path.split(".")
    for part in parts[:-1]:
        target = target[part]
        assert isinstance(target, dict)
    target[parts[-1]] = malformed_value

    with pytest.raises(ValueError, match=re.escape(message)):
        TradeIdea.from_dict(payload)
