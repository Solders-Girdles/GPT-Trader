from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.core import Candle
from gpt_trader.features.trade_ideas import (
    BaselineProposer,
    BaselineProposerConfig,
    MarketSnapshot,
    Proposer,
    SymbolSeries,
    TradeDirection,
    evaluate_eligibility,
)

AS_OF = datetime(2026, 6, 12, 0, 0, tzinfo=UTC)
CONFIG = BaselineProposerConfig(short_window=5, long_window=20, crossover_lookback=3)


def make_series(
    closes: list[str], symbol: str = "BTC-USD", last_volume: str = "1000"
) -> SymbolSeries:
    candles = []
    for index, close in enumerate(closes):
        price = Decimal(close)
        volume = Decimal(last_volume) if index == len(closes) - 1 else Decimal("1000")
        candles.append(
            Candle(
                ts=AS_OF - timedelta(days=len(closes) - index),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
            )
        )
    return SymbolSeries(symbol=symbol, granularity="1d", candles=tuple(candles))


def snapshot_of(*series: SymbolSeries) -> MarketSnapshot:
    return MarketSnapshot(as_of=AS_OF, source="coinbase:candles", series=series)


GOLDEN_CROSS = ["100"] * 28 + ["102", "104"]
FLAT = ["100"] * 30
DOWNTREND = [str(130 - i) for i in range(30)]


def test_satisfies_proposer_protocol() -> None:
    assert isinstance(BaselineProposer(CONFIG), Proposer)


def test_fresh_golden_cross_produces_long_idea() -> None:
    ideas = BaselineProposer(CONFIG).propose(snapshot_of(make_series(GOLDEN_CROSS)))

    assert len(ideas) == 1
    idea = ideas[0]
    assert idea.direction is TradeDirection.LONG
    assert idea.instrument == "BTC-USD"
    assert evaluate_eligibility(idea) == []


def test_idea_records_are_complete_and_pinned() -> None:
    idea = BaselineProposer(CONFIG).propose(snapshot_of(make_series(GOLDEN_CROSS)))[0]

    assert idea.time_horizon.expires_at == AS_OF + timedelta(hours=CONFIG.expiry_hours)
    assert idea.data_used == (f"coinbase:candles:BTC-USD:1d:as_of={AS_OF.isoformat()}",)
    assert idea.max_loss.percent_of_account == CONFIG.risk_per_idea_pct
    assert idea.entry_zone.lower is not None
    assert idea.entry_zone.upper is not None
    assert idea.entry_zone.lower < idea.entry_zone.upper


def test_flat_market_produces_nothing() -> None:
    assert BaselineProposer(CONFIG).propose(snapshot_of(make_series(FLAT))) == []


def test_downtrend_produces_nothing() -> None:
    assert BaselineProposer(CONFIG).propose(snapshot_of(make_series(DOWNTREND))) == []


def test_stale_crossover_outside_lookback_produces_nothing() -> None:
    stale = ["100"] * 20 + [str(102 + 2 * i) for i in range(10)]

    assert BaselineProposer(CONFIG).propose(snapshot_of(make_series(stale))) == []


def test_insufficient_history_produces_nothing() -> None:
    short_history = ["100"] * 10 + ["102", "104"]

    assert BaselineProposer(CONFIG).propose(snapshot_of(make_series(short_history))) == []


def test_identical_snapshots_produce_identical_records() -> None:
    proposer = BaselineProposer(CONFIG)

    first = proposer.propose(snapshot_of(make_series(GOLDEN_CROSS)))[0]
    second = proposer.propose(snapshot_of(make_series(GOLDEN_CROSS)))[0]

    assert first.record_hash() == second.record_hash()
    assert first.decision_id == second.decision_id


def test_volume_confirmation_raises_confidence() -> None:
    confirmed = BaselineProposer(CONFIG).propose(
        snapshot_of(make_series(GOLDEN_CROSS, last_volume="5000"))
    )[0]
    unconfirmed = BaselineProposer(CONFIG).propose(
        snapshot_of(make_series(GOLDEN_CROSS, last_volume="1000"))
    )[0]

    assert confirmed.confidence.label.value == "medium"
    assert unconfirmed.confidence.label.value == "low"


def test_multi_symbol_snapshot_only_signals_where_warranted() -> None:
    snapshot = snapshot_of(
        make_series(GOLDEN_CROSS, symbol="BTC-USD"),
        make_series(FLAT, symbol="ETH-USD"),
    )

    ideas = BaselineProposer(CONFIG).propose(snapshot)

    assert [idea.instrument for idea in ideas] == ["BTC-USD"]
    assert "btcusd" in ideas[0].decision_id


@pytest.mark.parametrize("symbol", ["BTC-USD", "ETH-USD"])
def test_decision_ids_are_distinct_per_symbol(symbol: str) -> None:
    idea = BaselineProposer(CONFIG).propose(snapshot_of(make_series(GOLDEN_CROSS, symbol=symbol)))[
        0
    ]

    assert symbol.lower().replace("-", "") in idea.decision_id
