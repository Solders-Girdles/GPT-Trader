from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea

from gpt_trader.core import Candle
from gpt_trader.features.trade_ideas import (
    BaselineProposer,
    BaselineProposerConfig,
    EntryZone,
    MarketSnapshot,
    ReplayOutcome,
    ReplayRunnerConfig,
    ReplayScoringError,
    ScoringLevels,
    TimeHorizon,
    TradeDirection,
    TradeIdea,
    TradeIdeaReplayRunner,
    extract_numeric_scoring_levels,
    score_trade_idea,
)

AS_OF = datetime(2026, 6, 12, 12, 0, tzinfo=UTC)


def candle(
    offset_hours: int,
    *,
    open_: str = "101",
    high: str = "102",
    low: str = "100",
    close: str = "101",
) -> Candle:
    return Candle(
        ts=AS_OF + timedelta(hours=offset_hours),
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal("1000"),
    )


def scoreable_idea(**overrides: object) -> TradeIdea:
    fields = {
        "entry_zone": EntryZone(lower=Decimal("100"), upper=Decimal("102")),
        "invalidation": "Close below 95",
        "target_exit": "Take profit at 113 or exit at expiry",
        "time_horizon": TimeHorizon(
            expected_hold="1-4 hours",
            expires_at=AS_OF + timedelta(hours=4),
        ),
    }
    fields.update(overrides)
    return build_trade_idea(**fields)


def test_extract_numeric_scoring_levels_prefers_parenthesized_baseline_stop() -> None:
    idea = scoreable_idea(
        invalidation="Close below the 50-bar average (97.50)",
        target_exit="Take profit near 113.00 (2R) or exit at expiry",
    )

    assert extract_numeric_scoring_levels(idea) == ScoringLevels(
        entry_lower=Decimal("100"),
        entry_upper=Decimal("102"),
        stop=Decimal("97.50"),
        target=Decimal("113.00"),
    )


def test_score_trade_idea_records_target_hit() -> None:
    result = score_trade_idea(
        scoreable_idea(),
        as_of=AS_OF,
        future_candles=(
            candle(0, high="103", low="100", close="102"),
            candle(1, high="114", low="101", close="113"),
        ),
    )

    assert result.outcome is ReplayOutcome.TARGET_HIT
    assert result.entry_price == Decimal("101")
    assert result.exit_price == Decimal("113")
    assert result.return_r == Decimal("2")
    assert result.bars_evaluated == 2


def test_score_trade_idea_requires_midpoint_entry_fill() -> None:
    result = score_trade_idea(
        scoreable_idea(),
        as_of=AS_OF,
        future_candles=(
            candle(0, high="100.10", low="99.50", close="100"),
            candle(1, high="114", low="101", close="113"),
        ),
    )

    assert result.outcome is ReplayOutcome.TARGET_HIT
    assert result.entry_time == AS_OF + timedelta(hours=1)
    assert result.entry_price == Decimal("101")
    assert result.bars_evaluated == 2


def test_score_trade_idea_reports_not_filled_when_only_entry_zone_edge_trades() -> None:
    result = score_trade_idea(
        scoreable_idea(),
        as_of=AS_OF,
        future_candles=(
            candle(0, high="100.10", low="99.50", close="100"),
            candle(1, high="100.50", low="99.75", close="100.25"),
        ),
    )

    assert result.outcome is ReplayOutcome.NOT_FILLED
    assert result.entry_time is None
    assert result.entry_price is None


def test_score_trade_idea_uses_conservative_stop_first_when_bar_hits_both() -> None:
    result = score_trade_idea(
        scoreable_idea(),
        as_of=AS_OF,
        future_candles=(candle(0, high="114", low="94", close="101"),),
    )

    assert result.outcome is ReplayOutcome.STOP_HIT
    assert result.exit_price == Decimal("95")
    assert result.return_r == Decimal("-1")


def test_score_trade_idea_records_short_target_hit() -> None:
    result = score_trade_idea(
        scoreable_idea(
            direction=TradeDirection.SHORT,
            invalidation="Close above 106",
            target_exit="Take profit at 95 or exit at expiry",
        ),
        as_of=AS_OF,
        future_candles=(
            candle(0, high="102", low="100", close="101"),
            candle(1, high="101", low="94", close="95"),
        ),
    )

    assert result.outcome is ReplayOutcome.TARGET_HIT
    assert result.entry_price == Decimal("101")
    assert result.exit_price == Decimal("95")
    assert result.return_r == Decimal("1.2")


def test_score_trade_idea_uses_conservative_short_stop_first_when_bar_hits_both() -> None:
    result = score_trade_idea(
        scoreable_idea(
            direction=TradeDirection.SHORT,
            invalidation="Close above 106",
            target_exit="Take profit at 95 or exit at expiry",
        ),
        as_of=AS_OF,
        future_candles=(candle(0, high="107", low="94", close="101"),),
    )

    assert result.outcome is ReplayOutcome.STOP_HIT
    assert result.exit_price == Decimal("106")
    assert result.return_r == Decimal("-1")


def test_score_trade_idea_times_out_at_last_close_before_expiry() -> None:
    result = score_trade_idea(
        scoreable_idea(),
        as_of=AS_OF,
        future_candles=(
            candle(0, high="103", low="100", close="102"),
            candle(3, high="106", low="99", close="104"),
            candle(4, high="120", low="104", close="119"),
        ),
    )

    assert result.outcome is ReplayOutcome.TIMED_OUT
    assert result.exit_time == AS_OF + timedelta(hours=3)
    assert result.exit_price == Decimal("104")
    assert result.return_r == Decimal("0.5")


def test_score_trade_idea_counts_waiting_bars_before_delayed_entry() -> None:
    result = score_trade_idea(
        scoreable_idea(),
        as_of=AS_OF,
        future_candles=(
            candle(0, high="99", low="96", close="98"),
            candle(1, high="99.5", low="97", close="99"),
            candle(2, high="103", low="100", close="101"),
            candle(3, high="114", low="101", close="113"),
        ),
    )

    assert result.outcome is ReplayOutcome.TARGET_HIT
    assert result.entry_time == AS_OF + timedelta(hours=2)
    assert result.exit_time == AS_OF + timedelta(hours=3)
    assert result.bars_evaluated == 4


def test_score_trade_idea_reports_not_filled_when_entry_zone_never_trades() -> None:
    result = score_trade_idea(
        scoreable_idea(),
        as_of=AS_OF,
        future_candles=(
            candle(0, high="99", low="94", close="98"),
            candle(1, high="99.5", low="96", close="99"),
        ),
    )

    assert result.outcome is ReplayOutcome.NOT_FILLED
    assert result.entry_time is None
    assert result.return_r is None
    assert result.bars_evaluated == 2


def test_score_trade_idea_reports_no_future_data() -> None:
    result = score_trade_idea(scoreable_idea(), as_of=AS_OF, future_candles=())

    assert result.outcome is ReplayOutcome.NO_FUTURE_DATA
    assert result.bars_evaluated == 0


def test_replay_runner_feeds_point_in_time_snapshots_and_reports_aggregates() -> None:
    class ScriptedProposer:
        proposer_id = "scripted"

        def propose(self, snapshot: MarketSnapshot) -> list[TradeIdea]:
            series = snapshot.series_for("BTC-USD")
            assert series is not None
            assert all(item.ts < snapshot.as_of for item in series.candles)
            if len(series.candles) != 2:
                return []
            return [
                scoreable_idea(
                    decision_id=f"trade-{snapshot.as_of:%Y%m%d%H}",
                    time_horizon=TimeHorizon(
                        expected_hold="1-4 hours",
                        expires_at=snapshot.as_of + timedelta(hours=4),
                    ),
                )
            ]

    candles = (
        candle(-2, close="99", high="100", low="98"),
        candle(-1, close="101", high="102", low="100"),
        candle(0, close="102", high="103", low="100"),
        candle(1, close="113", high="114", low="101"),
    )

    report = TradeIdeaReplayRunner(
        ScriptedProposer(),
        config=ReplayRunnerConfig(min_history=2),
    ).run_series(symbol="BTC-USD", granularity="ONE_HOUR", candles=candles)

    assert report.snapshots_evaluated == 2
    assert report.ideas_proposed == 1
    assert report.target_hits == 1
    assert report.stop_hits == 0
    assert report.target_hit_rate == Decimal("1")
    assert report.average_return_r == Decimal("2")
    assert report.to_dict()["ideas"][0]["outcome"] == "target_hit"


def test_replay_runner_scores_baseline_proposer_on_historical_candles() -> None:
    candles = (
        candle(-5, close="100", high="100", low="100"),
        candle(-4, close="100", high="100", low="100"),
        candle(-3, close="100", high="100", low="100"),
        candle(-2, close="100", high="100", low="100"),
        candle(-1, close="110", high="110", low="110"),
        candle(0, open_="110", close="111", high="112", low="109"),
        candle(1, open_="111", close="126", high="126", low="111"),
    )

    report = TradeIdeaReplayRunner(
        BaselineProposer(
            BaselineProposerConfig(
                short_window=2,
                long_window=4,
                crossover_lookback=1,
                expiry_hours=3,
            )
        ),
        config=ReplayRunnerConfig(source="fixture:candles", min_history=5),
    ).run_series(symbol="BTC-USD", granularity="ONE_HOUR", candles=candles)

    assert report.ideas_proposed == 1
    assert report.target_hits == 1
    assert report.ideas[0].outcome is ReplayOutcome.TARGET_HIT
    assert report.ideas[0].levels.stop == Decimal("102.50")
    assert report.ideas[0].levels.target == Decimal("125.00")


def test_replay_runner_config_rejects_non_positive_min_history() -> None:
    try:
        ReplayRunnerConfig(min_history=0)
    except ReplayScoringError as exc:
        assert exc.context["field"] == "min_history"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("ReplayRunnerConfig should reject min_history=0")
