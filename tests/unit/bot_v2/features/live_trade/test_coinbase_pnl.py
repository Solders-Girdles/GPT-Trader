"""Comprehensive Coinbase PnL behavior coverage."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import Mock, patch

import pytest
from tests.unit.bot_v2.features.live_trade.pnl_test_utils import (
    TradeOp,
    apply_marks,
    apply_trades,
    ensure_advanced_pnl_available,
    make_position,
)

from bot_v2.features.brokerages.coinbase.models import to_position
from bot_v2.features.live_trade.pnl_tracker import FundingCalculator, PnLTracker

ensure_advanced_pnl_available()


# --- Tracker fills and realized/unrealized behaviour -----------------------------------------

ROUNDING_TRADES = [
    (Decimal("0.033"), Decimal("50000.33")),
    (Decimal("0.033"), Decimal("50001.67")),
    (Decimal("0.034"), Decimal("49998.50")),
]


def test_multiple_fills_compute_weighted_average_price() -> None:
    tracker = PnLTracker()
    trades = [
        TradeOp(symbol="BTC-PERP", side="buy", size=Decimal("0.05"), price=Decimal("50000")),
        TradeOp(symbol="BTC-PERP", side="buy", size=Decimal("0.05"), price=Decimal("51000")),
    ]
    apply_trades(tracker, trades)

    position = tracker.positions["BTC-PERP"]
    assert position.quantity == Decimal("0.1")
    assert position.avg_entry_price == Decimal("50500")


def test_partial_close_realizes_expected_pnl() -> None:
    tracker = PnLTracker()
    trades = [
        TradeOp(symbol="BTC-PERP", side="buy", size=Decimal("0.1"), price=Decimal("50000")),
        TradeOp(
            symbol="BTC-PERP",
            side="sell",
            size=Decimal("0.05"),
            price=Decimal("52000"),
            is_reduce=True,
        ),
    ]
    results = apply_trades(tracker, trades)
    assert results[1]["realized_pnl"] == Decimal("100")

    position = tracker.positions["BTC-PERP"]
    assert position.quantity == Decimal("0.05")


def test_flip_from_long_to_short_tracks_new_entry() -> None:
    tracker = PnLTracker()
    trades = [
        TradeOp(symbol="BTC-PERP", side="buy", size=Decimal("0.1"), price=Decimal("50000")),
        TradeOp(symbol="BTC-PERP", side="sell", size=Decimal("0.2"), price=Decimal("51000")),
    ]
    results = apply_trades(tracker, trades)
    assert results[1]["realized_pnl"] == Decimal("100")

    position = tracker.positions["BTC-PERP"]
    assert position.side == "short"
    assert position.quantity == Decimal("0.1")
    assert position.avg_entry_price == Decimal("51000")


def test_multiple_products_unrealized_pnl() -> None:
    tracker = PnLTracker()
    apply_trades(
        tracker,
        [
            TradeOp(symbol="BTC-PERP", side="buy", size=Decimal("0.1"), price=Decimal("50000")),
            TradeOp(symbol="ETH-PERP", side="sell", size=Decimal("1.0"), price=Decimal("3000")),
        ],
    )

    apply_marks(tracker, {"BTC-PERP": Decimal("51000"), "ETH-PERP": Decimal("2900")})
    assert tracker.positions["BTC-PERP"].unrealized_pnl == Decimal("100")
    assert tracker.positions["ETH-PERP"].unrealized_pnl == Decimal("100")

    total = tracker.get_total_pnl()
    assert total["unrealized"] == Decimal("200")


def test_weighted_average_rounding_consistency() -> None:
    tracker = PnLTracker()
    apply_trades(
        tracker,
        [
            TradeOp(symbol="BTC-PERP", side="buy", size=size, price=price)
            for size, price in ROUNDING_TRADES
        ],
    )

    position = tracker.positions["BTC-PERP"]
    assert position.quantity == Decimal("0.1")

    total_value = sum(size * price for size, price in ROUNDING_TRADES)
    expected_avg = total_value / Decimal("0.1")
    assert abs(position.avg_entry_price - expected_avg) < Decimal("0.01")


# --- Coinbase parity checks --------------------------------------------------------------------

PNL_MATCH_CASES = [
    pytest.param(
        {
            "id": "single-long",
            "trades": [
                TradeOp(
                    symbol="BTC-PERP",
                    side="buy",
                    size=Decimal("0.01"),
                    price=Decimal("95000"),
                )
            ],
            "marks": {"BTC-PERP": Decimal("96000")},
            "coinbase": {
                "product_id": "BTC-PERP",
                "size": Decimal("0.01"),
                "side": "long",
                "entry_price": Decimal("95000"),
                "mark_price": Decimal("96000"),
                "unrealized_pnl": Decimal("10.00"),
                "realized_pnl": Decimal("0.00"),
            },
        },
        id="single-long",
    ),
    pytest.param(
        {
            "id": "multi-fill",
            "trades": [
                TradeOp(
                    symbol="BTC-PERP", side="buy", size=Decimal("0.01"), price=Decimal("95000")
                ),
                TradeOp(
                    symbol="BTC-PERP", side="buy", size=Decimal("0.015"), price=Decimal("95100")
                ),
                TradeOp(
                    symbol="BTC-PERP", side="buy", size=Decimal("0.02"), price=Decimal("94950")
                ),
            ],
            "marks": {"BTC-PERP": Decimal("96000")},
            "coinbase": {
                "product_id": "BTC-PERP",
                "size": Decimal("0.045"),
                "side": "long",
                "entry_price": Decimal("95011.1111111111"),
                "mark_price": Decimal("96000"),
                "realized_pnl": Decimal("0.00"),
            },
        },
        id="multi-fill",
    ),
]


@pytest.mark.parametrize("case", PNL_MATCH_CASES, ids=lambda c: c["id"])
def test_tracker_matches_coinbase(case: dict[str, Any]) -> None:
    tracker = PnLTracker()
    apply_trades(tracker, case["trades"])
    apply_marks(tracker, case["marks"])

    symbol = case["trades"][0].symbol
    our_position = tracker.positions[symbol]

    coinbase_payload = {
        "product_id": case["coinbase"]["product_id"],
        "size": str(case["coinbase"]["size"]),
        "side": case["coinbase"].get("side", "long"),
        "entry_price": str(case["coinbase"]["entry_price"]),
        "mark_price": str(case["coinbase"]["mark_price"]),
        "unrealized_pnl": str(
            case["coinbase"].get(
                "unrealized_pnl",
                (case["coinbase"]["mark_price"] - our_position.avg_entry_price)
                * case["coinbase"]["size"],
            )
        ),
        "realized_pnl": str(case["coinbase"].get("realized_pnl", Decimal("0"))),
        "leverage": 1,
    }
    cb_position = to_position(coinbase_payload)

    assert abs(our_position.avg_entry_price - cb_position.entry_price) < Decimal("0.01")
    assert abs(our_position.unrealized_pnl - cb_position.unrealized_pnl) < Decimal("0.01")
    assert our_position.realized_pnl == cb_position.realized_pnl


def test_partial_close_matches_coinbase() -> None:
    tracker = PnLTracker()
    trades = [
        TradeOp(symbol="BTC-PERP", side="buy", size=Decimal("0.1"), price=Decimal("95000")),
        TradeOp(
            symbol="BTC-PERP",
            side="sell",
            size=Decimal("0.03"),
            price=Decimal("96000"),
            is_reduce=True,
        ),
    ]
    results = apply_trades(tracker, trades)
    assert results[1]["realized_pnl"] == Decimal("30")

    apply_marks(tracker, {"BTC-PERP": Decimal("96000")})

    coinbase_payload = {
        "product_id": "BTC-PERP",
        "size": "0.07",
        "side": "long",
        "entry_price": "95000",
        "mark_price": "96000",
        "unrealized_pnl": "70.00",
        "realized_pnl": "30.00",
        "leverage": 1,
    }
    cb_position = to_position(coinbase_payload)
    our_position = tracker.positions["BTC-PERP"]

    assert our_position.realized_pnl == cb_position.realized_pnl
    assert our_position.unrealized_pnl == cb_position.unrealized_pnl


def test_funding_adjusts_total_pnl() -> None:
    tracker = PnLTracker()
    apply_trades(
        tracker,
        [TradeOp(symbol="BTC-PERP", side="buy", size=Decimal("0.1"), price=Decimal("95000"))],
    )

    position = tracker.positions["BTC-PERP"]
    funding_payment = Decimal("0.95")
    position.funding_paid = funding_payment

    coinbase_payload = {
        "product_id": "BTC-PERP",
        "size": "0.1",
        "side": "long",
        "entry_price": "95000",
        "mark_price": "95000",
        "unrealized_pnl": "0.00",
        "realized_pnl": "-0.95",
        "leverage": 1,
    }
    cb_position = to_position(coinbase_payload)

    totals = tracker.get_total_pnl()
    cb_total = cb_position.unrealized_pnl + cb_position.realized_pnl
    our_total = totals["unrealized"] + totals["realized"] - totals["funding"]

    assert abs(our_total - cb_total) < Decimal("0.01")


DISCREPANCY_CASES = [
    pytest.param(
        {
            "ours": Decimal("100.01"),
            "coinbase": Decimal("100.00"),
            "needs_investigation": False,
        },
        id="rounding",
    ),
    pytest.param(
        {
            "ours": Decimal("105"),
            "coinbase": Decimal("100"),
            "needs_investigation": True,
            "pct": Decimal("5"),
        },
        id="large-diff",
    ),
]


def _calculate_discrepancy(ours: Decimal, coinbase: Decimal) -> dict[str, Any]:
    diff = ours - coinbase
    pct = (diff / coinbase * Decimal("100")) if coinbase else Decimal("0")
    return {
        "absolute_diff": diff,
        "percentage_diff": pct,
        "needs_investigation": abs(diff) > Decimal("1") or abs(pct) > Decimal("1"),
    }


@pytest.mark.parametrize(
    "case",
    DISCREPANCY_CASES,
    ids=lambda c: c["needs_investigation"] and "investigate" or "normal",
)
def test_discrepancy_detection(case: dict[str, Any]) -> None:
    result = _calculate_discrepancy(case["ours"], case["coinbase"])
    assert result["needs_investigation"] is case["needs_investigation"]
    if "pct" in case:
        assert result["percentage_diff"] == case["pct"]


def test_funding_adjustment_consistency() -> None:
    tracker = PnLTracker()
    apply_trades(
        tracker,
        [TradeOp(symbol="BTC-PERP", side="buy", size=Decimal("0.1"), price=Decimal("95000"))],
    )
    position = tracker.positions["BTC-PERP"]
    position.funding_paid = Decimal("5")

    coinbase_payload = {
        "product_id": "BTC-PERP",
        "size": "0.1",
        "side": "long",
        "entry_price": "95000",
        "mark_price": "95000",
        "unrealized_pnl": "0.00",
        "realized_pnl": "-5.00",
    }
    cb_position = to_position(coinbase_payload)

    our_economic = position.realized_pnl - position.funding_paid
    assert our_economic == cb_position.realized_pnl


# --- Position state behaviour -----------------------------------------------------------------

POSITION_CASES = [
    pytest.param(
        {
            "id": "long-profit",
            "side": "long",
            "entry": Decimal("50000"),
            "quantity": Decimal("0.1"),
            "mark": Decimal("52000"),
            "expected": Decimal("200"),
        },
        id="long-profit",
    ),
    pytest.param(
        {
            "id": "long-loss",
            "side": "long",
            "entry": Decimal("50000"),
            "quantity": Decimal("0.1"),
            "mark": Decimal("48000"),
            "expected": Decimal("-200"),
        },
        id="long-loss",
    ),
    pytest.param(
        {
            "id": "short-profit",
            "side": "short",
            "entry": Decimal("50000"),
            "quantity": Decimal("0.1"),
            "mark": Decimal("48000"),
            "expected": Decimal("200"),
        },
        id="short-profit",
    ),
    pytest.param(
        {
            "id": "short-loss",
            "side": "short",
            "entry": Decimal("50000"),
            "quantity": Decimal("0.1"),
            "mark": Decimal("52000"),
            "expected": Decimal("-200"),
        },
        id="short-loss",
    ),
]


@pytest.mark.parametrize("case", POSITION_CASES, ids=lambda c: c["id"])
def test_unrealized_pnl_matches_expectations(case: dict[str, Any]) -> None:
    position = make_position(
        product_id="BTC-PERP",
        side=case["side"],
        entry_price=case["entry"],
        size=case["quantity"],
        timestamp=datetime.now(),
    )

    unrealized = position.update_mark(case["mark"])
    assert unrealized == case["expected"]


def test_zero_quantity_position_has_no_pnl() -> None:
    position = make_position(
        product_id="BTC-PERP",
        side="long",
        entry_price=Decimal("50000"),
        size=Decimal("0"),
        timestamp=datetime.now(),
    )

    unrealized = position.update_mark(Decimal("60000"))
    assert unrealized == Decimal("0")


def test_breakeven_price_returns_zero_pnl() -> None:
    position = make_position(
        product_id="BTC-PERP",
        side="long",
        entry_price=Decimal("50000"),
        size=Decimal("0.1"),
        timestamp=datetime.now(),
    )

    unrealized = position.update_mark(Decimal("50000"))
    assert unrealized == Decimal("0")


def test_small_position_precision() -> None:
    position = make_position(
        product_id="BTC-PERP",
        side="long",
        entry_price=Decimal("50000"),
        size=Decimal("0.001"),
        timestamp=datetime.now(),
    )

    unrealized = position.update_mark(Decimal("50010"))
    assert unrealized == Decimal("0.01")


# --- Reconciliation workflows -----------------------------------------------------------------


def _generate_reconciliation_report(
    tracker: PnLTracker, coinbase_positions: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    report = {
        "timestamp": "2024-01-01T00:00:00Z",
        "positions_compared": 0,
        "discrepancies": [],
        "summary": {
            "total_our_pnl": Decimal("0"),
            "total_cb_pnl": Decimal("0"),
            "total_difference": Decimal("0"),
        },
    }

    for payload in coinbase_positions:
        symbol = payload["product_id"]
        cb_pos = to_position(payload)
        our_pos = tracker.positions.get(symbol)
        if our_pos is None:
            continue

        report["positions_compared"] += 1

        unrealized_diff = our_pos.unrealized_pnl - cb_pos.unrealized_pnl
        if abs(unrealized_diff) > Decimal("0.01"):
            report["discrepancies"].append(
                {
                    "symbol": symbol,
                    "type": "unrealized_pnl",
                    "our_value": float(our_pos.unrealized_pnl),
                    "cb_value": float(cb_pos.unrealized_pnl),
                    "difference": float(unrealized_diff),
                }
            )

        realized_diff = our_pos.realized_pnl - cb_pos.realized_pnl
        if abs(realized_diff) > Decimal("0.01"):
            report["discrepancies"].append(
                {
                    "symbol": symbol,
                    "type": "realized_pnl",
                    "our_value": float(our_pos.realized_pnl),
                    "cb_value": float(cb_pos.realized_pnl),
                    "difference": float(realized_diff),
                }
            )

        our_total = our_pos.unrealized_pnl + our_pos.realized_pnl
        cb_total = cb_pos.unrealized_pnl + cb_pos.realized_pnl
        report["summary"]["total_our_pnl"] += our_total
        report["summary"]["total_cb_pnl"] += cb_total

    report["summary"]["total_difference"] = (
        report["summary"]["total_our_pnl"] - report["summary"]["total_cb_pnl"]
    )
    report["summary"] = {k: float(v) for k, v in report["summary"].items()}
    return report


@patch("bot_v2.features.brokerages.coinbase.adapter.CoinbaseBrokerage")
def test_live_position_reconciliation(mock_adapter) -> None:
    adapter = mock_adapter.return_value
    adapter.list_positions.return_value = [
        Mock(
            symbol="BTC-PERP",
            quantity=Decimal("0.05"),
            entry_price=Decimal("94000"),
            mark_price=Decimal("96000"),
            unrealized_pnl=Decimal("100"),
            realized_pnl=Decimal("50"),
            side="long",
        ),
        Mock(
            symbol="ETH-PERP",
            quantity=Decimal("0.5"),
            entry_price=Decimal("3300"),
            mark_price=Decimal("3250"),
            unrealized_pnl=Decimal("25"),
            realized_pnl=Decimal("10"),
            side="short",
        ),
    ]

    tracker = PnLTracker()
    apply_trades(
        tracker,
        [
            TradeOp(symbol="BTC-PERP", side="buy", size=Decimal("0.05"), price=Decimal("94000")),
            TradeOp(symbol="ETH-PERP", side="sell", size=Decimal("0.5"), price=Decimal("3300")),
        ],
    )
    apply_marks(tracker, {"BTC-PERP": Decimal("96000"), "ETH-PERP": Decimal("3250")})

    tracker.positions["BTC-PERP"].realized_pnl = Decimal("50")
    tracker.positions["ETH-PERP"].realized_pnl = Decimal("10")

    discrepancies = []
    for cb_pos in adapter.list_positions():
        our_pos = tracker.positions.get(cb_pos.symbol)
        if our_pos is None:
            continue
        if abs(our_pos.unrealized_pnl - cb_pos.unrealized_pnl) > Decimal("0.01"):
            discrepancies.append(cb_pos.symbol)
        if abs(our_pos.realized_pnl - cb_pos.realized_pnl) > Decimal("0.01"):
            discrepancies.append(cb_pos.symbol)

    assert discrepancies == []


def test_reconciliation_report_generation() -> None:
    tracker = PnLTracker()
    apply_trades(
        tracker,
        [TradeOp(symbol="BTC-PERP", side="buy", size=Decimal("0.1"), price=Decimal("95000"))],
    )
    apply_marks(tracker, {"BTC-PERP": Decimal("96000")})

    report = _generate_reconciliation_report(
        tracker,
        [
            {
                "product_id": "BTC-PERP",
                "size": "0.1",
                "side": "long",
                "entry_price": "95000",
                "mark_price": "96000",
                "unrealized_pnl": "100.00",
                "realized_pnl": "0.00",
            }
        ],
    )

    assert report["positions_compared"] == 1
    assert report["discrepancies"] == []
    assert report["summary"]["total_difference"] == 0.0


def test_reconciliation_with_known_differences() -> None:
    tracker = PnLTracker()
    apply_trades(
        tracker,
        [TradeOp(symbol="BTC-PERP", side="buy", size=Decimal("0.1"), price=Decimal("95000"))],
    )
    tracker.positions["BTC-PERP"].funding_paid = Decimal("5")

    cb_position = to_position(
        {
            "product_id": "BTC-PERP",
            "size": "0.1",
            "side": "long",
            "entry_price": "95000",
            "mark_price": "95000",
            "unrealized_pnl": "0.00",
            "realized_pnl": "-5.00",
        }
    )

    our_economic = (
        tracker.positions["BTC-PERP"].realized_pnl - tracker.positions["BTC-PERP"].funding_paid
    )
    assert our_economic == cb_position.realized_pnl


# --- Funding mechanics ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "side, rate, expected",
    [
        pytest.param("long", Decimal("0.01"), Decimal("-100"), id="long-pays"),
        pytest.param("short", Decimal("0.01"), Decimal("100"), id="short-receives"),
        pytest.param("long", Decimal("-0.01"), Decimal("100"), id="long-receives"),
        pytest.param("short", Decimal("-0.01"), Decimal("-100"), id="short-pays"),
    ],
)
def test_funding_payment_signs(side: str, rate: Decimal, expected: Decimal) -> None:
    calculator = FundingCalculator()
    payment = calculator.calculate_funding(
        position_size=Decimal("0.2"),
        side=side,
        mark_price=Decimal("50000"),
        funding_rate=rate,
    )
    assert payment == expected


def test_cumulative_funding_tracking() -> None:
    tracker = PnLTracker()
    tracker.update_position("BTC-PERP", "buy", Decimal("0.1"), Decimal("50000"))

    tracker.positions["BTC-PERP"].last_funding_time = datetime.now() - timedelta(hours=9)

    tracker.accrue_funding("BTC-PERP", Decimal("50000"), Decimal("0.01"))
    tracker.positions["BTC-PERP"].last_funding_time -= timedelta(hours=9)

    tracker.accrue_funding("BTC-PERP", Decimal("50000"), Decimal("-0.005"))

    position = tracker.positions["BTC-PERP"]
    assert position.funding_paid == Decimal("25")


def test_accrue_if_due_respects_interval() -> None:
    calculator = FundingCalculator()
    now = datetime.now()
    seven_hours = now + timedelta(hours=7)
    eight_hours = now + timedelta(hours=8)

    assert calculator.is_funding_due(now, None, seven_hours) is False
    assert calculator.is_funding_due(now, None, eight_hours) is True
