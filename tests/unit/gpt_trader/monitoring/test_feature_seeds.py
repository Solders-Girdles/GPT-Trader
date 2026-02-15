from __future__ import annotations

import re

from gpt_trader.monitoring.feature_seeds import build_feature_seed, summarize_seed_reason

_MERGED_TITLE_HEX_SUFFIX = re.compile(r"^seed-[a-f0-9]+$")


def _merged_title_normalize(title: str) -> str:
    tokens = title.lower().split()
    normalized = [token for token in tokens if not _MERGED_TITLE_HEX_SUFFIX.match(token)]
    return " ".join(normalized)


def test_build_feature_seed_is_deterministic() -> None:
    seed_a = build_feature_seed(
        "Trade gate blocked: sizing",
        signature={"gate": "sizing", "reason": "quantity_zero"},
    )
    seed_b = build_feature_seed(
        "Trade gate blocked: sizing",
        signature={"gate": "sizing", "reason": "quantity_zero"},
    )

    assert seed_a == seed_b
    assert seed_a.key.startswith("trade-gate-blocked-sizing-")
    assert seed_a.title.startswith("Trade gate blocked: sizing seed-")
    assert seed_a.title.split("seed-")[-1]


def test_build_feature_seed_changes_with_signature() -> None:
    seed_a = build_feature_seed(
        "Trade gate blocked: sizing",
        signature={"gate": "sizing", "reason": "quantity_zero"},
    )
    seed_b = build_feature_seed(
        "Trade gate blocked: sizing",
        signature={"gate": "sizing", "reason": "min_notional"},
    )

    assert seed_a.key != seed_b.key
    assert seed_a.title != seed_b.title
    assert seed_a.title.split("seed-")[-1] == seed_a.key.rsplit("-", 1)[-1]
    assert seed_b.title.split("seed-")[-1] == seed_b.key.rsplit("-", 1)[-1]


def test_build_feature_seed_survives_merged_title_stripping() -> None:
    seed_a = build_feature_seed(
        "Trade gate blocked: sizing",
        signature={"gate": "sizing", "reason": "quantity_zero"},
    )
    seed_b = build_feature_seed(
        "Trade gate blocked: sizing",
        signature={"gate": "sizing", "reason": "min_notional"},
    )

    assert _merged_title_normalize(seed_a.title) != _merged_title_normalize(seed_b.title)


def test_build_feature_seed_title_avoids_bracket_suffixes() -> None:
    seed = build_feature_seed(
        "Latency spike",
        signature={"bucket": "warn"},
    )

    assert "[" not in seed.title
    assert "]" not in seed.title
    assert seed.title.endswith(f"seed-{seed.key.rsplit('-', 1)[-1]}")


def test_build_feature_seed_enforces_suffix_length_bounds() -> None:
    seed = build_feature_seed(
        "Latency spike",
        signature={"bucket": "warn"},
        suffix_length=2,
    )

    suffix = seed.key.rsplit("-", 1)[-1]
    assert len(suffix) == 4


def test_summarize_seed_reason_accepts_simple_codes() -> None:
    assert summarize_seed_reason("quantity_zero") == "quantity_zero"
    assert summarize_seed_reason("paused:mark_staleness") == "paused:mark_staleness"
    assert summarize_seed_reason("RISK_MANAGER_UNAVAILABLE") == "risk_manager_unavailable"


def test_summarize_seed_reason_rejects_noisy_values() -> None:
    assert summarize_seed_reason(None) is None
    assert summarize_seed_reason("") is None
    assert summarize_seed_reason("reason with spaces") is None
    assert summarize_seed_reason("details=price:123.45") is None
