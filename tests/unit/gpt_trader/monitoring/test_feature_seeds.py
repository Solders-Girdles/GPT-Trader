from __future__ import annotations

from gpt_trader.monitoring.feature_seeds import build_feature_seed


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
