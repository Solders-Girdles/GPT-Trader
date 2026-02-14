"""Tests for preflight check graph assembly."""

from __future__ import annotations

import pytest

from gpt_trader.preflight.check_graph import (
    CORE_PREFLIGHT_CHECKS,
    PreflightCheckGraphError,
    PreflightCheckNode,
    assemble_preflight_check_graph,
)


def test_core_graph_composes_expected_order() -> None:
    ordered = assemble_preflight_check_graph(CORE_PREFLIGHT_CHECKS)
    ordered_names = [node.name for node in ordered]

    assert ordered_names == [node.name for node in CORE_PREFLIGHT_CHECKS]


def test_graph_missing_dependency_errors_deterministically() -> None:
    nodes = [
        PreflightCheckNode("check_alpha", dependencies=("check_missing_b",)),
        PreflightCheckNode("check_beta", dependencies=("check_missing_a",)),
    ]

    with pytest.raises(PreflightCheckGraphError) as excinfo:
        assemble_preflight_check_graph(nodes)

    assert str(excinfo.value) == (
        "Missing preflight check dependencies: "
        "check_alpha -> check_missing_b, check_beta -> check_missing_a"
    )


def test_graph_order_is_stable_for_independent_nodes() -> None:
    nodes = [
        PreflightCheckNode("check_alpha"),
        PreflightCheckNode("check_beta"),
        PreflightCheckNode("check_gamma", dependencies=("check_alpha",)),
    ]

    ordered = assemble_preflight_check_graph(nodes)
    ordered_names = [node.name for node in ordered]

    assert ordered_names == ["check_alpha", "check_beta", "check_gamma"]
