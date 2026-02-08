from __future__ import annotations

from types import SimpleNamespace

from gpt_trader.ci import local_ci


def _make_args(profile: str) -> SimpleNamespace:
    return SimpleNamespace(
        include_snapshots=False,
        include_property_tests=False,
        include_contract_tests=False,
        include_agent_health=False,
        profile=profile,
    )


def _find_step(steps, label: str):
    return next(step for step in steps if step.label == label)


def test_profile_aliases_map_to_canonical_names() -> None:
    assert local_ci.resolve_profile("full").canonical_name == "strict"
    assert local_ci.resolve_profile("dev").canonical_name == "quick"


def test_quick_profile_skips_readiness_and_agent_artifacts() -> None:
    args = _make_args("quick")
    profile = local_ci.resolve_profile(args.profile)
    steps = local_ci.build_steps(profile, args)

    readiness_step = _find_step(steps, "Readiness gate (3-day streak)")
    assert readiness_step.enabled is False
    assert (
        "Use the strict profile when you need the readiness gate" in readiness_step.skip_reason
    )

    artifacts_step = _find_step(steps, "Agent artifacts freshness")
    assert artifacts_step.enabled is False
    assert (
        "Agent artifacts freshness is disabled in quick/dev" in artifacts_step.skip_reason
    )


def test_strict_profile_runs_readiness_and_agent_artifacts() -> None:
    args = _make_args("strict")
    profile = local_ci.resolve_profile(args.profile)
    steps = local_ci.build_steps(profile, args)

    readiness_step = _find_step(steps, "Readiness gate (3-day streak)")
    artifacts_step = _find_step(steps, "Agent artifacts freshness")

    assert readiness_step.enabled is True
    assert readiness_step.skip_reason is None
    assert artifacts_step.enabled is True
    assert artifacts_step.skip_reason is None


def test_print_profile_banner_reports_alias_and_status(capsys) -> None:
    selection = "dev"
    profile = local_ci.resolve_profile(selection)
    local_ci.print_profile_banner(selection, profile)

    output = capsys.readouterr().out
    assert "Local CI profile: quick (alias 'dev')" in output
    assert "Readiness gate: disabled" in output
    assert "Agent artifacts freshness: disabled" in output
