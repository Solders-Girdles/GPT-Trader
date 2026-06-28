from __future__ import annotations

from pathlib import Path

import pytest
from scripts.maintenance import (
    docs_link_audit,
    docs_reachability_check,
    generate_decision_index,
)


@pytest.mark.parametrize(
    "iter_links",
    [docs_link_audit.iter_links, docs_reachability_check.iter_links],
)
def test_iter_links_skips_external_strips_anchors_and_decodes(iter_links) -> None:
    content = """
    [External](https://example.com/path)
    [Email](mailto:test@example.com)
    [Phone](tel:+15551234567)
    [Anchor](#section)
    [Internal](docs/guide.md#intro)
    [Spaced](docs/Design%20Doc.md "Title")
    [Quoted]('docs/quoted.md')
    """

    assert iter_links(content) == [
        "docs/guide.md",
        "docs/Design Doc.md",
        "docs/quoted.md",
    ]


def test_iter_repo_path_references_filters_invalid_patterns() -> None:
    content = (
        "Valid: src/gpt_trader/app/main.py, scripts/maintenance/docs_link_audit.py; "
        "config/environments/.env.template. "
        "Invalid: src/gpt_trader/... and tests/*/test_*.py and src/gpt_trader/{foo}.py "
        "and src/gpt_trader/<bar>.py"
    )

    assert docs_link_audit.iter_repo_path_references(content) == [
        "src/gpt_trader/app/main.py",
        "scripts/maintenance/docs_link_audit.py",
        "config/environments/.env.template",
    ]


def test_docs_link_audit_skips_repo_path_checks_for_proposed_specs() -> None:
    root = Path("/repo")

    assert (
        docs_link_audit.should_check_repo_paths(
            root / "docs" / "specs" / "future_interface.md",
            root=root,
        )
        is False
    )
    assert (
        docs_link_audit.should_check_repo_paths(
            root / "docs" / "architecture" / "current_system.md",
            root=root,
        )
        is True
    )


def test_docs_link_audit_skips_review_artifact_tmp_markdown(tmp_path: Path) -> None:
    kept = tmp_path / "review_artifacts" / "durable.md"
    kept.parent.mkdir(parents=True)
    kept.write_text("# Durable\n", encoding="utf-8")
    skipped = tmp_path / "review_artifacts" / "tmp" / "draft.md"
    skipped.parent.mkdir(parents=True)
    skipped.write_text("[Broken](missing.md)\n", encoding="utf-8")

    assert docs_link_audit.iter_markdown_files(tmp_path) == [kept]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (Path("docs/architecture/system.md"), ["Core Documentation > Architecture & Design"]),
        (Path("docs/tui/overview.md"), ["Core Documentation > TUI"]),
        (Path("docs/agents/oncall.md"), ["Quick Links", "Getting Help"]),
        (Path("docs/misc/notes.md"), ["Quick Links", "Additional Resources"]),
    ],
)
def test_suggest_sections_for_representative_paths(path: Path, expected: list[str]) -> None:
    available_sections = {
        "Quick Links",
        "Getting Help",
        "Core Documentation > Architecture & Design",
        "Core Documentation > TUI",
        "Core Documentation > Trading Operations",
        "Core Documentation > Coinbase Integration",
        "Core Documentation > Development",
        "Core Documentation > Getting Started",
        "Configuration",
        "Additional Resources",
    }

    assert docs_reachability_check.suggest_sections(path, available_sections) == expected


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        # status alone is sufficient (date keys are optional)
        ("---\nstatus: current\n---", True),
        ("---\nstatus: current\nlast-updated: 2026-06-27\n---", True),
        # decision-record lifecycle statuses are accepted
        ("---\nstatus: proposed\n---", True),
        ("---\nstatus: accepted\n---", True),
        ("---\nstatus: rejected\n---", True),
        # status must be present and recognized
        ("---\nlast-updated: 2026-06-27\n---", False),
        ("---\nstatus: bogus\n---", False),
        # no metadata block at all
        ("(no frontmatter here)", False),
        # each optional date key is validated independently
        ("---\nstatus: current\nlast-reviewed: NOT-A-DATE\n---", False),
        ("---\nstatus: current\nlast-verified: NOT-A-DATE\n---", False),
        # a malformed date key must fail even when a later key is valid
        (
            "---\nstatus: current\nlast-updated: NOT-A-DATE\nlast-reviewed: 2026-06-27\n---",
            False,
        ),
    ],
)
def test_has_required_metadata(tmp_path: Path, block: str, expected: bool) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text(f"# Title\n\n{block}\n\nbody\n", encoding="utf-8")
    assert docs_reachability_check.has_required_metadata(doc) is expected


def _write_decision(directory: Path, slug: str, *, title: str, status: str, date: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / slug).write_text(
        f"# {title}\n\n---\nstatus: {status}\ndate: {date}\n---\n\nbody\n",
        encoding="utf-8",
    )


def test_collect_decisions_orders_newest_first_and_skips_non_records(tmp_path: Path) -> None:
    _write_decision(tmp_path, "older.md", title="Older", status="accepted", date="2026-01-01")
    _write_decision(tmp_path, "newer.md", title="Newer", status="proposed", date="2026-06-28")
    # README and underscore-prefixed template are not decision records.
    (tmp_path / "README.md").write_text(
        "# Decisions\n\n---\nstatus: current\n---\n", encoding="utf-8"
    )
    _write_decision(
        tmp_path, "_template.md", title="Template", status="proposed", date="2026-06-28"
    )

    records = generate_decision_index.collect_decisions(tmp_path)

    assert [r.slug for r in records] == ["newer.md", "older.md"]
    assert records[0].title == "Newer"
    assert records[0].status == "proposed"


def test_render_index_table_links_each_record(tmp_path: Path) -> None:
    _write_decision(
        tmp_path, "venue.md", title="Venue choice", status="proposed", date="2026-06-28"
    )
    records = generate_decision_index.collect_decisions(tmp_path)

    table = generate_decision_index.render_index_table(records)

    assert "| 2026-06-28 | [Venue choice](venue.md) | proposed |" in table


def test_splice_index_replaces_only_between_markers() -> None:
    readme = (
        "intro\n"
        f"{generate_decision_index.BEGIN_MARKER}\n"
        "OLD TABLE\n"
        f"{generate_decision_index.END_MARKER}\n"
        "outro\n"
    )

    spliced = generate_decision_index.splice_index(readme, "NEW TABLE")

    assert "OLD TABLE" not in spliced
    assert "NEW TABLE" in spliced
    assert spliced.startswith("intro\n")
    assert spliced.endswith("outro\n")


def test_splice_index_requires_markers() -> None:
    with pytest.raises(ValueError):
        generate_decision_index.splice_index("no markers here", "TABLE")


def test_generate_decision_index_check_detects_stale(tmp_path: Path) -> None:
    _write_decision(tmp_path, "alpha.md", title="Alpha", status="accepted", date="2026-02-02")
    (tmp_path / "README.md").write_text(
        "# Decisions\n\n---\nstatus: current\n---\n\n"
        f"{generate_decision_index.BEGIN_MARKER}\n"
        "| Date | Decision | Status |\n|------|----------|--------|\n"
        f"{generate_decision_index.END_MARKER}\n",
        encoding="utf-8",
    )

    assert generate_decision_index.main(["--decisions-dir", str(tmp_path), "--check"]) == 1
    assert generate_decision_index.main(["--decisions-dir", str(tmp_path)]) == 0
    assert generate_decision_index.main(["--decisions-dir", str(tmp_path), "--check"]) == 0
