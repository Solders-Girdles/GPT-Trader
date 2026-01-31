from __future__ import annotations

from pathlib import Path

import pytest

from scripts.maintenance import docs_link_audit, docs_reachability_check


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
