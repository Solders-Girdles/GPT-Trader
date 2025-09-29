from pathlib import Path
import json

from scripts.agents import naming_inventory


PATTERN = "qty"  # naming: allow
SUMMARY_LABEL = f"Pattern `{PATTERN}`"


def test_default_patterns_include_target():
    assert PATTERN in naming_inventory.DEFAULT_PATTERNS


def test_scan_detects_known_anti_pattern(tmp_path, monkeypatch):
    sample = tmp_path / "sample.py"
    sample.write_text(f"value = {PATTERN}  # test pattern\n")

    monkeypatch.setattr(naming_inventory, "REPO_ROOT", tmp_path)
    patterns = naming_inventory.compile_patterns([PATTERN])
    records = naming_inventory.scan(["."], patterns)

    assert any(rec.pattern == PATTERN for rec in records)

    summary = naming_inventory.format_summary(records, [PATTERN])
    assert SUMMARY_LABEL in summary

    json_path = tmp_path / "report.json"
    naming_inventory.write_json(json_path, records)
    data = json.loads(json_path.read_text())
    assert data[0]["pattern"] == PATTERN


def test_repo_scan_has_no_target_occurrences():
    patterns = naming_inventory.compile_patterns([PATTERN])
    records = naming_inventory.scan(["src"], patterns)
    assert not any(rec.pattern == PATTERN for rec in records)
