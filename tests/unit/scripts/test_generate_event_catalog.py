from __future__ import annotations

from scripts.agents.generate_event_catalog import scan_logging_calls


def test_exception_logging_counts_as_error_level(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "sample.py").write_text(
        """
def record_failure(logger):
    logger.exception(
        "Failed to extract mark from message",
        operation="extract_mark",
        error_type="InvalidOperation",
    )
""",
        encoding="utf-8",
    )

    results = scan_logging_calls(source_dir)

    event = results["operations"]["extract_mark"][0]
    assert event["level"] == "ERROR"
    assert results["levels"] == {"ERROR": 1}


def test_structured_logging_fields_do_not_depend_on_operation_position(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "sample.py").write_text(
        """
def record_close(logger, pos, quantity):
    logger.info(
        "Emergency close submitted",
        symbol=pos.symbol,
        quantity=str(quantity),
        reduce_only=True,
        bypass_reason="emergency_shutdown",
        operation="flatten_and_stop",
    )
""",
        encoding="utf-8",
    )

    results = scan_logging_calls(source_dir)

    event = results["operations"]["flatten_and_stop"][0]
    assert event["fields"] == [
        "bypass_reason",
        "quantity",
        "reduce_only",
        "symbol",
    ]
