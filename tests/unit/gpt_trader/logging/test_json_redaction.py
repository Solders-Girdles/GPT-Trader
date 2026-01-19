import json
import logging

from gpt_trader.logging.json_formatter import StructuredJSONFormatter


def test_json_redaction() -> None:
    formatter = StructuredJSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    record.extra = {
        "api_key": "secret_key_value",
        "apiKey": "camel_key",
        "privateKey": "camel_private",
        "nested": {"private_key": "secret_private_key", "public_data": "visible"},
        "list_data": [{"token": "secret_token"}, {"other": "visible"}],
    }

    json_output = formatter.format(record)
    data = json.loads(json_output)

    assert data["api_key"] == "[REDACTED]"
    assert data["apiKey"] == "[REDACTED]"
    assert data["privateKey"] == "[REDACTED]"
    assert data["nested"]["private_key"] == "[REDACTED]"
    assert data["nested"]["public_data"] == "visible"
    assert data["list_data"][0]["token"] == "[REDACTED]"
    assert data["list_data"][1]["other"] == "visible"
