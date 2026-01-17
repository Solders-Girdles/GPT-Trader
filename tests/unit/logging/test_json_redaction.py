import json
import logging
import unittest

from gpt_trader.logging.json_formatter import StructuredJSONFormatter


class TestJSONRedaction(unittest.TestCase):
    def test_redaction(self):
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

        # Add sensitive data to extra fields
        record.extra = {
            "api_key": "secret_key_value",
            "apiKey": "camel_key",
            "privateKey": "camel_private",
            "nested": {"private_key": "secret_private_key", "public_data": "visible"},
            "list_data": [{"token": "secret_token"}, {"other": "visible"}],
        }

        json_output = formatter.format(record)
        data = json.loads(json_output)

        # Verify redaction
        self.assertEqual(data["api_key"], "[REDACTED]")
        self.assertEqual(data["apiKey"], "[REDACTED]")
        self.assertEqual(data["privateKey"], "[REDACTED]")
        self.assertEqual(data["nested"]["private_key"], "[REDACTED]")
        self.assertEqual(data["nested"]["public_data"], "visible")
        self.assertEqual(data["list_data"][0]["token"], "[REDACTED]")
        self.assertEqual(data["list_data"][1]["other"], "visible")


if __name__ == "__main__":
    unittest.main()
