from __future__ import annotations

from urllib.error import URLError

import pytest

from gpt_trader.preflight import PreflightCheck


class StubClient:
    def __init__(self, responses: list[Exception | dict[str, object]]) -> None:
        self._responses = responses
        self.calls = 0

    def get_key_permissions(self) -> dict[str, object]:
        response = self._responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.parametrize("final_response", ["success", "fail"])
def test_key_permission_retry_behaviour(
    monkeypatch: pytest.MonkeyPatch, final_response: str
) -> None:
    if final_response == "success":
        responses = [
            URLError("temporary network issue"),
            URLError("rate limit"),
            {
                "can_trade": True,
                "can_view": True,
                "portfolio_type": "INTX",
                "portfolio_uuid": "uuid",
            },
        ]
        expected_result = True
    else:
        responses = [
            URLError("temporary network issue"),
            URLError("still bad"),
            URLError("third time unlucky"),
        ]
        expected_result = False

    client = StubClient(responses)
    checker = PreflightCheck(verbose=False, profile="dev")

    monkeypatch.setenv("COINBASE_ENABLE_DERIVATIVES", "1")
    monkeypatch.setenv("COINBASE_PREFLIGHT_FORCE_REMOTE", "1")
    monkeypatch.setattr(checker, "_build_cdp_client", lambda: (client, None))
    monkeypatch.setattr("gpt_trader.preflight.checks.connectivity.time.sleep", lambda _seconds: None)

    result = checker.check_key_permissions()

    assert result is expected_result
    assert client.calls == len(responses)
