from __future__ import annotations

import pytest

from gpt_trader.utilities.parsing import (
    FALSE_BOOLEAN_TOKENS,
    TRUE_BOOLEAN_TOKENS,
    interpret_tristate_bool,
)


@pytest.mark.parametrize("token", sorted(TRUE_BOOLEAN_TOKENS))
def test_interpret_tristate_bool_truthy_tokens(token: str) -> None:
    assert interpret_tristate_bool(token) is True
    assert interpret_tristate_bool(token.upper()) is True


@pytest.mark.parametrize("token", sorted(FALSE_BOOLEAN_TOKENS))
def test_interpret_tristate_bool_falsy_tokens(token: str) -> None:
    assert interpret_tristate_bool(token) is False
    assert interpret_tristate_bool(token.upper()) is False


def test_interpret_tristate_bool_handles_whitespace() -> None:
    assert interpret_tristate_bool("  yes  ") is True
    assert interpret_tristate_bool("  off  ") is False


def test_interpret_tristate_bool_returns_none_for_unknown_values() -> None:
    assert interpret_tristate_bool(None) is None
    assert interpret_tristate_bool("") is None
    assert interpret_tristate_bool("    ") is None
    assert interpret_tristate_bool("maybe") is None
