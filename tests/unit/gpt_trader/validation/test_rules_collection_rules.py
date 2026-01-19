import pytest

from gpt_trader.validation import ListRule, MappingRule, RuleError


class TestMappingRule:
    def test_parses_mapping_from_string(self) -> None:
        rule = MappingRule(value_converter=int)
        result = rule("btc:1,eth:2")
        assert result == {"btc": 1, "eth": 2}

    def test_invalid_entry_raises(self) -> None:
        rule = MappingRule(value_converter=float)
        with pytest.raises(RuleError):
            rule("btc=1")

    def test_disallows_blank_entries_when_requested(self) -> None:
        rule = MappingRule(value_converter=int, allow_blank_items=False)
        with pytest.raises(RuleError):
            rule("alpha:1,,beta:2", "my_map")


class TestListRule:
    def test_parses_string_list(self) -> None:
        rule = ListRule(item_converter=str)
        assert rule("alpha, beta ,gamma") == ["alpha", "beta", "gamma"]

    def test_disallows_empty_entries(self) -> None:
        rule = ListRule(item_converter=str, allow_blank_items=False)
        with pytest.raises(RuleError):
            rule("one,,two", "numbers")

    def test_cast_failure_surfaces(self) -> None:
        rule = ListRule(item_converter=int)
        with pytest.raises(RuleError):
            rule("1, two", "numbers")
