"""Tests for CLI argument groups and registrar."""

import argparse
from decimal import Decimal

import pytest

from bot_v2.cli.argument_groups import (
    ACCOUNT_ARGS,
    BOT_CONFIG_ARGS,
    CONVERT_ARGS,
    DEV_ARGS,
    MOVE_FUNDS_ARGS,
    ORDER_TOOLING_ARGS,
    ArgumentGroupRegistrar,
    ArgumentSpec,
)


# ============================================================================
# Test: ArgumentSpec
# ============================================================================


class TestArgumentSpec:
    """Test ArgumentSpec dataclass."""

    def test_spec_with_action(self):
        """Test spec with action creates store_true argument."""
        parser = argparse.ArgumentParser()
        spec = ArgumentSpec(name="--test-flag", action="store_true", help="Test flag")

        spec.add_to_parser(parser)
        args = parser.parse_args(["--test-flag"])

        assert args.test_flag is True

    def test_spec_with_type_and_default(self):
        """Test spec with type and default."""
        parser = argparse.ArgumentParser()
        spec = ArgumentSpec(name="--test-int", type=int, default=42, help="Test int")

        spec.add_to_parser(parser)
        args = parser.parse_args([])

        assert args.test_int == 42

    def test_spec_with_choices(self):
        """Test spec with choices validation."""
        parser = argparse.ArgumentParser()
        spec = ArgumentSpec(
            name="--test-choice", choices=["a", "b", "c"], default="a", help="Test choice"
        )

        spec.add_to_parser(parser)
        args = parser.parse_args(["--test-choice", "b"])

        assert args.test_choice == "b"

    def test_spec_with_dest(self):
        """Test spec with custom destination."""
        parser = argparse.ArgumentParser()
        spec = ArgumentSpec(name="--test-name", dest="custom_dest", type=str, help="Test dest")

        spec.add_to_parser(parser)
        args = parser.parse_args(["--test-name", "value"])

        assert args.custom_dest == "value"

    def test_spec_with_nargs_plus(self):
        """Test spec with nargs='+'."""
        parser = argparse.ArgumentParser()
        spec = ArgumentSpec(name="--test-multi", type=str, nargs="+", help="Test multi")

        spec.add_to_parser(parser)
        args = parser.parse_args(["--test-multi", "a", "b", "c"])

        assert args.test_multi == ["a", "b", "c"]


# ============================================================================
# Test: Argument Groups
# ============================================================================


class TestArgumentGroups:
    """Test argument group definitions."""

    def test_bot_config_args_count(self):
        """Test bot config args has expected count."""
        assert len(BOT_CONFIG_ARGS) == 12

    def test_account_args_count(self):
        """Test account args has expected count."""
        assert len(ACCOUNT_ARGS) == 1

    def test_convert_args_count(self):
        """Test convert args has expected count."""
        assert len(CONVERT_ARGS) == 1

    def test_move_funds_args_count(self):
        """Test move funds args has expected count."""
        assert len(MOVE_FUNDS_ARGS) == 1

    def test_order_tooling_args_count(self):
        """Test order tooling args has expected count."""
        assert len(ORDER_TOOLING_ARGS) == 13

    def test_dev_args_count(self):
        """Test dev args has expected count."""
        assert len(DEV_ARGS) == 1


# ============================================================================
# Test: ArgumentGroupRegistrar
# ============================================================================


class TestArgumentGroupRegistrar:
    """Test ArgumentGroupRegistrar."""

    def test_register_bot_config(self):
        """Test registering bot config arguments."""
        parser = argparse.ArgumentParser()
        ArgumentGroupRegistrar.register_bot_config(parser)

        args = parser.parse_args(["--profile", "prod", "--dry-run"])

        assert args.profile == "prod"
        assert args.dry_run is True

    def test_register_bot_config_with_limits(self):
        """Test parsing of max-trade-value and symbol-position-caps arguments."""
        parser = argparse.ArgumentParser()
        ArgumentGroupRegistrar.register_bot_config(parser)

        args = parser.parse_args(
            [
                "--max-trade-value",
                "25000",
                "--symbol-position-caps",
                "BTC-USD:1.5",
                "ETH-USD:10",
            ]
        )

        assert args.max_trade_value == Decimal("25000")
        assert args.symbol_position_caps == ["BTC-USD:1.5", "ETH-USD:10"]

    def test_register_account(self):
        """Test registering account arguments."""
        parser = argparse.ArgumentParser()
        ArgumentGroupRegistrar.register_account(parser)

        args = parser.parse_args(["--account-snapshot"])

        assert args.account_snapshot is True

    def test_register_convert(self):
        """Test registering convert arguments."""
        parser = argparse.ArgumentParser()
        ArgumentGroupRegistrar.register_convert(parser)

        args = parser.parse_args(["--convert", "USD:USDC:100"])

        assert args.convert == "USD:USDC:100"

    def test_register_move_funds(self):
        """Test registering move funds arguments."""
        parser = argparse.ArgumentParser()
        ArgumentGroupRegistrar.register_move_funds(parser)

        args = parser.parse_args(["--move-funds", "DEFAULT:PERP:50"])

        assert args.move_funds == "DEFAULT:PERP:50"

    def test_register_order_tooling(self):
        """Test registering order tooling arguments."""
        parser = argparse.ArgumentParser()
        ArgumentGroupRegistrar.register_order_tooling(parser)

        args = parser.parse_args(
            [
                "--preview-order",
                "--order-symbol",
                "BTC-PERP",
                "--order-side",
                "buy",
                "--order-quantity",
                "0.5",
            ]
        )

        assert args.preview_order is True
        assert args.order_symbol == "BTC-PERP"
        assert args.order_side == "buy"
        assert args.order_quantity == Decimal("0.5")

    def test_register_dev(self):
        """Test registering dev arguments."""
        parser = argparse.ArgumentParser()
        ArgumentGroupRegistrar.register_dev(parser)

        args = parser.parse_args(["--dev-fast"])

        assert args.dev_fast is True

    def test_register_all(self):
        """Test registering all argument groups."""
        parser = argparse.ArgumentParser()
        ArgumentGroupRegistrar.register_all(parser)

        args = parser.parse_args(
            [
                "--profile",
                "dev",
                "--dry-run",
                "--account-snapshot",
                "--dev-fast",
            ]
        )

        assert args.profile == "dev"
        assert args.dry_run is True
        assert args.account_snapshot is True
        assert args.dev_fast is True


# ============================================================================
# Test: Specific Argument Properties
# ============================================================================


class TestArgumentProperties:
    """Test specific argument properties."""

    def test_profile_choices(self):
        """Test profile argument has correct choices."""
        parser = argparse.ArgumentParser()
        ArgumentGroupRegistrar.register_bot_config(parser)

        # Valid choice
        args = parser.parse_args(["--profile", "canary"])
        assert args.profile == "canary"

        # Invalid choice should raise
        with pytest.raises(SystemExit):
            parser.parse_args(["--profile", "invalid"])

    def test_symbols_nargs(self):
        """Test symbols accepts multiple values."""
        parser = argparse.ArgumentParser()
        ArgumentGroupRegistrar.register_bot_config(parser)

        args = parser.parse_args(["--symbols", "BTC-PERP", "ETH-PERP", "SOL-PERP"])

        assert args.symbols == ["BTC-PERP", "ETH-PERP", "SOL-PERP"]

    def test_leverage_dest(self):
        """Test leverage uses custom dest."""
        parser = argparse.ArgumentParser()
        ArgumentGroupRegistrar.register_bot_config(parser)

        args = parser.parse_args(["--leverage", "3"])

        assert args.target_leverage == 3

    def test_order_quantity_decimal_type(self):
        """Test order quantity uses Decimal type."""
        parser = argparse.ArgumentParser()
        ArgumentGroupRegistrar.register_order_tooling(parser)

        args = parser.parse_args(["--order-quantity", "0.123456"])

        assert args.order_quantity == Decimal("0.123456")
        assert isinstance(args.order_quantity, Decimal)
