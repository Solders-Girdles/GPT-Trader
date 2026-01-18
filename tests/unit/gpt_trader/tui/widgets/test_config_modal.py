from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

from textual.widgets import Input

from gpt_trader.tui.widgets.config import ConfigModal


def test_config_modal_save_updates_risk_config(mock_bot) -> None:
    mock_bot.config.risk = MagicMock()
    mock_bot.config.risk.max_leverage = 1.0
    mock_bot.config.risk.daily_loss_limit_pct = 0.02

    modal = ConfigModal(mock_bot.config)

    with patch("textual.widget.Widget.app", new_callable=PropertyMock) as mock_app_prop:
        mock_app = MagicMock()
        mock_app_prop.return_value = mock_app

        input_leverage = MagicMock(spec=Input)
        input_leverage.value = "5.0"

        input_loss = MagicMock(spec=Input)
        input_loss.value = "0.05"

        def query_side_effect(selector: str, type: type | None = None):
            if "leverage" in selector:
                return input_leverage
            if "loss" in selector:
                return input_loss
            return MagicMock()

        modal.query_one = MagicMock(side_effect=query_side_effect)

        modal._save_config()

        assert mock_bot.config.risk.max_leverage == 5.0
        assert mock_bot.config.risk.daily_loss_limit_pct == 0.05
        mock_app.notify.assert_called_with("Configuration updated successfully.", title="Config")
