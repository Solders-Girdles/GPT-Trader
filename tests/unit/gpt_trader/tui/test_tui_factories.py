from __future__ import annotations

from tests.unit.gpt_trader.tui.factories import BotStatusFactory, TuiStateFactory


class TestBotStatusFactory:
    def test_creates_valid_status(self) -> None:
        status = BotStatusFactory.create_running(uptime=500.0, cycle_count=100)

        assert status.engine.running is True
        assert status.engine.uptime_seconds == 500.0
        assert status.engine.cycle_count == 100

    def test_creates_with_positions(self) -> None:
        status = BotStatusFactory.create_with_positions()

        assert status.positions.count > 0
        assert len(status.positions.symbols) > 0
        assert "BTC-USD" in status.positions.symbols


class TestTuiStateFactory:
    def test_creates_valid_state(self) -> None:
        state = TuiStateFactory.create_running(mode="paper")

        assert state.running is True
        assert state.data_source_mode == "paper"

    def test_creates_with_positions(self) -> None:
        state = TuiStateFactory.create_with_positions()

        assert state.running is True
        assert len(state.position_data.positions) > 0
        assert "BTC-USD" in state.position_data.positions
