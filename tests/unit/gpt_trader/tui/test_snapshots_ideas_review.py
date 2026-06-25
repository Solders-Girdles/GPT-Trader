"""Visual snapshots for the trade-idea review screen."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea
from textual.app import App

from gpt_trader.features.trade_ideas import TradeIdeaService
from gpt_trader.tui.screens.ideas_review_screen import IdeasReviewScreen


class IdeasReviewSnapshotApp(App):
    def __init__(self, service: TradeIdeaService) -> None:
        super().__init__()
        self._service = service

    async def action_open_ideas(self) -> None:
        await self.push_screen(IdeasReviewScreen(service=self._service, reviewer_id="rj"))


def _service(root: Path) -> TradeIdeaService:
    return TradeIdeaService(
        root / "trade_ideas",
        now_factory=lambda: datetime(2026, 6, 12, 10, 0, tzinfo=UTC),
    )


class TestIdeasReviewSnapshots:
    def test_empty_ideas_review_screen(self, snap_compare, tmp_path: Path) -> None:
        async def open_ideas(pilot) -> None:
            await pilot.app.action_open_ideas()
            await pilot.pause()

        assert snap_compare(
            IdeasReviewSnapshotApp(_service(tmp_path)),
            terminal_size=(120, 36),
            run_before=open_ideas,
        )

    def test_populated_ideas_review_screen(self, snap_compare, tmp_path: Path) -> None:
        service = _service(tmp_path)
        service.propose(build_trade_idea(), actor_id="idea-generator-v1")
        service.propose(
            build_trade_idea(decision_id="trade-20260612-002", thesis="ETH failed breakout"),
            actor_id="idea-generator-v1",
        )
        service.request_changes(
            "trade-20260612-002",
            actor_id="rj",
            reason="Add stronger invalidation evidence",
        )

        async def open_ideas(pilot) -> None:
            await pilot.app.action_open_ideas()
            await pilot.pause()

        assert snap_compare(
            IdeasReviewSnapshotApp(service),
            terminal_size=(120, 36),
            run_before=open_ideas,
        )
