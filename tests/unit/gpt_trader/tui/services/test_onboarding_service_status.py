"""Tests for onboarding status aggregation logic."""

from gpt_trader.tui.services.onboarding_service import ChecklistItem, OnboardingStatus


class TestOnboardingStatus:
    """Tests for OnboardingStatus dataclass."""

    def test_empty_status_is_ready(self):
        status = OnboardingStatus(items=[], mode="demo")
        assert status.is_ready is True
        assert status.progress_pct == 1.0
        assert status.ready_label == "Ready"

    def test_completed_count(self):
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", completed=True),
                ChecklistItem(id="b", label="B", description="B", completed=False),
                ChecklistItem(id="c", label="C", description="C", completed=True),
            ],
            mode="demo",
        )
        assert status.completed_count == 2

    def test_required_count_excludes_skippable(self):
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", required=True),
                ChecklistItem(id="b", label="B", description="B", required=True, skippable=True),
                ChecklistItem(id="c", label="C", description="C", required=False),
            ],
            mode="demo",
        )
        assert status.required_count == 1

    def test_required_completed(self):
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", required=True, completed=True),
                ChecklistItem(id="b", label="B", description="B", required=True, completed=False),
                ChecklistItem(id="c", label="C", description="C", required=False, completed=True),
            ],
            mode="demo",
        )
        assert status.required_completed == 1

    def test_is_ready_all_required_complete(self):
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", required=True, completed=True),
                ChecklistItem(id="b", label="B", description="B", required=True, completed=True),
                ChecklistItem(id="c", label="C", description="C", required=False, completed=False),
            ],
            mode="paper",
        )
        assert status.is_ready is True

    def test_is_ready_required_incomplete(self):
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", required=True, completed=True),
                ChecklistItem(id="b", label="B", description="B", required=True, completed=False),
            ],
            mode="paper",
        )
        assert status.is_ready is False

    def test_progress_pct(self):
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", completed=True),
                ChecklistItem(id="b", label="B", description="B", completed=False),
                ChecklistItem(id="c", label="C", description="C", completed=True),
                ChecklistItem(id="d", label="D", description="D", completed=False),
            ],
            mode="demo",
        )
        assert status.progress_pct == 0.5

    def test_ready_label_when_ready(self):
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", required=True, completed=True),
            ],
            mode="paper",
        )
        assert status.ready_label == "Ready"

    def test_ready_label_when_not_ready(self):
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", required=True, completed=True),
                ChecklistItem(id="b", label="B", description="B", required=True, completed=False),
                ChecklistItem(id="c", label="C", description="C", required=True, completed=False),
            ],
            mode="paper",
        )
        assert status.ready_label == "1/3"

    def test_get_next_step_returns_first_incomplete_required(self):
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", required=True, completed=True),
                ChecklistItem(id="b", label="B", description="B", required=True, completed=False),
                ChecklistItem(id="c", label="C", description="C", required=True, completed=False),
            ],
            mode="paper",
        )
        next_step = status.get_next_step()
        assert next_step is not None
        assert next_step.id == "b"

    def test_get_next_step_returns_none_when_ready(self):
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", required=True, completed=True),
                ChecklistItem(id="b", label="B", description="B", required=False, completed=False),
            ],
            mode="paper",
        )
        assert status.get_next_step() is None
