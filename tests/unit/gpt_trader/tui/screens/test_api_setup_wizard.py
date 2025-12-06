"""Tests for API setup wizard screen."""

from gpt_trader.tui.screens.api_setup_wizard import (
    STEP_CONTENT,
    TOTAL_STEPS,
    APISetupWizardScreen,
)


class TestAPISetupWizardScreen:
    """Tests for the API setup wizard."""

    def test_wizard_starts_at_step_1(self):
        """Wizard should start at step 1."""
        screen = APISetupWizardScreen()
        assert screen.current_step == 1

    def test_wizard_has_all_steps(self):
        """All steps should have content defined."""
        assert len(STEP_CONTENT) == TOTAL_STEPS
        for step in range(1, TOTAL_STEPS + 1):
            assert step in STEP_CONTENT
            assert "title" in STEP_CONTENT[step]
            assert "content" in STEP_CONTENT[step]

    def test_next_advances_step(self):
        """Next should advance to next step."""
        screen = APISetupWizardScreen()
        screen.current_step = 1
        # Mock _update_content since we're not mounted
        screen._update_content = lambda: None
        screen.action_next()
        assert screen.current_step == 2

    def test_back_returns_to_previous(self):
        """Back should return to previous step."""
        screen = APISetupWizardScreen()
        screen.current_step = 3
        screen._update_content = lambda: None
        screen.action_previous()
        assert screen.current_step == 2

    def test_back_does_not_go_below_1(self):
        """Back should not go below step 1."""
        screen = APISetupWizardScreen()
        screen.current_step = 1
        screen._update_content = lambda: None
        screen.action_previous()
        assert screen.current_step == 1

    def test_final_step_returns_verify(self):
        """Final step should return 'verify'."""
        results = []
        screen = APISetupWizardScreen()
        screen.dismiss = lambda x: results.append(x)
        screen.current_step = TOTAL_STEPS
        screen.action_next()
        assert results == ["verify"]

    def test_cancel_returns_none(self):
        """Cancel should return None."""
        results = []
        screen = APISetupWizardScreen()
        screen.dismiss = lambda x: results.append(x)
        screen.action_cancel()
        assert results == [None]


class TestWizardContent:
    """Tests for wizard step content."""

    def test_step_1_is_welcome(self):
        """Step 1 should be welcome."""
        assert "Welcome" in STEP_CONTENT[1]["title"]

    def test_step_2_has_portal_url(self):
        """Step 2 should include developer portal URL."""
        content = "\n".join(STEP_CONTENT[2]["content"])
        assert "portal.cdp.coinbase.com" in content

    def test_step_2_mentions_ecdsa(self):
        """Step 2 should mention ECDSA algorithm."""
        content = "\n".join(STEP_CONTENT[2]["content"])
        assert "ECDSA" in content

    def test_step_2_warns_about_ed25519(self):
        """Step 2 should warn that Ed25519 is not yet supported."""
        content = "\n".join(STEP_CONTENT[2]["content"])
        assert "Ed25519" in content
        assert "NOT" in content or "not" in content.lower()

    def test_step_3_mentions_private_key(self):
        """Step 3 should mention private key format."""
        content = "\n".join(STEP_CONTENT[3]["content"])
        assert "BEGIN EC PRIVATE KEY" in content

    def test_step_4_has_env_var_examples(self):
        """Step 4 should show environment variable examples."""
        content = "\n".join(STEP_CONTENT[4]["content"])
        assert "COINBASE_CDP_API_KEY" in content
        assert "COINBASE_CDP_PRIVATE_KEY" in content

    def test_step_4_mentions_export(self):
        """Step 4 should show export command."""
        content = "\n".join(STEP_CONTENT[4]["content"])
        assert "export" in content

    def test_step_5_is_verify(self):
        """Step 5 should be verification step."""
        assert "Verify" in STEP_CONTENT[5]["title"]

    def test_all_steps_have_non_empty_content(self):
        """All steps should have non-empty content."""
        for step in range(1, TOTAL_STEPS + 1):
            content = STEP_CONTENT[step]["content"]
            assert isinstance(content, list)
            assert len(content) > 0
            # At least some lines should have actual text
            text_lines = [line for line in content if line.strip()]
            assert len(text_lines) > 0


class TestWizardNavigation:
    """Tests for wizard navigation logic."""

    def test_can_navigate_through_all_steps(self):
        """Should be able to navigate through all steps."""
        screen = APISetupWizardScreen()
        screen._update_content = lambda: None

        # Navigate forward through all steps
        for expected_step in range(1, TOTAL_STEPS + 1):
            assert screen.current_step == expected_step
            if expected_step < TOTAL_STEPS:
                screen.action_next()

        # Navigate backward through all steps
        for expected_step in range(TOTAL_STEPS, 0, -1):
            assert screen.current_step == expected_step
            if expected_step > 1:
                screen.action_previous()

    def test_step_never_exceeds_bounds(self):
        """Step should never exceed bounds during navigation."""
        screen = APISetupWizardScreen()
        screen._update_content = lambda: None
        screen.dismiss = lambda x: None  # Mock dismiss for final step

        # Try to go back from step 1
        screen.current_step = 1
        screen.action_previous()
        assert screen.current_step >= 1

        # Try to go forward past final step (should dismiss)
        screen.current_step = TOTAL_STEPS
        screen.action_next()
        # Step should not have changed since it dismisses
        assert screen.current_step == TOTAL_STEPS
