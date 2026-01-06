"""Tests for PanicModal widget."""

from unittest.mock import MagicMock

from textual.widgets import Button, Input

from gpt_trader.tui.widgets.panic import PanicModal


class TestPanicModalConfirmButton:
    """Test suite for PanicModal confirm button enable/disable behavior."""

    def _create_modal_with_mocks(self):
        """Create PanicModal with mocked query_one."""
        modal = PanicModal()
        mock_button = MagicMock(spec=Button)
        mock_button.disabled = True

        def query_side_effect(selector, widget_type=None):
            if selector == "#btn-panic-confirm" or widget_type == Button:
                return mock_button
            return MagicMock()

        modal.query_one = MagicMock(side_effect=query_side_effect)
        modal.dismiss = MagicMock()
        return modal, mock_button

    def test_confirm_button_disabled_with_empty_input(self):
        """Test that confirm button stays disabled with empty input."""
        modal, mock_button = self._create_modal_with_mocks()

        # Simulate empty input - button should stay disabled
        event = MagicMock(spec=Input.Changed)
        event.value = ""

        modal.on_input_changed(event)

        assert mock_button.disabled is True

    def test_input_flatten_enables_confirm_button(self):
        """Test that typing 'FLATTEN' enables the confirm button."""
        modal, mock_button = self._create_modal_with_mocks()

        event = MagicMock(spec=Input.Changed)
        event.value = "FLATTEN"

        modal.on_input_changed(event)

        assert mock_button.disabled is False

    def test_input_lowercase_flatten_does_not_enable(self):
        """Test that lowercase 'flatten' does not enable the confirm button."""
        modal, mock_button = self._create_modal_with_mocks()

        event = MagicMock(spec=Input.Changed)
        event.value = "flatten"

        modal.on_input_changed(event)

        assert mock_button.disabled is True

    def test_input_partial_flatten_does_not_enable(self):
        """Test that partial 'FLAT' does not enable the confirm button."""
        modal, mock_button = self._create_modal_with_mocks()

        event = MagicMock(spec=Input.Changed)
        event.value = "FLAT"

        modal.on_input_changed(event)

        assert mock_button.disabled is True

    def test_input_with_extra_chars_does_not_enable(self):
        """Test that 'FLATTEN ' with extra chars does not enable."""
        modal, mock_button = self._create_modal_with_mocks()

        event = MagicMock(spec=Input.Changed)
        event.value = "FLATTEN "

        modal.on_input_changed(event)

        assert mock_button.disabled is True

    def test_input_wrong_text_disables_button(self):
        """Test that wrong text keeps button disabled."""
        modal, mock_button = self._create_modal_with_mocks()

        # First enable with correct input
        event = MagicMock(spec=Input.Changed)
        event.value = "FLATTEN"
        modal.on_input_changed(event)
        assert mock_button.disabled is False

        # Then change to wrong input
        event.value = "WRONG"
        modal.on_input_changed(event)
        assert mock_button.disabled is True


class TestPanicModalDismissBehavior:
    """Test suite for PanicModal dismiss behavior."""

    def _create_modal_with_mocks(self):
        """Create PanicModal with mocked dismiss."""
        modal = PanicModal()
        modal.dismiss = MagicMock()
        return modal

    def test_cancel_button_dismisses_with_false(self):
        """Test that cancel button dismisses modal with False."""
        modal = self._create_modal_with_mocks()

        mock_button = MagicMock(spec=Button)
        mock_button.id = "btn-panic-cancel"
        event = MagicMock(spec=Button.Pressed)
        event.button = mock_button

        modal.on_button_pressed(event)

        modal.dismiss.assert_called_once_with(False)

    def test_confirm_button_dismisses_with_true(self):
        """Test that confirm button dismisses modal with True."""
        modal = self._create_modal_with_mocks()

        mock_button = MagicMock(spec=Button)
        mock_button.id = "btn-panic-confirm"
        event = MagicMock(spec=Button.Pressed)
        event.button = mock_button

        modal.on_button_pressed(event)

        modal.dismiss.assert_called_once_with(True)

    def test_unknown_button_does_not_dismiss(self):
        """Test that unknown button ID does not call dismiss."""
        modal = self._create_modal_with_mocks()

        mock_button = MagicMock(spec=Button)
        mock_button.id = "unknown-button"
        event = MagicMock(spec=Button.Pressed)
        event.button = mock_button

        modal.on_button_pressed(event)

        modal.dismiss.assert_not_called()


class TestPanicModalIntegration:
    """Integration tests for PanicModal behavior."""

    def test_modal_class_exists_and_inherits_from_modal_screen(self):
        """Test that PanicModal exists and inherits from ModalScreen."""
        from textual.screen import ModalScreen

        assert issubclass(PanicModal, ModalScreen)

    def test_modal_has_compose_method(self):
        """Test that PanicModal has a compose method."""
        modal = PanicModal()
        assert hasattr(modal, "compose")
        assert callable(modal.compose)

    def test_modal_has_on_input_changed_method(self):
        """Test that PanicModal has on_input_changed handler."""
        modal = PanicModal()
        assert hasattr(modal, "on_input_changed")
        assert callable(modal.on_input_changed)

    def test_modal_has_on_button_pressed_method(self):
        """Test that PanicModal has on_button_pressed handler."""
        modal = PanicModal()
        assert hasattr(modal, "on_button_pressed")
        assert callable(modal.on_button_pressed)
