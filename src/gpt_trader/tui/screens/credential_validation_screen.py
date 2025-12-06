"""Modal screen displaying API credential validation results.

This screen shows the results of credential validation, including format checks,
connectivity tests, and permission verification. It blocks progression if
there are critical errors and allows the user to proceed or cancel.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from gpt_trader.preflight.validation_result import (
    CredentialValidationResult,
    ValidationCategory,
    ValidationFinding,
    ValidationSeverity,
)

# Icons for each severity level
SEVERITY_ICONS = {
    ValidationSeverity.SUCCESS: "✓",
    ValidationSeverity.INFO: "ℹ",
    ValidationSeverity.WARNING: "⚠",
    ValidationSeverity.ERROR: "✗",
}

# Category display names
CATEGORY_NAMES = {
    ValidationCategory.KEY_FORMAT: "Key Format",
    ValidationCategory.CONNECTIVITY: "Connectivity",
    ValidationCategory.PERMISSIONS: "Permissions",
    ValidationCategory.ACCOUNT_STATUS: "Account",
    ValidationCategory.MODE_COMPATIBILITY: "Mode Check",
}


class CredentialValidationScreen(ModalScreen[bool | str]):
    """Modal displaying credential validation findings.

    This screen presents the results of API credential validation in a
    structured format, showing what passed, failed, and any warnings.

    Returns:
        True - User chooses to proceed (only available if valid_for_mode).
        False - User cancels or there are blocking errors.
        "setup" - User wants to launch the API key setup wizard.
    """

    BINDINGS = [
        Binding("enter", "proceed", "Proceed", show=True),
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    CSS = """
    CredentialValidationScreen {
        align: center middle;
        background: rgba(58, 53, 48, 0.8);
    }

    #validation-container {
        width: 70;
        max-width: 90%;
        height: auto;
        max-height: 85%;
        background: #2A2520;
        border: thick #D4744F;
        padding: 1 2;
    }

    .validation-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        margin-bottom: 1;
    }

    .validation-title-passed {
        color: #85B77F;
        background: #2A3A28;
    }

    .validation-title-failed {
        color: #E08580;
        background: #4C342F;
    }

    .validation-title-warning {
        color: #E0B366;
        background: #4A3D28;
    }

    #findings-scroll {
        height: auto;
        max-height: 20;
        margin-bottom: 1;
    }

    .finding-row {
        height: auto;
        padding: 0 1;
        margin-bottom: 0;
    }

    .finding-icon-success {
        color: #85B77F;
        width: 3;
    }

    .finding-icon-info {
        color: #7FACD4;
        width: 3;
    }

    .finding-icon-warning {
        color: #E0B366;
        width: 3;
    }

    .finding-icon-error {
        color: #E08580;
        width: 3;
    }

    .finding-category {
        color: #B8B5B2;
        width: 14;
    }

    .finding-message {
        color: #F0EDE9;
    }

    .finding-details {
        color: #7A7672;
        margin-left: 17;
        text-style: italic;
    }

    .finding-suggestion {
        color: #7FACD4;
        margin-left: 17;
    }

    .summary-line {
        text-align: center;
        padding: 1;
        margin-top: 1;
    }

    .summary-success {
        color: #85B77F;
    }

    .summary-warning {
        color: #E0B366;
    }

    .summary-error {
        color: #E08580;
    }

    #button-container {
        height: auto;
        align: center middle;
        margin-top: 1;
    }

    #button-container Button {
        margin: 0 1;
        min-width: 16;
    }

    #proceed-btn {
        background: #3D6B35;
    }

    #proceed-btn:hover {
        background: #4A8040;
    }

    #cancel-btn {
        background: #4C342F;
    }

    #cancel-btn:hover {
        background: #5C443F;
    }

    #setup-btn {
        background: #3D5B6B;
    }

    #setup-btn:hover {
        background: #4A6B7B;
    }

    .button-disabled {
        background: #3D3833;
        color: #7A7672;
    }
    """

    def __init__(
        self,
        result: CredentialValidationResult,
        name: str | None = None,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
    ) -> None:
        """Initialize the validation screen.

        Args:
            result: The validation result to display.
            name: Optional widget name.
            id: Optional widget ID.
            classes: Optional CSS classes.
        """
        super().__init__(name=name, id=id, classes=classes)
        self.result = result

    def compose(self) -> ComposeResult:
        """Build the validation result display."""
        with Container(id="validation-container"):
            # Title based on result
            yield self._create_title()

            # Scrollable findings list
            with VerticalScroll(id="findings-scroll"):
                for finding in self.result.findings:
                    yield from self._create_finding_row(finding)

            # Summary line
            yield self._create_summary()

            # Action buttons
            with Container(id="button-container"):
                if self.result.valid_for_mode:
                    yield Button("Proceed", id="proceed-btn", variant="success")
                else:
                    # Offer setup wizard when validation fails
                    yield Button("Setup API Key", id="setup-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn", variant="error")

    def _create_title(self) -> Label:
        """Create the title label based on validation status."""
        mode = self.result.mode.upper()

        if self.result.valid_for_mode and not self.result.has_warnings:
            title = f"API Validation: PASSED ({mode})"
            css_class = "validation-title validation-title-passed"
        elif self.result.valid_for_mode and self.result.has_warnings:
            title = f"API Validation: PASSED with warnings ({mode})"
            css_class = "validation-title validation-title-warning"
        else:
            title = f"API Validation: FAILED ({mode})"
            css_class = "validation-title validation-title-failed"

        return Label(title, classes=css_class)

    def _create_finding_row(self, finding: ValidationFinding) -> list[Static]:
        """Create display elements for a single finding."""
        widgets = []

        # Icon
        icon = SEVERITY_ICONS.get(finding.severity, "·")

        # Category name
        category = CATEGORY_NAMES.get(finding.category, finding.category.value)

        # Main row with icon, category, and message
        row = Static(
            f"[{icon}] [{category}] {finding.message}",
            classes="finding-row",
        )
        # Apply icon color class based on severity
        if finding.severity == ValidationSeverity.SUCCESS:
            row.styles.color = "#85B77F"
        elif finding.severity == ValidationSeverity.INFO:
            row.styles.color = "#7FACD4"
        elif finding.severity == ValidationSeverity.WARNING:
            row.styles.color = "#E0B366"
        elif finding.severity == ValidationSeverity.ERROR:
            row.styles.color = "#E08580"

        widgets.append(row)

        # Details line (if present)
        if finding.details:
            widgets.append(Static(f"    {finding.details}", classes="finding-details"))

        # Suggestion line (if present)
        if finding.suggestion:
            widgets.append(Static(f"    → {finding.suggestion}", classes="finding-suggestion"))

        return widgets

    def _create_summary(self) -> Static:
        """Create the summary line."""
        success = self.result.success_count
        warning = self.result.warning_count
        error = self.result.error_count

        parts = []
        if success:
            parts.append(f"{success} passed")
        if warning:
            parts.append(f"{warning} warnings")
        if error:
            parts.append(f"{error} errors")

        summary_text = " • ".join(parts) if parts else "No checks performed"

        if error > 0:
            css_class = "summary-line summary-error"
            if not self.result.valid_for_mode:
                summary_text += " - Cannot proceed"
        elif warning > 0:
            css_class = "summary-line summary-warning"
            summary_text += " - Proceed with caution"
        else:
            css_class = "summary-line summary-success"
            summary_text += " - Ready to proceed"

        return Static(summary_text, classes=css_class)

    def action_proceed(self) -> None:
        """Allow proceeding only if valid_for_mode is True."""
        if self.result.valid_for_mode:
            self.dismiss(True)
        else:
            # Flash the cancel button or show notification
            self.notify(
                "Cannot proceed - fix errors above",
                severity="error",
                timeout=3,
            )

    def action_cancel(self) -> None:
        """Cancel and return to mode selection."""
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "proceed-btn":
            self.action_proceed()
        elif event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "setup-btn":
            # Signal to launch the setup wizard
            self.dismiss("setup")
