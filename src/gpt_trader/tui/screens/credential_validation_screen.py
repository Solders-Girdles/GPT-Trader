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
        Binding("r", "retry", "Retry", show=False),
        Binding("s", "setup", "Setup", show=False),
    ]

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

            # Scrollable findings list grouped by category
            with VerticalScroll(id="findings-scroll"):
                yield from self._create_grouped_findings()

            # Summary line
            yield self._create_summary()

            # Action buttons
            with Container(id="button-container"):
                if self.result.valid_for_mode:
                    yield Button("Proceed", id="proceed-btn", variant="success")
                else:
                    # Offer retry and setup wizard when validation fails
                    yield Button("Retry", id="retry-btn", variant="warning")
                    yield Button("Setup API Key", id="setup-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn", variant="error")

    def _create_grouped_findings(self) -> list[Static | Label]:
        """Create findings grouped by category with section headers."""
        from collections import defaultdict

        widgets: list[Static | Label] = []

        # Group findings by category
        by_category: dict[ValidationCategory, list[ValidationFinding]] = defaultdict(list)
        for finding in self.result.findings:
            by_category[finding.category].append(finding)

        # Display each category group
        for category in ValidationCategory:
            findings = by_category.get(category, [])
            if not findings:
                continue

            # Category section header
            category_name = CATEGORY_NAMES.get(category, category.value)

            # Determine category status based on findings
            has_error = any(f.severity == ValidationSeverity.ERROR for f in findings)
            has_warning = any(f.severity == ValidationSeverity.WARNING for f in findings)

            if has_error:
                status_icon = "✗"
                header_class = "category-header category-error"
            elif has_warning:
                status_icon = "⚠"
                header_class = "category-header category-warning"
            else:
                status_icon = "✓"
                header_class = "category-header category-success"

            widgets.append(
                Label(f"{status_icon} {category_name}", classes=header_class)
            )

            # Individual findings under this category
            for finding in findings:
                widgets.extend(self._create_finding_row(finding))

        return widgets

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
        severity_class = f"finding-{finding.severity.value}"
        row = Static(
            f"[{icon}] [{category}] {finding.message}",
            classes=f"finding-row {severity_class}",
        )

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

    def action_retry(self) -> None:
        """Retry credential validation."""
        self.dismiss("retry")

    def action_setup(self) -> None:
        """Launch the API setup wizard."""
        self.dismiss("setup")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "proceed-btn":
            self.action_proceed()
        elif event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "setup-btn":
            # Signal to launch the setup wizard
            self.dismiss("setup")
        elif event.button.id == "retry-btn":
            # Signal to retry validation
            self.dismiss("retry")
