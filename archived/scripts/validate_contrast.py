"""Validate WCAG AA color contrast ratios."""

from gpt_trader.tui.theme import THEME


def calculate_luminance(hex_color: str) -> float:
    """Calculate relative luminance of a color."""
    # Remove '#' and convert to RGB
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Apply gamma correction
    def gamma(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = gamma(r), gamma(g), gamma(b)

    # Calculate luminance
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(color1: str, color2: str) -> float:
    """Calculate WCAG contrast ratio between two colors."""
    l1 = calculate_luminance(color1)
    l2 = calculate_luminance(color2)

    lighter = max(l1, l2)
    darker = min(l1, l2)

    return (lighter + 0.05) / (darker + 0.05)


def validate_wcag_aa():
    """Validate all color combinations meet WCAG AA (4.5:1)."""
    colors = THEME.colors

    checks = [
        ("text_primary", "background_primary", 4.5),
        ("text_secondary", "background_primary", 4.5),
        ("accent_primary", "background_primary", 3.0),  # Large text
        ("success", "background_primary", 4.5),
        ("error", "background_primary", 4.5),
        ("warning", "background_primary", 4.5),
    ]

    print("WCAG AA Contrast Validation")
    print("=" * 60)

    for fg_attr, bg_attr, min_ratio in checks:
        fg = getattr(colors, fg_attr)
        bg = getattr(colors, bg_attr)
        ratio = contrast_ratio(fg, bg)
        status = "✓ PASS" if ratio >= min_ratio else "✗ FAIL"

        print(f"{status} {fg_attr} / {bg_attr}: {ratio:.2f}:1 (min: {min_ratio}:1)")


if __name__ == "__main__":
    validate_wcag_aa()
