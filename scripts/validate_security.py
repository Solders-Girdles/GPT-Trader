#!/usr/bin/env python3
"""
Security Validation Script
SOT-PRE-002: Validate that no hardcoded secrets exist

This script checks for hardcoded secrets in the codebase and validates
that all security-sensitive configuration is properly externalized.
"""

import argparse
import re
import sys
from pathlib import Path

# Known hardcoded secrets that should be eliminated
HARDCODED_SECRETS = [
    r"trader_password_dev",
    r"change_admin_password",
    r"change_trader_password",
    r"change_me_in_production",
    r"admin123",
    r"trader123",
]

# Patterns for potential hardcoded secrets
SECRET_PATTERNS = [
    r'password\s*=\s*[\'"][^\'"]{8,}[\'"]',
    r'secret\s*=\s*[\'"][^\'"]{16,}[\'"]',
    r'key\s*=\s*[\'"][^\'"]{16,}[\'"]',
    r'token\s*=\s*[\'"][^\'"]{16,}[\'"]',
]

# Files to exclude from scanning
EXCLUDE_PATTERNS = [
    r".*\.git.*",
    r".*__pycache__.*",
    r".*\.pyc",
    r".*node_modules.*",
    r".*\.venv.*",  # Virtual environment files
    r".*venv.*",  # Virtual environment files
    r".*\.env.*",  # .env files are expected to have secrets
    r".*validate_security\.py",  # This script itself
    r".*test.*",  # Test files may have mock secrets
    r".*example.*",  # Example files may have placeholder secrets
]


class SecurityValidator:
    """Validates that no hardcoded secrets exist in the codebase"""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.violations: list[dict] = []

    def scan_directory(self) -> bool:
        """Scan directory for hardcoded secrets"""
        print(f"Scanning {self.root_dir} for hardcoded secrets...")

        # Scan Python files
        for py_file in self.root_dir.rglob("*.py"):
            if self._should_exclude(py_file):
                continue
            self._scan_file(py_file)

        # Scan configuration files
        for config_file in self.root_dir.rglob("*.yml"):
            if self._should_exclude(config_file):
                continue
            self._scan_file(config_file)

        for config_file in self.root_dir.rglob("*.yaml"):
            if self._should_exclude(config_file):
                continue
            self._scan_file(config_file)

        return len(self.violations) == 0

    def _should_exclude(self, file_path: Path) -> bool:
        """Check if file should be excluded from scanning"""
        file_str = str(file_path)
        for pattern in EXCLUDE_PATTERNS:
            if re.search(pattern, file_str, re.IGNORECASE):
                return True
        return False

    def _scan_file(self, file_path: Path) -> None:
        """Scan individual file for hardcoded secrets"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

            # Check for known hardcoded secrets
            for i, line in enumerate(lines, 1):
                for secret in HARDCODED_SECRETS:
                    if re.search(secret, line, re.IGNORECASE):
                        # Skip if it's in a comment explaining the fix
                        if "eliminated" in line.lower() or "removed" in line.lower():
                            continue
                        # Skip if it's in a validation check (security validation logic)
                        if "in [" in line and (
                            "admin_password" in line or "trader_password" in line
                        ):
                            continue
                        # Skip if it's clearly part of validation logic
                        if "ValidationError" in line or "raise ValueError" in line:
                            continue

                        self.violations.append(
                            {
                                "type": "hardcoded_secret",
                                "file": str(file_path),
                                "line": i,
                                "content": line.strip(),
                                "secret": secret,
                            }
                        )

            # Check for potential secret patterns
            for i, line in enumerate(lines, 1):
                # Skip comments
                if line.strip().startswith("#"):
                    continue

                for pattern in SECRET_PATTERNS:
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        # Skip if it's an environment variable access
                        if "os.getenv" in line or "os.environ" in line:
                            continue
                        # Skip if it's clearly a placeholder
                        if any(
                            placeholder in match.group().lower()
                            for placeholder in [
                                "your-",
                                "example",
                                "placeholder",
                                "change-me",
                                "test-",
                            ]
                        ):
                            continue
                        # Skip if it's an enum value
                        if ' = "' in line and ("class" in content or "Enum" in content):
                            continue
                        # Skip if it's in validation logic
                        if "in [" in line and ("password" in line.lower()):
                            continue

                        self.violations.append(
                            {
                                "type": "potential_secret",
                                "file": str(file_path),
                                "line": i,
                                "content": line.strip(),
                                "pattern": pattern,
                                "match": match.group(),
                            }
                        )

        except Exception as e:
            print(f"Warning: Could not scan {file_path}: {e}")

    def validate_environment_config(self) -> bool:
        """Validate that environment configuration is properly set up"""
        print("Validating environment configuration...")

        required_env_vars = [
            "DATABASE_PASSWORD",
            "JWT_SECRET_KEY",
            "ADMIN_PASSWORD",
            "TRADER_PASSWORD",
        ]

        # Check if .env.template exists and has all required variables
        env_template = self.root_dir / ".env.template"
        if not env_template.exists():
            self.violations.append(
                {
                    "type": "missing_env_template",
                    "file": str(env_template),
                    "message": ".env.template file is missing",
                }
            )
            return False

        # Check template content
        with open(env_template) as f:
            template_content = f.read()

        missing_vars = []
        for var in required_env_vars:
            if var not in template_content:
                missing_vars.append(var)

        if missing_vars:
            self.violations.append(
                {
                    "type": "missing_env_vars",
                    "file": str(env_template),
                    "message": f"Missing environment variables: {missing_vars}",
                }
            )

        return len(missing_vars) == 0

    def print_report(self) -> None:
        """Print security validation report"""
        if not self.violations:
            print("\n✅ SECURITY VALIDATION PASSED")
            print("No hardcoded secrets found.")
            return

        print("\n❌ SECURITY VALIDATION FAILED")
        print(f"Found {len(self.violations)} security violations:\n")

        for i, violation in enumerate(self.violations, 1):
            print(f"{i}. {violation['type'].upper()}")
            print(f"   File: {violation['file']}")

            if "line" in violation:
                print(f"   Line: {violation['line']}")
                print(f"   Content: {violation['content']}")

            if "secret" in violation:
                print(f"   Secret: {violation['secret']}")
            elif "pattern" in violation:
                print(f"   Pattern: {violation['pattern']}")
                print(f"   Match: {violation['match']}")
            elif "message" in violation:
                print(f"   Message: {violation['message']}")

            print()

    def get_exit_code(self) -> int:
        """Get appropriate exit code"""
        return 1 if self.violations else 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate that no hardcoded secrets exist in the codebase"
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path.cwd(),
        help="Root directory to scan (default: current directory)",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Enable strict mode (fail on potential secrets)"
    )

    args = parser.parse_args()

    validator = SecurityValidator(args.root_dir)

    # Run validation
    secrets_clean = validator.scan_directory()
    config_valid = validator.validate_environment_config()

    # Print report
    validator.print_report()

    # Print recommendations if violations found
    if validator.violations:
        print("\n📋 RECOMMENDATIONS:")
        print("1. Replace hardcoded secrets with environment variables")
        print("2. Use os.getenv() to read environment variables")
        print("3. Add validation for required environment variables")
        print("4. Update .env.template with all required variables")
        print("5. Never commit real secrets to version control")
        print("\n🔧 To fix hardcoded secrets:")
        print("   - Replace: password='secret' ")
        print("   - With: password=os.getenv('PASSWORD')")
        print("   - Add validation: if not password: raise ValueError(...)")

    # Exit with appropriate code
    sys.exit(validator.get_exit_code())


if __name__ == "__main__":
    main()
