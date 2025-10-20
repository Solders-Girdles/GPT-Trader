"""MFA verification tests for AuthHandler."""

from __future__ import annotations

from typing import Any

from bot_v2.security.auth_handler import AuthHandler


class TestMFAVerification:
    """Validate TOTP verification edge cases."""

    def test_verify_valid_mfa_code(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.mfa_enabled = True
        user.mfa_secret = "JBSWY3DPEHPK3PXP"
        auth_handler._users[user.id] = user

        assert auth_handler._verify_mfa(user, "123456") is True

    def test_verify_invalid_mfa_code(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.mfa_enabled = True
        user.mfa_secret = "JBSWY3DPEHPK3PXP"
        auth_handler._users[user.id] = user

        assert auth_handler._verify_mfa(user, "000000") is False

    def test_verify_mfa_without_secret(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.mfa_enabled = True
        user.mfa_secret = None
        auth_handler._users[user.id] = user

        assert auth_handler._verify_mfa(user, "123456") is False

    def test_verify_mfa_with_valid_window(
        self, auth_handler: AuthHandler, fake_pyotp_module: Any, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        class MockTOTPWithWindow:
            def __init__(self, secret: str):
                self.secret = secret

            def verify(self, code: str, valid_window: int = 0) -> bool:
                return (
                    code in ["123456", "123455", "123457"]
                    if valid_window >= 1
                    else code == "123456"
                )

        fake_pyotp_module.TOTP = MockTOTPWithWindow

        user = User(**sample_user)
        user.mfa_enabled = True
        user.mfa_secret = "JBSWY3DPEHPK3PXP"
        auth_handler._users[user.id] = user

        assert auth_handler._verify_mfa(user, "123455") is True
        assert auth_handler._verify_mfa(user, "123457") is True
