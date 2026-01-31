# Security Package

This package contains security-related utilities and validators.

## Public API

Use `gpt_trader.security.validate` for all security validation calls outside
`src/gpt_trader/security/`. That module is the stable entrypoint for:

- `SecurityValidator` (facade)
- `get_validator` / convenience helpers
- `ValidationResult` and `RateLimitConfig` types

## Layering

- Public surface: `validate.py`
- Internal implementation: `security_validator.py` plus helper modules
  (input sanitizing, symbol checks, rate limiting, suspicious activity detection)
- Other security utilities (e.g. `secrets_manager.py`) are separate and should not
  depend on the validation internals.

If you need to extend validation, add it behind the facade and re-export via
`validate.py`.
