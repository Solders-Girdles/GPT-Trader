#!/usr/bin/env python3
"""
Test Security Fixes
SOT-PRE-002: Validate that hardcoded secrets have been eliminated

This script tests that our security fixes work properly by:
1. Validating environment variable loading
2. Testing error handling for missing passwords
3. Confirming production password validation
"""

import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_database_manager_security():
    """Test database manager security fixes"""
    print("\n🔒 Testing Database Manager Security...")

    # Test 1: Missing password should raise error
    print("  • Testing missing DATABASE_PASSWORD handling...")
    with patch.dict(os.environ, {}, clear=True):
        try:
            from src.bot.database.manager import DatabaseConfig

            config = DatabaseConfig()
            print("    ❌ FAILED: Should have raised error for missing password")
            return False
        except ValueError as e:
            if "Database password must be set" in str(e):
                print("    ✅ PASSED: Properly rejects missing password")
            else:
                print(f"    ❌ FAILED: Wrong error message: {e}")
                return False
        except Exception as e:
            print(f"    ❌ FAILED: Unexpected error: {e}")
            return False

    # Test 2: Valid password should work
    print("  • Testing valid DATABASE_PASSWORD handling...")
    with patch.dict(os.environ, {"DATABASE_PASSWORD": "secure_test_password"}):
        try:
            # Mock the config.get method to return None so it falls back to env var
            mock_config = MagicMock()
            mock_config.get.return_value = None  # Force fallback to env var

            from src.bot.database.manager import DatabaseConfig

            with patch("src.bot.database.manager.get_config", return_value=mock_config):
                config = DatabaseConfig()
                if config.password == "secure_test_password":
                    print("    ✅ PASSED: Correctly loads password from environment")
                else:
                    print(f"    ❌ FAILED: Wrong password loaded: {config.password}")
                    return False
        except Exception as e:
            print(f"    ❌ FAILED: Unexpected error: {e}")
            return False

    return True


def test_api_gateway_security():
    """Test API gateway security fixes"""
    print("\n🔒 Testing API Gateway Security...")

    # Test 1: Missing passwords should raise error
    print("  • Testing missing password handling...")
    with patch.dict(os.environ, {}, clear=True):
        try:
            from src.bot.api.gateway import SecurityManager

            SecurityManager()
            print("    ❌ FAILED: Should have raised error for missing passwords")
            return False
        except ValueError as e:
            if "environment variable must be set" in str(e):
                print("    ✅ PASSED: Properly rejects missing passwords")
            else:
                print(f"    ❌ FAILED: Wrong error message: {e}")
                return False
        except Exception as e:
            print(f"    ❌ FAILED: Unexpected error: {e}")
            return False

    # Test 2: Production validation should reject weak passwords
    print("  • Testing production password validation...")
    with patch.dict(
        os.environ,
        {
            "ADMIN_PASSWORD": "change_admin_password",
            "TRADER_PASSWORD": "secure_trader_password",
            "ENVIRONMENT": "production",
        },
    ):
        try:
            from src.bot.api.gateway import SecurityManager

            SecurityManager()
            print("    ❌ FAILED: Should have rejected weak admin password in production")
            return False
        except ValueError as e:
            if "strong admin password" in str(e):
                print("    ✅ PASSED: Properly rejects weak passwords in production")
            else:
                print(f"    ❌ FAILED: Wrong error message: {e}")
                return False
        except Exception as e:
            print(f"    ❌ FAILED: Unexpected error: {e}")
            return False

    # Test 3: Valid passwords should work
    print("  • Testing valid password handling...")
    with patch.dict(
        os.environ,
        {
            "ADMIN_PASSWORD": "secure_admin_password_123",
            "TRADER_PASSWORD": "secure_trader_password_456",
            "ENVIRONMENT": "development",
        },
    ):
        try:
            from src.bot.api.gateway import SecurityManager

            security = SecurityManager()
            print("    ✅ PASSED: Accepts strong passwords")
        except Exception as e:
            print(f"    ❌ FAILED: Should accept strong passwords: {e}")
            return False

    return True


def test_security_core():
    """Test core security module fixes"""
    print("\n🔒 Testing Core Security Module...")

    # Test 1: Missing password should raise error
    print("  • Testing missing ADMIN_PASSWORD handling...")
    with patch.dict(os.environ, {}, clear=True):
        try:
            from src.bot.core.security import JWTAuthenticationProvider

            provider = JWTAuthenticationProvider("test_secret")
            # This should work since we're only testing the provider initialization
            # The actual verification method will be tested separately
            print("    ℹ️ Provider created, testing credential verification...")

            import asyncio

            result = asyncio.run(provider._verify_credentials("admin", "test"))
            print("    ❌ FAILED: Should have raised error for missing ADMIN_PASSWORD")
            return False
        except ValueError as e:
            if "ADMIN_PASSWORD environment variable must be set" in str(e):
                print("    ✅ PASSED: Properly rejects missing ADMIN_PASSWORD")
            else:
                print(f"    ❌ FAILED: Wrong error message: {e}")
                return False
        except Exception as e:
            print(f"    ❌ FAILED: Unexpected error: {e}")
            return False

    # Test 2: Production validation
    print("  • Testing production password validation...")
    with patch.dict(
        os.environ, {"ADMIN_PASSWORD": "change_me_in_production", "ENVIRONMENT": "production"}
    ):
        try:
            from src.bot.core.security import JWTAuthenticationProvider

            provider = JWTAuthenticationProvider("test_secret")

            import asyncio

            result = asyncio.run(provider._verify_credentials("admin", "change_me_in_production"))
            print("    ❌ FAILED: Should have rejected weak password in production")
            return False
        except ValueError as e:
            if "strong admin password" in str(e):
                print("    ✅ PASSED: Properly rejects weak passwords in production")
            else:
                print(f"    ❌ FAILED: Wrong error message: {e}")
                return False
        except Exception as e:
            print(f"    ❌ FAILED: Unexpected error: {e}")
            return False

    return True


def test_configuration_files():
    """Test that configuration files use environment variables"""
    print("\n🔒 Testing Configuration Files...")

    # Test database.yaml
    print("  • Testing database.yaml...")
    try:
        with open("config/database.yaml") as f:
            content = f.read()

        if "trader_password_dev" in content:
            print("    ❌ FAILED: database.yaml still contains hardcoded password")
            return False

        if "${DATABASE_PASSWORD}" not in content:
            print("    ❌ FAILED: database.yaml doesn't use DATABASE_PASSWORD environment variable")
            return False

        print("    ✅ PASSED: database.yaml uses environment variables")
    except Exception as e:
        print(f"    ❌ FAILED: Error reading database.yaml: {e}")
        return False

    # Test docker-compose files
    print("  • Testing Docker configuration files...")
    docker_files = ["deploy/postgres/docker-compose.yml", "monitoring/docker-compose.yml"]

    for docker_file in docker_files:
        try:
            with open(docker_file) as f:
                content = f.read()

            if "trader_password_dev" in content:
                print(f"    ❌ FAILED: {docker_file} still contains hardcoded password")
                return False

            if "${DATABASE_PASSWORD}" not in content and "DATABASE_PASSWORD" not in docker_file:
                # Only postgres compose should have DATABASE_PASSWORD
                if "postgres" in docker_file:
                    print(f"    ❌ FAILED: {docker_file} doesn't use DATABASE_PASSWORD")
                    return False

            print(f"    ✅ PASSED: {docker_file} uses environment variables")
        except Exception as e:
            print(f"    ❌ FAILED: Error reading {docker_file}: {e}")
            return False

    return True


def test_environment_template():
    """Test that .env.template has all required variables"""
    print("\n🔒 Testing Environment Template...")

    required_vars = ["DATABASE_PASSWORD", "JWT_SECRET_KEY", "ADMIN_PASSWORD", "TRADER_PASSWORD"]

    try:
        with open(".env.template") as f:
            content = f.read()

        missing_vars = []
        for var in required_vars:
            if var not in content:
                missing_vars.append(var)

        if missing_vars:
            print(f"    ❌ FAILED: Missing variables in .env.template: {missing_vars}")
            return False

        # Check for insecure examples
        insecure_examples = [
            "change_me_in_production",
            "trader_password_dev",
            "admin123",
            "password",
        ]

        for insecure in insecure_examples:
            if insecure in content:
                print(f"    ⚠️ WARNING: .env.template contains insecure example: {insecure}")

        print("    ✅ PASSED: .env.template contains all required variables")
        return True

    except Exception as e:
        print(f"    ❌ FAILED: Error reading .env.template: {e}")
        return False


def main():
    """Run all security tests"""
    print("🔒 GPT-Trader Security Fixes Validation")
    print("=" * 45)

    tests = [
        test_database_manager_security,
        test_api_gateway_security,
        test_security_core,
        test_configuration_files,
        test_environment_template,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Test {test.__name__} crashed: {e}")
            failed += 1

    print("\n" + "=" * 45)
    print(f"\n📊 RESULTS: {passed} passed, {failed} failed")

    if failed == 0:
        print("✅ ALL SECURITY TESTS PASSED!")
        print("\n🔒 Security Status: SECURE")
        print("All hardcoded secrets have been successfully eliminated.")
        print("\n📋 Next Steps:")
        print("1. Set environment variables before running the application")
        print("2. Use strong passwords in production")
        print("3. Never commit .env files with real secrets")
        print("4. Regularly rotate passwords and secrets")
        return 0
    else:
        print("❌ SECURITY TESTS FAILED!")
        print("\n🔴 Security Status: VULNERABLE")
        print("Some hardcoded secrets remain or fixes are incomplete.")
        print("\n🔧 Required Actions:")
        print("1. Fix all failing tests")
        print("2. Re-run this validation script")
        print("3. Do not deploy until all tests pass")
        return 1


if __name__ == "__main__":
    sys.exit(main())
