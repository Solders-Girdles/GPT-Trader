"""
Comprehensive API key permissions testing for Coinbase.

This module tests all available API key permissions that can be granted 
according to Coinbase CDP documentation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

import pytest

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import CDPAuthV2
from bot_v2.features.brokerages.core.interfaces import AuthError


class TestAPIKeyPermissions(unittest.TestCase):
    """Test API key permissions validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_auth = Mock(spec=CDPAuthV2)
        self.client = CoinbaseClient(
            base_url="https://api.coinbase.com",
            auth=self.mock_auth,
            api_mode="advanced"
        )
        
    def test_get_key_permissions_endpoint(self):
        """Test that get_key_permissions calls the correct endpoint."""
        with patch.object(self.client, '_request') as mock_request:
            mock_request.return_value = {
                "can_view": True,
                "can_trade": True,
                "can_transfer": True,
                "portfolio_uuid": "test-uuid",
                "portfolio_type": "DEFAULT"
            }
            
            result = self.client.get_key_permissions()
            
            mock_request.assert_called_once_with(
                "GET", 
                "/api/v3/brokerage/key_permissions"
            )
            self.assertTrue(result["can_view"])
            self.assertTrue(result["can_trade"])
            self.assertTrue(result["can_transfer"])
            
    def test_all_permissions_granted(self):
        """Test response when all permissions are granted."""
        with patch.object(self.client, '_request') as mock_request:
            mock_request.return_value = {
                "can_view": True,
                "can_trade": True,
                "can_transfer": True,
                "portfolio_uuid": "123e4567-e89b-12d3-a456-426614174000",
                "portfolio_type": "DEFAULT"
            }
            
            perms = self.client.get_key_permissions()
            
            # Verify all permissions are True
            self.assertTrue(perms["can_view"], "View permission should be granted")
            self.assertTrue(perms["can_trade"], "Trade permission should be granted")
            self.assertTrue(perms["can_transfer"], "Transfer permission should be granted")
            
            # Verify portfolio info is present
            self.assertEqual(perms["portfolio_uuid"], "123e4567-e89b-12d3-a456-426614174000")
            self.assertEqual(perms["portfolio_type"], "DEFAULT")
            
    def test_partial_permissions(self):
        """Test response when only some permissions are granted."""
        test_cases = [
            # View only
            {
                "response": {
                    "can_view": True,
                    "can_trade": False,
                    "can_transfer": False,
                    "portfolio_uuid": "test-uuid",
                    "portfolio_type": "DEFAULT"
                },
                "description": "View-only access"
            },
            # View and Trade
            {
                "response": {
                    "can_view": True,
                    "can_trade": True,
                    "can_transfer": False,
                    "portfolio_uuid": "test-uuid",
                    "portfolio_type": "DEFAULT"
                },
                "description": "View and Trade access"
            },
            # No permissions
            {
                "response": {
                    "can_view": False,
                    "can_trade": False,
                    "can_transfer": False,
                    "portfolio_uuid": None,
                    "portfolio_type": "UNDEFINED"
                },
                "description": "No permissions granted"
            }
        ]
        
        for test_case in test_cases:
            with self.subTest(permissions=test_case["description"]):
                with patch.object(self.client, '_request') as mock_request:
                    mock_request.return_value = test_case["response"]
                    
                    perms = self.client.get_key_permissions()
                    
                    self.assertEqual(
                        perms["can_view"], 
                        test_case["response"]["can_view"],
                        f"can_view mismatch for {test_case['description']}"
                    )
                    self.assertEqual(
                        perms["can_trade"], 
                        test_case["response"]["can_trade"],
                        f"can_trade mismatch for {test_case['description']}"
                    )
                    self.assertEqual(
                        perms["can_transfer"], 
                        test_case["response"]["can_transfer"],
                        f"can_transfer mismatch for {test_case['description']}"
                    )
                    
    def test_unauthorized_request(self):
        """Test handling of 401 unauthorized response."""
        with patch.object(self.client, '_request') as mock_request:
            mock_request.side_effect = AuthError("401 Unauthorized")
            
            with self.assertRaises(AuthError) as context:
                self.client.get_key_permissions()
                
            self.assertIn("401", str(context.exception))
            
    def test_invalid_api_key(self):
        """Test handling of invalid API key."""
        with patch.object(self.client, '_request') as mock_request:
            mock_request.side_effect = AuthError("Invalid API Key")
            
            with self.assertRaises(AuthError):
                self.client.get_key_permissions()


class TestPermissionValidation(unittest.TestCase):
    """Test permission validation logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = APIConfig(
            api_key="",
            api_secret="",
            passphrase="",
            base_url="https://api.coinbase.com",
            cdp_api_key="test_key",
            cdp_private_key="test_secret",
            api_mode='advanced',
            sandbox=False
        )
        
    def create_broker_with_permissions(self, permissions: Dict[str, Any]) -> CoinbaseBrokerage:
        """Helper to create a broker with mocked permissions."""
        with patch.object(CoinbaseClient, 'get_key_permissions') as mock_perms:
            mock_perms.return_value = permissions
            broker = CoinbaseBrokerage(self.config)
            return broker
            
    def test_validate_trading_permissions(self):
        """Test validation of trading permissions."""
        # Mock broker with trading permissions
        with patch.object(CoinbaseClient, 'get_key_permissions') as mock_perms:
            mock_perms.return_value = {
                "can_view": True,
                "can_trade": True,
                "can_transfer": False,
                "portfolio_uuid": "test-uuid",
                "portfolio_type": "DEFAULT"
            }
            
            broker = CoinbaseBrokerage(self.config)
            
            # Should be able to check permissions
            with patch.object(broker.client, 'get_key_permissions', return_value=mock_perms.return_value):
                perms = broker.client.get_key_permissions()
                self.assertTrue(perms["can_trade"], "Should have trading permission")
                
    def test_validate_insufficient_permissions(self):
        """Test detection of insufficient permissions for trading."""
        with patch.object(CoinbaseClient, 'get_key_permissions') as mock_perms:
            mock_perms.return_value = {
                "can_view": True,
                "can_trade": False,  # No trading permission
                "can_transfer": False,
                "portfolio_uuid": "test-uuid",
                "portfolio_type": "DEFAULT"
            }
            
            broker = CoinbaseBrokerage(self.config)
            
            with patch.object(broker.client, 'get_key_permissions', return_value=mock_perms.return_value):
                perms = broker.client.get_key_permissions()
                self.assertFalse(perms["can_trade"], "Should not have trading permission")
                self.assertTrue(perms["can_view"], "Should have view permission")


class TestPermissionCheckerUtility(unittest.TestCase):
    """Test permission checking utility functions."""
    
    @staticmethod
    def check_required_permissions(
        permissions: Dict[str, bool], 
        required: list[str]
    ) -> tuple[bool, list[str]]:
        """
        Check if required permissions are granted.
        
        Args:
            permissions: Dictionary of permission flags
            required: List of required permission names
            
        Returns:
            Tuple of (all_granted, missing_permissions)
        """
        missing = []
        for perm in required:
            if not permissions.get(perm, False):
                missing.append(perm)
        return len(missing) == 0, missing
        
    def test_permission_checker_all_granted(self):
        """Test permission checker when all required permissions are granted."""
        permissions = {
            "can_view": True,
            "can_trade": True,
            "can_transfer": True
        }
        
        # Test different requirement sets
        test_cases = [
            (["can_view"], "View only"),
            (["can_view", "can_trade"], "View and Trade"),
            (["can_view", "can_trade", "can_transfer"], "All permissions"),
        ]
        
        for required, description in test_cases:
            with self.subTest(required=description):
                all_granted, missing = self.check_required_permissions(permissions, required)
                self.assertTrue(all_granted, f"Should have {description}")
                self.assertEqual(missing, [], f"No permissions should be missing for {description}")
                
    def test_permission_checker_missing_permissions(self):
        """Test permission checker when some permissions are missing."""
        permissions = {
            "can_view": True,
            "can_trade": False,
            "can_transfer": False
        }
        
        # Check for trading permissions
        all_granted, missing = self.check_required_permissions(
            permissions, 
            ["can_view", "can_trade"]
        )
        self.assertFalse(all_granted, "Should not have all permissions")
        self.assertEqual(missing, ["can_trade"], "Should be missing trade permission")
        
        # Check for transfer permissions
        all_granted, missing = self.check_required_permissions(
            permissions,
            ["can_transfer"]
        )
        self.assertFalse(all_granted, "Should not have transfer permission")
        self.assertEqual(missing, ["can_transfer"], "Should be missing transfer permission")
        
    def test_permission_checker_empty_permissions(self):
        """Test permission checker with no permissions granted."""
        permissions = {
            "can_view": False,
            "can_trade": False,
            "can_transfer": False
        }
        
        all_granted, missing = self.check_required_permissions(
            permissions,
            ["can_view", "can_trade", "can_transfer"]
        )
        self.assertFalse(all_granted, "Should have no permissions")
        self.assertEqual(
            sorted(missing), 
            ["can_trade", "can_transfer", "can_view"],
            "Should be missing all permissions"
        )


class TestPortfolioPermissions(unittest.TestCase):
    """Test portfolio-specific permissions."""
    
    def test_portfolio_uuid_present(self):
        """Test that portfolio UUID is returned with permissions."""
        mock_auth = Mock(spec=CDPAuthV2)
        client = CoinbaseClient(
            base_url="https://api.coinbase.com",
            auth=mock_auth,
            api_mode="advanced"
        )
        
        with patch.object(client, '_request') as mock_request:
            mock_request.return_value = {
                "can_view": True,
                "can_trade": True,
                "can_transfer": False,
                "portfolio_uuid": "550e8400-e29b-41d4-a716-446655440000",
                "portfolio_type": "DEFAULT"
            }
            
            perms = client.get_key_permissions()
            
            self.assertIn("portfolio_uuid", perms)
            self.assertEqual(perms["portfolio_uuid"], "550e8400-e29b-41d4-a716-446655440000")
            self.assertEqual(perms["portfolio_type"], "DEFAULT")
            
    def test_portfolio_type_variations(self):
        """Test different portfolio types."""
        portfolio_types = ["DEFAULT", "CONSUMER", "INTX", "UNDEFINED"]
        
        mock_auth = Mock(spec=CDPAuthV2)
        client = CoinbaseClient(
            base_url="https://api.coinbase.com",
            auth=mock_auth,
            api_mode="advanced"
        )
        
        for portfolio_type in portfolio_types:
            with self.subTest(portfolio_type=portfolio_type):
                with patch.object(client, '_request') as mock_request:
                    mock_request.return_value = {
                        "can_view": True,
                        "can_trade": True,
                        "can_transfer": False,
                        "portfolio_uuid": "test-uuid",
                        "portfolio_type": portfolio_type
                    }
                    
                    perms = client.get_key_permissions()
                    self.assertEqual(perms["portfolio_type"], portfolio_type)


@pytest.mark.integration
class TestLivePermissionCheck(unittest.TestCase):
    """Integration tests for live permission checking (requires valid API keys)."""
    
    @pytest.mark.real_api
    def test_live_permission_check(self):
        """Test actual API permission check with live credentials."""
        import os
        from dotenv import load_dotenv

        load_dotenv()
        # TODO(2024-12-31): integrate with shared credential fixture so this can run in scheduled jobs.
        
        api_key = os.getenv("COINBASE_PROD_CDP_API_KEY")
        private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY")
        
        if not api_key or not private_key:
            self.skipTest("No live credentials available")
            
        auth = CDPAuthV2(
            api_key_name=api_key,
            private_key_pem=private_key,
            base_host="api.coinbase.com"
        )
        
        client = CoinbaseClient(
            base_url="https://api.coinbase.com",
            auth=auth,
            api_mode="advanced"
        )
        
        try:
            perms = client.get_key_permissions()
            
            # Verify response structure
            self.assertIn("can_view", perms)
            self.assertIn("can_trade", perms)
            self.assertIn("can_transfer", perms)
            self.assertIn("portfolio_uuid", perms)
            self.assertIn("portfolio_type", perms)
            
            # Log actual permissions for debugging
            print(f"\nLive API Permissions:")
            print(f"  can_view: {perms['can_view']}")
            print(f"  can_trade: {perms['can_trade']}")
            print(f"  can_transfer: {perms['can_transfer']}")
            print(f"  portfolio_type: {perms['portfolio_type']}")
            
        except Exception as e:
            self.fail(f"Failed to get permissions: {e}")


if __name__ == '__main__':
    unittest.main()
