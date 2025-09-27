#!/usr/bin/env python3
"""
Complete fix for all remaining hardcoded endpoints in client.py

This script generates the complete fixes needed.
Run this to see what needs to be changed, then apply manually or via patch.
"""

print("""
COMPLETE ENDPOINT ROUTING FIX
==============================

Step 1: Add missing endpoints to ENDPOINT_MAP in _get_endpoint_path
--------------------------------------------------------------------

Add these to the 'advanced' section of ENDPOINT_MAP:

                'order_edit_preview': '/api/v3/brokerage/orders/edit_preview',
                'move_funds': '/api/v3/brokerage/portfolios/move_funds',
                'orders_batch': '/api/v3/brokerage/orders/historical/batch',
                # INTX (derivatives)
                'intx_allocate': '/api/v3/brokerage/intx/allocate',
                'intx_balances': '/api/v3/brokerage/intx/balances/{portfolio_uuid}',
                'intx_portfolio': '/api/v3/brokerage/intx/portfolio/{portfolio_uuid}',
                'intx_positions': '/api/v3/brokerage/intx/positions/{portfolio_uuid}',
                'intx_position': '/api/v3/brokerage/intx/positions/{portfolio_uuid}/{symbol}',
                'intx_multi_asset_collateral': '/api/v3/brokerage/intx/multi_asset_collateral',
                # CFM
                'cfm_balance_summary': '/api/v3/brokerage/cfm/balance_summary',
                'cfm_positions': '/api/v3/brokerage/cfm/positions',
                'cfm_position': '/api/v3/brokerage/cfm/positions/{product_id}',
                'cfm_sweeps': '/api/v3/brokerage/cfm/sweeps',
                'cfm_schedule_sweep': '/api/v3/brokerage/cfm/sweeps/schedule',

Step 2: Fix remaining methods
------------------------------
""")

# Generate fixes for each method
fixes = [
    ("list_orders_batch", None, "orders_batch"),
    ("preview_order", None, "order_preview"),
    ("edit_order_preview", None, "order_edit_preview"),
    ("edit_order", None, "order_edit"),
    ("close_position", None, "close_position"),
    ("list_payment_methods", None, "payment_methods"),
    ("get_payment_method", "payment_method_id", "payment_method"),
    ("list_portfolios", None, "portfolios"),
    ("get_portfolio", "portfolio_uuid", "portfolio"),
    ("get_portfolio_breakdown", "portfolio_uuid", "portfolio_breakdown"),
    ("move_funds", None, "move_funds"),
    ("intx_allocate", None, "intx_allocate"),
    ("intx_balances", "portfolio_uuid", "intx_balances"),
    ("intx_portfolio", "portfolio_uuid", "intx_portfolio"),
    ("intx_positions", "portfolio_uuid", "intx_positions"),
    ("intx_position", "portfolio_uuid,symbol", "intx_position"),
    ("intx_multi_asset_collateral", None, "intx_multi_asset_collateral"),
    ("cfm_balance_summary", None, "cfm_balance_summary"),
    ("cfm_positions", None, "cfm_positions"),
    ("cfm_position", "product_id", "cfm_position"),
    ("cfm_sweeps", None, "cfm_sweeps"),
    ("cfm_schedule_sweep", None, "cfm_schedule_sweep"),
]

for method_name, params, endpoint_name in fixes:
    if params:
        param_list = params.split(',')
        param_args = ", ".join(f"{p}={p}" for p in param_list)
        print(f"""
    def {method_name}(self, {params.replace(',', ', ')}...):
        path = self._get_endpoint_path('{endpoint_name}', {param_args})
        return self._request(..., path, ...)
""")
    else:
        print(f"""
    def {method_name}(self, ...):
        path = self._get_endpoint_path('{endpoint_name}')
        return self._request(..., path, ...)
""")

print("""
Step 3: Add InvalidRequestError for exchange-unsupported methods
-----------------------------------------------------------------

For methods that don't exist in exchange mode, add this check at the start:
""")

unsupported_in_exchange = [
    'get_key_permissions', 'get_limits', 'convert_quote', 'get_convert_trade',
    'preview_order', 'edit_order_preview', 'edit_order', 'close_position',
    'list_payment_methods', 'get_payment_method',
    'list_portfolios', 'get_portfolio', 'get_portfolio_breakdown', 'move_funds',
    'intx_allocate', 'intx_balances', 'intx_portfolio', 'intx_positions',
    'intx_position', 'intx_multi_asset_collateral',
    'cfm_balance_summary', 'cfm_positions', 'cfm_position', 'cfm_sweeps', 'cfm_schedule_sweep'
]

for method in unsupported_in_exchange:
    print(f"""
    def {method}(self, ...):
        if self.api_mode == 'exchange':
            from .errors import InvalidRequestError
            raise InvalidRequestError(
                f"{method} not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path(...)
        return self._request(...)
""")

print("""
Step 4: Test each method
------------------------

Create tests that verify:
1. In advanced mode: correct path is used
2. In exchange mode: either works with legacy path OR raises InvalidRequestError
""")