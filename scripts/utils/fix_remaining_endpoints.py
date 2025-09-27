#!/usr/bin/env python3
"""
Script to fix all remaining hardcoded endpoints in client.py
"""

import re

# Read the current client.py
with open('/Users/rj/PycharmProjects/GPT-Trader/src/bot_v2/features/brokerages/coinbase/client.py', 'r') as f:
    content = f.read()

# Find all methods with hardcoded /api/v3/brokerage paths
pattern = r'return self\._request\([^,]+,\s*["\']\/api\/v3\/brokerage[^"\']*["\'](.*?)\)'
matches = re.findall(pattern, content)

print(f"Found {len(matches)} hardcoded paths remaining")

# Methods that still need fixing (from line 435 onwards)
methods_to_fix = [
    # Order management
    ('preview_order', 'order_preview'),
    ('edit_order_preview', 'order_edit_preview'), 
    ('edit_order', 'order_edit'),
    ('close_position', 'close_position'),
    
    # Payment methods
    ('list_payment_methods', 'payment_methods'),
    ('get_payment_method', 'payment_method'),
    
    # Portfolios
    ('list_portfolios', 'portfolios'),
    ('get_portfolio', 'portfolio'),
    ('get_portfolio_breakdown', 'portfolio_breakdown'),
    ('move_funds', 'move_funds'),
    
    # System info (still needs fixing from earlier)
    ('list_orders_batch', 'orders_batch'),
    
    # INTX (derivatives) - NOT in exchange mode
    ('intx_allocate', 'intx_allocate'),
    ('intx_balances', 'intx_balances'),
    ('intx_portfolio', 'intx_portfolio'),
    ('intx_positions', 'intx_positions'),
    ('intx_position', 'intx_position'),
    ('intx_multi_asset_collateral', 'intx_multi_asset_collateral'),
    
    # CFM - NOT in exchange mode
    ('cfm_balance_summary', 'cfm_balance_summary'),
    ('cfm_positions', 'cfm_positions'),
    ('cfm_position', 'cfm_position'),
    ('cfm_sweeps', 'cfm_sweeps'),
    ('cfm_schedule_sweep', 'cfm_schedule_sweep'),
]

# Methods that need exchange mode checks
exchange_unsupported = [
    'key_permissions', 'limits', 'convert_quote', 'get_convert_trade',
    'preview_order', 'edit_order_preview', 'edit_order', 'close_position',
    'payment_methods', 'get_payment_method', 
    'portfolios', 'get_portfolio', 'get_portfolio_breakdown', 'move_funds',
    'intx_allocate', 'intx_balances', 'intx_portfolio', 'intx_positions', 
    'intx_position', 'intx_multi_asset_collateral',
    'cfm_balance_summary', 'cfm_positions', 'cfm_position', 'cfm_sweeps', 'cfm_schedule_sweep'
]

print("\nMethods needing exchange mode checks:")
for method in exchange_unsupported:
    print(f"  - {method}")

print("\n" + "="*60)
print("Add these endpoints to ENDPOINT_MAP in _get_endpoint_path:")
print("="*60)

# Additional endpoints needed in map
additional_endpoints = {
    'orders_batch': '/api/v3/brokerage/orders/historical/batch',
    'order_edit_preview': '/api/v3/brokerage/orders/edit_preview',
    'move_funds': '/api/v3/brokerage/portfolios/move_funds',
    'intx_allocate': '/api/v3/brokerage/intx/allocate',
    'intx_balances': '/api/v3/brokerage/intx/balances/{portfolio_uuid}',
    'intx_portfolio': '/api/v3/brokerage/intx/portfolio/{portfolio_uuid}',
    'intx_positions': '/api/v3/brokerage/intx/positions/{portfolio_uuid}',
    'intx_position': '/api/v3/brokerage/intx/positions/{portfolio_uuid}/{symbol}',
    'intx_multi_asset_collateral': '/api/v3/brokerage/intx/multi_asset_collateral',
    'cfm_balance_summary': '/api/v3/brokerage/cfm/balance_summary',
    'cfm_positions': '/api/v3/brokerage/cfm/positions',
    'cfm_position': '/api/v3/brokerage/cfm/positions/{product_id}',
    'cfm_sweeps': '/api/v3/brokerage/cfm/sweeps',
    'cfm_schedule_sweep': '/api/v3/brokerage/cfm/sweeps/schedule',
}

for key, path in additional_endpoints.items():
    print(f"                '{key}': '{path}',")