#!/usr/bin/env python3
"""
Script to generate the comprehensive endpoint fixes for client.py

This generates the updated endpoint map and method implementations.
"""

# Complete endpoint mappings for both modes
ENDPOINT_MAP = {
    'advanced': {
        # Market data
        'products': '/api/v3/brokerage/market/products',
        'product': '/api/v3/brokerage/market/products/{product_id}',
        'ticker': '/api/v3/brokerage/market/products/{product_id}/ticker',
        'candles': '/api/v3/brokerage/market/products/{product_id}/candles',
        'order_book': '/api/v3/brokerage/market/product_book',
        'best_bid_ask': '/api/v3/brokerage/best_bid_ask',
        
        # Account
        'accounts': '/api/v3/brokerage/accounts',
        'account': '/api/v3/brokerage/accounts/{account_uuid}',
        
        # Orders
        'orders': '/api/v3/brokerage/orders',
        'order': '/api/v3/brokerage/orders/historical/{order_id}',
        'orders_historical': '/api/v3/brokerage/orders/historical',
        'orders_batch_cancel': '/api/v3/brokerage/orders/batch_cancel',
        'orders_batch': '/api/v3/brokerage/orders/historical/batch',
        'order_preview': '/api/v3/brokerage/orders/preview',
        'order_edit_preview': '/api/v3/brokerage/orders/edit_preview',
        'order_edit': '/api/v3/brokerage/orders/edit',
        'close_position': '/api/v3/brokerage/orders/close_position',
        
        # Fills
        'fills': '/api/v3/brokerage/orders/historical/fills',
        
        # System
        'time': '/api/v3/brokerage/time',
        'fees': '/api/v3/brokerage/fees',
        'limits': '/api/v3/brokerage/limits',
        'key_permissions': '/api/v3/brokerage/key_permissions',
        
        # Portfolios
        'portfolios': '/api/v3/brokerage/portfolios',
        'portfolio': '/api/v3/brokerage/portfolios/{portfolio_uuid}',
        'portfolio_breakdown': '/api/v3/brokerage/portfolios/{portfolio_uuid}/breakdown',
        'move_funds': '/api/v3/brokerage/portfolios/move_funds',
        
        # Convert
        'convert_quote': '/api/v3/brokerage/convert/quote',
        'convert_trade': '/api/v3/brokerage/convert/trade/{trade_id}',
        
        # Payment methods
        'payment_methods': '/api/v3/brokerage/payment_methods',
        'payment_method': '/api/v3/brokerage/payment_methods/{payment_method_id}',
        
        # INTX (derivatives)
        'intx_allocate': '/api/v3/brokerage/intx/allocate',
        'intx_balances': '/api/v3/brokerage/intx/balances/{portfolio_uuid}',
        'intx_portfolio': '/api/v3/brokerage/intx/portfolio/{portfolio_uuid}',
        'intx_positions': '/api/v3/brokerage/intx/positions/{portfolio_uuid}',
        'intx_position': '/api/v3/brokerage/intx/positions/{portfolio_uuid}/{symbol}',
        'intx_multi_asset_collateral': '/api/v3/brokerage/intx/multi_asset_collateral',
        
        # CFM (Cross-Functional Margin)
        'cfm_balance_summary': '/api/v3/brokerage/cfm/balance_summary',
        'cfm_positions': '/api/v3/brokerage/cfm/positions',
        'cfm_position': '/api/v3/brokerage/cfm/positions/{product_id}',
        'cfm_sweeps': '/api/v3/brokerage/cfm/sweeps',
        'cfm_schedule_sweep': '/api/v3/brokerage/cfm/sweeps/schedule',
    },
    'exchange': {
        # Market data
        'products': '/products',
        'product': '/products/{product_id}',
        'ticker': '/products/{product_id}/ticker',
        'candles': '/products/{product_id}/candles',
        'order_book': '/products/{product_id}/book',
        'trades': '/products/{product_id}/trades',
        'stats': '/products/{product_id}/stats',
        
        # Account
        'accounts': '/accounts',
        'account': '/accounts/{account_id}',
        'account_history': '/accounts/{account_id}/ledger',
        'account_holds': '/accounts/{account_id}/holds',
        
        # Orders
        'orders': '/orders',
        'order': '/orders/{order_id}',
        'cancel_order': '/orders/{order_id}',
        'cancel_all': '/orders',
        
        # Fills
        'fills': '/fills',
        
        # System
        'time': '/time',
        'fees': '/fees',
        'currencies': '/currencies',
        'currency': '/currencies/{currency_id}',
        
        # Transfers (Exchange mode specific)
        'transfers': '/transfers',
        'transfer': '/transfers/{transfer_id}',
        
        # Reports
        'reports': '/reports',
        'report': '/reports/{report_id}',
        
        # Note: Many advanced features not available in exchange mode
    }
}

# Methods that need updating
METHODS_TO_UPDATE = [
    ('get_time', 'time', {}),
    ('get_key_permissions', 'key_permissions', {}),
    ('cancel_orders', 'orders_batch_cancel', {}),
    ('get_order_historical', 'order', {'order_id': 'order_id'}),
    ('list_orders', 'orders_historical', {}),
    ('list_orders_batch', 'orders_batch', {}),
    ('list_fills', 'fills', {}),
    ('get_fees', 'fees', {}),
    ('get_limits', 'limits', {}),
    ('convert_quote', 'convert_quote', {}),
    ('get_convert_trade', 'convert_trade', {'trade_id': 'trade_id'}),
    ('preview_order', 'order_preview', {}),
    ('edit_order_preview', 'order_edit_preview', {}),
    ('edit_order', 'order_edit', {}),
    ('close_position', 'close_position', {}),
    ('list_payment_methods', 'payment_methods', {}),
    ('get_payment_method', 'payment_method', {'payment_method_id': 'payment_method_id'}),
    ('list_portfolios', 'portfolios', {}),
    ('get_portfolio', 'portfolio', {'portfolio_uuid': 'portfolio_uuid'}),
    ('get_portfolio_breakdown', 'portfolio_breakdown', {'portfolio_uuid': 'portfolio_uuid'}),
    ('move_funds', 'move_funds', {}),
    ('intx_allocate', 'intx_allocate', {}),
    ('intx_balances', 'intx_balances', {'portfolio_uuid': 'portfolio_uuid'}),
    ('intx_portfolio', 'intx_portfolio', {'portfolio_uuid': 'portfolio_uuid'}),
    ('intx_positions', 'intx_positions', {'portfolio_uuid': 'portfolio_uuid'}),
    ('intx_position', 'intx_position', {'portfolio_uuid': 'portfolio_uuid', 'symbol': 'symbol'}),
    ('intx_multi_asset_collateral', 'intx_multi_asset_collateral', {}),
    ('cfm_balance_summary', 'cfm_balance_summary', {}),
    ('cfm_positions', 'cfm_positions', {}),
    ('cfm_position', 'cfm_position', {'product_id': 'product_id'}),
    ('cfm_sweeps', 'cfm_sweeps', {}),
    ('cfm_schedule_sweep', 'cfm_schedule_sweep', {}),
]

def generate_updated_endpoint_map():
    """Generate the updated _get_endpoint_path method."""
    print("Updated _get_endpoint_path method:")
    print("=" * 60)
    print('''
    def _get_endpoint_path(self, endpoint_name: str, **kwargs) -> str:
        """Get the correct endpoint path based on API mode.
        
        Args:
            endpoint_name: Name of the endpoint (e.g., 'products', 'accounts')
            **kwargs: Additional parameters for path formatting
            
        Returns:
            The correct path for the current API mode
            
        Raises:
            InvalidRequestError: If endpoint not supported in current mode
        """
        from .errors import InvalidRequestError
        
        # Complete endpoint mappings for both modes
        ENDPOINT_MAP = {
            'advanced': {''')
    
    # Print advanced endpoints
    for key, path in ENDPOINT_MAP['advanced'].items():
        print(f"                '{key}': '{path}',")
    
    print('''            },
            'exchange': {''')
    
    # Print exchange endpoints
    for key, path in ENDPOINT_MAP['exchange'].items():
        print(f"                '{key}': '{path}',")
    
    print('''            }
        }
        
        if self.api_mode not in ENDPOINT_MAP:
            raise InvalidRequestError(f"Unknown API mode: {self.api_mode}")
        
        mode_endpoints = ENDPOINT_MAP[self.api_mode]
        
        if endpoint_name not in mode_endpoints:
            # Check if endpoint exists in other mode
            available_modes = [mode for mode in ENDPOINT_MAP if endpoint_name in ENDPOINT_MAP[mode]]
            if available_modes:
                raise InvalidRequestError(
                    f"Endpoint '{endpoint_name}' not available in {self.api_mode} mode. "
                    f"Available in: {', '.join(available_modes)}. "
                    f"Set COINBASE_API_MODE={available_modes[0]} to use this endpoint."
                )
            else:
                raise InvalidRequestError(f"Unknown endpoint: {endpoint_name}")
        
        path = mode_endpoints[endpoint_name]
        
        # Format path with provided kwargs
        if kwargs:
            path = path.format(**kwargs)
        
        return path
''')

def generate_updated_methods():
    """Generate updated method implementations."""
    print("\nUpdated method implementations:")
    print("=" * 60)
    
    for method_name, endpoint_name, params in METHODS_TO_UPDATE:
        param_args = ", ".join(f"{k}={k}" for k in params.values()) if params else ""
        
        # Check if endpoint exists in exchange mode
        exchange_supported = endpoint_name in ENDPOINT_MAP['exchange']
        
        if not exchange_supported:
            # Method should check mode and raise error if in exchange mode
            print(f'''
    def {method_name}(self, ...):  # Keep original signature
        """..."""  # Keep original docstring
        if self.api_mode == "exchange":
            from .errors import InvalidRequestError
            raise InvalidRequestError(
                "'{method_name}' not available in exchange mode. "
                "Switch to advanced mode to use this feature."
            )
        path = self._get_endpoint_path('{endpoint_name}'{", " + param_args if param_args else ""})
        # ... rest of implementation
''')
        else:
            # Method works in both modes
            print(f'''
    def {method_name}(self, ...):  # Keep original signature
        """..."""  # Keep original docstring
        path = self._get_endpoint_path('{endpoint_name}'{", " + param_args if param_args else ""})
        # ... rest of implementation
''')

if __name__ == "__main__":
    generate_updated_endpoint_map()
    generate_updated_methods()
    
    print("\n" + "=" * 60)
    print("Summary of changes needed:")
    print("=" * 60)
    print(f"1. Update _get_endpoint_path with {len(ENDPOINT_MAP['advanced'])} advanced endpoints")
    print(f"2. Update _get_endpoint_path with {len(ENDPOINT_MAP['exchange'])} exchange endpoints")
    print(f"3. Update {len(METHODS_TO_UPDATE)} methods to use _get_endpoint_path")
    print(f"4. Add mode checks for endpoints not available in exchange mode")
    
    # List methods that won't work in exchange mode
    exchange_only = []
    for method_name, endpoint_name, _ in METHODS_TO_UPDATE:
        if endpoint_name not in ENDPOINT_MAP['exchange']:
            exchange_only.append(method_name)
    
    if exchange_only:
        print(f"\nMethods that need exchange mode checks ({len(exchange_only)}):")
        for method in exchange_only:
            print(f"  - {method}")