#!/usr/bin/env python3
"""Diagnose why perpetuals products aren't being found."""

import json
import os
from decimal import Decimal
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage

def main():
    print("=" * 70)
    print("🔍 Perpetuals Product Discovery Diagnostic")
    print("=" * 70)
    
    # Initialize the brokerage with None config (will use defaults)
    broker = CoinbaseBrokerage(config=None)
    
    print("\n1. Fetching all products from Coinbase API...")
    try:
        # Get raw response from API
        response = broker.client.get_products() or {}
        total_products = len(response.get("products", []))
        print(f"   Total products returned: {total_products}")
        
        if total_products > 0:
            # Check first product structure
            first_product = response["products"][0]
            print("\n2. Sample product structure (first product):")
            print(f"   Product ID: {first_product.get('product_id')}")
            print(f"   Available fields: {list(first_product.keys())[:10]}...")
            
            # Look for products with PERP in the name
            perp_by_name = [
                p for p in response["products"] 
                if "PERP" in p.get("product_id", "").upper()
            ]
            print(f"\n3. Products with 'PERP' in product_id: {len(perp_by_name)}")
            
            if perp_by_name:
                print("\n   Perpetual products found by name:")
                for p in perp_by_name[:5]:
                    print(f"   - {p.get('product_id')}")
                    
                print("\n4. Examining perpetual product structure:")
                perp_sample = perp_by_name[0]
                relevant_fields = {
                    "product_id": perp_sample.get("product_id"),
                    "product_type": perp_sample.get("product_type"),
                    "contract_type": perp_sample.get("contract_type"),
                    "future_product_details": perp_sample.get("future_product_details"),
                    "contract_display_name": perp_sample.get("contract_display_name"),
                    "trading_disabled": perp_sample.get("trading_disabled"),
                    "status": perp_sample.get("status"),
                }
                print(json.dumps(relevant_fields, indent=2))
                
                # Check if future_product_details contains contract info
                if perp_sample.get("future_product_details"):
                    print("\n5. Future product details:")
                    print(json.dumps(perp_sample["future_product_details"], indent=2))
            
            # Look for products by product_type
            futures_by_type = [
                p for p in response["products"]
                if p.get("product_type") == "FUTURE"
            ]
            print(f"\n6. Products with product_type='FUTURE': {len(futures_by_type)}")
            
            if futures_by_type:
                print("   First 5 futures products:")
                for p in futures_by_type[:5]:
                    print(f"   - {p.get('product_id')}")
            
            # Check what product types exist
            product_types = set(p.get("product_type") for p in response["products"] if p.get("product_type"))
            print(f"\n7. All product_type values found: {product_types}")
            
    except Exception as e:
        print(f"   Error fetching products: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n8. Testing list_products method...")
    try:
        from bot_v2.features.brokerages.core.interfaces import MarketType
        
        # Test listing all products
        all_products = broker.list_products()
        print(f"   Total products from list_products(): {len(all_products)}")
        
        # Test filtering for perpetuals
        perp_products = broker.list_products(market=MarketType.PERPETUAL)
        print(f"   Perpetual products from list_products(PERPETUAL): {len(perp_products)}")
        
        if perp_products:
            print("\n   Found perpetuals:")
            for p in perp_products[:5]:
                print(f"   - {p.symbol}: base={p.base_asset}, quote={p.quote_asset}, type={p.market_type}")
        
    except Exception as e:
        print(f"   Error in list_products: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Diagnostic complete")


if __name__ == "__main__":
    main()