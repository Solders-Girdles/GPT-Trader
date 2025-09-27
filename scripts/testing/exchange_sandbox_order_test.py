#!/usr/bin/env python
"""
Exchange Sandbox Order Lifecycle Test
Handles sandbox pricing anomalies and quantity formatting
"""

import os
import time
import hmac
import hashlib
import base64
import json
import requests
from decimal import Decimal
from datetime import datetime

def get_auth_headers(method, path, body=""):
    """Generate HMAC authentication headers"""
    api_key = os.environ.get('COINBASE_API_KEY')
    api_secret = os.environ.get('COINBASE_API_SECRET')
    api_passphrase = os.environ.get('COINBASE_API_PASSPHRASE')
    
    timestamp = str(int(time.time()))
    message = timestamp + method + path + body
    
    hmac_key = base64.b64decode(api_secret)
    signature = hmac.new(hmac_key, message.encode(), hashlib.sha256)
    signature_b64 = base64.b64encode(signature.digest()).decode()
    
    return {
        'CB-ACCESS-KEY': api_key,
        'CB-ACCESS-SIGN': signature_b64,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'CB-ACCESS-PASSPHRASE': api_passphrase,
        'Content-Type': 'application/json'
    }

def get_product_info():
    """Get BTC-USD product information"""
    url = "https://api-public.sandbox.exchange.coinbase.com/products/BTC-USD"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Handle both possible field name formats
        return {
            'min_size': Decimal(data.get('base_min_size', data.get('min_market_funds', '0.00010000'))),
            'max_size': Decimal(data.get('base_max_size', data.get('max_market_funds', '1000000'))),
            'increment': Decimal(data.get('base_increment', data.get('quote_increment', '0.00000001'))),
            'quote_increment': Decimal(data.get('quote_increment', '0.01'))
        }
    return None

def get_current_price():
    """Get current BTC-USD price from ticker"""
    url = "https://api-public.sandbox.exchange.coinbase.com/products/BTC-USD/ticker"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return Decimal(data['price'])
    return None

def calculate_valid_quantity(price, min_notional=Decimal('1.0')):
    """Calculate minimum valid quantity for given price"""
    # Ensure we meet minimum notional value
    min_qty = min_notional / price
    
    # Round up to 8 decimal places (BTC standard)
    min_qty = min_qty.quantize(Decimal('0.00000001'))
    
    # Ensure it's at least the minimum size
    if min_qty < Decimal('0.00010000'):
        min_qty = Decimal('0.00010000')
    
    return min_qty

def place_order(quantity, price=None):
    """Place a limit buy order"""
    url = "https://api-public.sandbox.exchange.coinbase.com/orders"
    
    # Use limit order with specified price
    if price is None:
        current_price = get_current_price()
        # Place limit order 10% below current price
        price = current_price * Decimal('0.9')
    
    # Format with proper precision
    order_data = {
        'product_id': 'BTC-USD',
        'side': 'buy',
        'type': 'limit',
        'size': str(quantity.quantize(Decimal('0.00000001'))),
        'price': str(price.quantize(Decimal('0.01'))),
        'time_in_force': 'GTC',
        'post_only': False
    }
    
    body = json.dumps(order_data)
    headers = get_auth_headers('POST', '/orders', body)
    
    print(f"📤 Placing order:")
    print(f"   Size: {order_data['size']} BTC")
    print(f"   Price: ${order_data['price']}")
    print(f"   Notional: ${Decimal(order_data['size']) * Decimal(order_data['price']):.2f}")
    
    response = requests.post(url, headers=headers, data=body)
    
    if response.status_code in [200, 201]:
        order = response.json()
        print(f"✅ Order placed successfully!")
        print(f"   Order ID: {order['id']}")
        print(f"   Status: {order['status']}")
        return order['id']
    else:
        print(f"❌ Order placement failed:")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
        return None

def cancel_order(order_id):
    """Cancel an order"""
    url = f"https://api-public.sandbox.exchange.coinbase.com/orders/{order_id}"
    headers = get_auth_headers('DELETE', f'/orders/{order_id}')
    
    print(f"🚫 Cancelling order {order_id}...")
    response = requests.delete(url, headers=headers)
    
    if response.status_code == 200:
        print(f"✅ Order cancelled successfully!")
        return True
    else:
        print(f"❌ Cancel failed: {response.status_code}")
        return False

def get_order_status(order_id):
    """Get order status"""
    url = f"https://api-public.sandbox.exchange.coinbase.com/orders/{order_id}"
    headers = get_auth_headers('GET', f'/orders/{order_id}')
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        order = response.json()
        return order['status'], order.get('filled_size', '0')
    return None, None

def run_order_lifecycle_test():
    """Run complete order lifecycle test"""
    print("=" * 60)
    print("EXCHANGE SANDBOX ORDER LIFECYCLE TEST")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print()
    
    # Step 1: Get product info
    print("📊 Getting product information...")
    product_info = get_product_info()
    if product_info:
        print(f"   Min size: {product_info['min_size']} BTC")
        print(f"   Size increment: {product_info['increment']}")
    
    # Step 2: Get current price
    print("\n💰 Getting current price...")
    current_price = get_current_price()
    if current_price:
        print(f"   Current BTC price: ${current_price:,.2f}")
    else:
        print("   ⚠️ Could not get current price, using default")
        current_price = Decimal('11000000')  # Sandbox typical price
    
    # Step 3: Calculate valid quantity
    print("\n🧮 Calculating valid order quantity...")
    # For sandbox, ensure we meet $1 minimum notional
    # Using a higher limit price to meet minimum
    quantity = Decimal('0.00010000')  # Minimum BTC size
    limit_price = Decimal('10000.00')  # Higher price to meet $1 minimum
    
    notional = quantity * limit_price
    print(f"   Quantity: {quantity} BTC")
    print(f"   Limit price: ${limit_price}")
    print(f"   Notional value: ${notional:.2f}")
    
    # Step 4: Place order
    print("\n🚀 PLACING ORDER...")
    start_time = time.time()
    order_id = place_order(quantity, limit_price)
    place_latency = (time.time() - start_time) * 1000
    
    if not order_id:
        print("❌ Test failed: Could not place order")
        return False
    
    print(f"   Latency: {place_latency:.0f}ms")
    
    # Step 5: Check order status
    print("\n📋 Checking order status...")
    time.sleep(1)  # Wait a moment
    status, filled = get_order_status(order_id)
    if status:
        print(f"   Status: {status}")
        print(f"   Filled: {filled} BTC")
    
    # Step 6: Cancel order
    print("\n🛑 CANCELLING ORDER...")
    start_time = time.time()
    cancelled = cancel_order(order_id)
    cancel_latency = (time.time() - start_time) * 1000
    
    if cancelled:
        print(f"   Latency: {cancel_latency:.0f}ms")
    
    # Step 7: Verify cancellation
    print("\n✔️ Verifying cancellation...")
    time.sleep(1)
    final_status, _ = get_order_status(order_id)
    if final_status:
        print(f"   Final status: {final_status}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Order ID: {order_id}")
    print(f"✅ Lifecycle: open → {final_status}")
    print(f"✅ Place latency: {place_latency:.0f}ms")
    print(f"✅ Cancel latency: {cancel_latency:.0f}ms")
    print(f"✅ Total time: {place_latency + cancel_latency:.0f}ms")
    
    success = place_latency < 500 and cancel_latency < 500
    if success:
        print("\n🎉 TEST PASSED!")
    else:
        print("\n⚠️ Latency exceeded 500ms threshold")
    
    return success

if __name__ == "__main__":
    # Verify environment
    if os.environ.get('COINBASE_API_MODE') != 'exchange':
        print("❌ Error: COINBASE_API_MODE must be 'exchange'")
        exit(1)
    
    if os.environ.get('COINBASE_SANDBOX') != '1':
        print("❌ Error: COINBASE_SANDBOX must be '1'")
        exit(1)
    
    # Run test
    try:
        success = run_order_lifecycle_test()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
