#!/usr/bin/env python3
"""
Test the allocation endpoint specifically
"""

import requests
import json

def test_allocation_endpoint():
    """Test the allocation endpoint that the frontend will use"""
    base_url = "http://localhost:8000"
    
    test_data = {
        "investment_amount": 100000.0,
        "investment_type": "lumpsum",
        "tenure_years": 10,
        "risk_profile": "Moderate",
        "return_expectation": 8.0
    }
    
    print("🔍 Testing Portfolio Allocation Endpoint...")
    print(f"Request: {json.dumps(test_data, indent=2)}")
    
    try:
        response = requests.post(f"{base_url}/api/portfolio/allocate", json=test_data, timeout=15)
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS!")
            
            allocation_data = data.get('data', {})
            allocation = allocation_data.get('allocation', {})
            
            print(f"\n📊 Detailed Allocation:")
            for asset, percentage in allocation.items():
                print(f"  {asset}: {percentage}%")
            
            # Calculate bonds total (what frontend will do)
            bonds_total = (allocation.get('t_bills', 0) + 
                          allocation.get('t_bonds', 0) + 
                          allocation.get('corporate_bonds', 0))
            
            print(f"\n📊 Frontend Aggregated Allocation:")
            print(f"  sp500: {allocation.get('sp500', 0)}%")
            print(f"  small_cap: {allocation.get('small_cap', 0)}%")
            print(f"  bonds: {bonds_total}% (aggregated)")
            print(f"  real_estate: {allocation.get('real_estate', 0)}%")
            print(f"  gold: {allocation.get('gold', 0)}%")
            
            # Verify total is 100%
            frontend_total = (allocation.get('sp500', 0) + 
                            allocation.get('small_cap', 0) + 
                            bonds_total + 
                            allocation.get('real_estate', 0) + 
                            allocation.get('gold', 0))
            
            print(f"\n✅ Total allocation: {frontend_total}%")
            
            if abs(frontend_total - 100.0) < 0.01:
                print("✅ Allocation sums to 100% - Perfect!")
            else:
                print("⚠️ Allocation doesn't sum to 100%")
                
        else:
            print(f"❌ FAILED!")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    test_allocation_endpoint()