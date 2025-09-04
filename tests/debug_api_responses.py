#!/usr/bin/env python3
"""
Debug script to check the exact API response formats
"""

import requests
import json
import sys

def debug_api_responses():
    """Debug the API response formats to understand the frontend integration issues"""
    base_url = "http://localhost:8000"
    
    print("🔍 Debugging API Response Formats...")
    
    # Test data
    test_data = {
        "investment_amount": 100000.0,
        "investment_type": "lumpsum",
        "tenure_years": 10,
        "risk_profile": "Moderate",
        "return_expectation": 8.0
    }
    
    print(f"Test Data: {json.dumps(test_data, indent=2)}")
    
    # Test 1: Portfolio allocation endpoint
    try:
        print("\n" + "="*60)
        print("1. PORTFOLIO ALLOCATION ENDPOINT")
        print("="*60)
        response = requests.post(f"{base_url}/api/portfolio/allocate", json=test_data, timeout=15)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Response Structure:")
            print(f"  success: {data.get('success')}")
            print(f"  message: {data.get('message')}")
            print(f"  data keys: {list(data.get('data', {}).keys())}")
            
            allocation_data = data.get('data', {})
            if 'allocation' in allocation_data:
                print(f"  allocation keys: {list(allocation_data['allocation'].keys())}")
                print(f"  allocation values: {allocation_data['allocation']}")
            
            if 'risk_metrics' in allocation_data:
                print(f"  risk_metrics: {allocation_data['risk_metrics']}")
                
            print(f"\nFull Response:")
            print(json.dumps(data, indent=2))
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 2: Investment calculation endpoint
    try:
        print("\n" + "="*60)
        print("2. INVESTMENT CALCULATION ENDPOINT")
        print("="*60)
        response = requests.post(f"{base_url}/api/investment/calculate", json=test_data, timeout=15)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Response Structure:")
            print(f"  success: {data.get('success')}")
            print(f"  message: {data.get('message')}")
            print(f"  data keys: {list(data.get('data', {}).keys())}")
            
            calc_data = data.get('data', {})
            if 'projections' in calc_data:
                projections = calc_data['projections']
                print(f"  projections type: {type(projections)}")
                print(f"  projections length: {len(projections) if isinstance(projections, list) else 'N/A'}")
                if isinstance(projections, list) and len(projections) > 0:
                    print(f"  first projection keys: {list(projections[0].keys()) if isinstance(projections[0], dict) else 'N/A'}")
                    print(f"  first projection: {projections[0]}")
                    print(f"  last projection: {projections[-1]}")
            
            if 'summary' in calc_data:
                print(f"  summary: {calc_data['summary']}")
                
            print(f"\nFull Response:")
            print(json.dumps(data, indent=2))
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 3: Test the old /api/portfolio/generate endpoint (if it exists)
    try:
        print("\n" + "="*60)
        print("3. OLD PORTFOLIO GENERATE ENDPOINT (if exists)")
        print("="*60)
        response = requests.post(f"{base_url}/api/portfolio/generate", json=test_data, timeout=15)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Response Structure:")
            print(f"  success: {data.get('success')}")
            print(f"  message: {data.get('message')}")
            print(f"  data keys: {list(data.get('data', {}).keys())}")
            print(f"\nFull Response:")
            print(json.dumps(data, indent=2))
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    try:
        debug_api_responses()
    except KeyboardInterrupt:
        print("\n🛑 Debug interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Debug failed with error: {e}")
        sys.exit(1)