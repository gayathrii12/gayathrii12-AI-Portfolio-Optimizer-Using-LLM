#!/usr/bin/env python3
"""
Quick test script to check if the backend API is working
"""

import requests
import json
import sys

def test_backend_connection():
    """Test basic backend connectivity and endpoints"""
    base_url = "http://localhost:8000"
    
    print("🔍 Testing Backend API Connection...")
    print(f"Base URL: {base_url}")
    
    # Test 1: Root endpoint
    try:
        print("\n1. Testing root endpoint...")
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Message: {data.get('message', 'N/A')}")
            print(f"   Pipeline Status: {data.get('pipeline_status', 'N/A')}")
            print(f"   Agents Active: {data.get('agents_active', 'N/A')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Test 2: Agent status
    try:
        print("\n2. Testing agent status endpoint...")
        response = requests.get(f"{base_url}/api/agent-status", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Success: {data.get('success', 'N/A')}")
            print(f"   Pipeline Status: {data.get('data', {}).get('pipeline_status', 'N/A')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 3: Dashboard endpoint
    try:
        print("\n3. Testing dashboard endpoint...")
        response = requests.get(f"{base_url}/api/dashboard", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Success: {data.get('success', 'N/A')}")
            print(f"   System Status: {data.get('data', {}).get('system_status', 'N/A')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 4: Portfolio allocation endpoint (new Task 9 endpoint)
    try:
        print("\n4. Testing portfolio allocation endpoint...")
        test_data = {
            "investment_amount": 100000.0,
            "investment_type": "lumpsum",
            "tenure_years": 10,
            "risk_profile": "Moderate",
            "return_expectation": 8.0
        }
        response = requests.post(f"{base_url}/api/portfolio/allocate", json=test_data, timeout=15)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Success: {data.get('success', 'N/A')}")
            allocation = data.get('data', {}).get('allocation', {})
            print(f"   Sample Allocation: SP500={allocation.get('sp500', 'N/A')}%, Bonds={allocation.get('t_bonds', 'N/A')}%")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 5: Investment calculation endpoint (new Task 9 endpoint)
    try:
        print("\n5. Testing investment calculation endpoint...")
        test_data = {
            "investment_amount": 50000.0,
            "investment_type": "lumpsum",
            "tenure_years": 5,
            "risk_profile": "Moderate",
            "return_expectation": 8.0
        }
        response = requests.post(f"{base_url}/api/investment/calculate", json=test_data, timeout=15)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Success: {data.get('success', 'N/A')}")
            projections = data.get('data', {}).get('projections', [])
            if projections:
                final_value = projections[-1].get('portfolio_value', 'N/A')
                print(f"   Final Value: ${final_value:,.2f}" if isinstance(final_value, (int, float)) else f"   Final Value: {final_value}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print("\n✅ Backend connection test completed!")
    return True

if __name__ == "__main__":
    try:
        test_backend_connection()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)