#!/usr/bin/env python3
"""
Simple test script to verify the backend API response structure
"""
import requests
import json
import sys

def test_portfolio_generate():
    """Test the /api/portfolio/generate endpoint"""
    url = "http://localhost:8000/api/portfolio/generate"
    
    test_data = {
        "investment_amount": 100000,
        "investment_type": "lumpsum",
        "tenure_years": 10,
        "risk_profile": "Moderate",
        "return_expectation": 12.0
    }
    
    try:
        print("Testing /api/portfolio/generate endpoint...")
        print(f"Request data: {json.dumps(test_data, indent=2)}")
        
        response = requests.post(url, json=test_data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Response structure:")
            print(json.dumps(data, indent=2))
            
            # Check if required fields are present
            if 'data' in data:
                response_data = data['data']
                required_fields = ['allocation', 'projections', 'risk_metrics', 'summary']
                missing_fields = []
                
                for field in required_fields:
                    if field not in response_data:
                        missing_fields.append(field)
                
                if missing_fields:
                    print(f"❌ Missing required fields: {missing_fields}")
                else:
                    print("✅ All required fields present")
                    
                    # Check summary fields
                    if 'summary' in response_data and 'risk_profile' in response_data['summary']:
                        print("✅ risk_profile field present in summary")
                    else:
                        print("❌ risk_profile field missing in summary")
            else:
                print("❌ No 'data' field in response")
        else:
            print(f"❌ Request failed: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_portfolio_generate()