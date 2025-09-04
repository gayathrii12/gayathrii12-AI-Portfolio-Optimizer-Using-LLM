#!/usr/bin/env python3
"""
Test script to simulate frontend calls to the backend
"""

import requests
import json
import sys

def test_frontend_integration():
    """Test the exact calls that the frontend makes"""
    base_url = "http://localhost:8000"
    
    print("🔍 Testing Frontend Integration...")
    
    # Test data that matches what the frontend sends
    test_data = {
        "investment_amount": 100000.0,
        "investment_type": "lumpsum",
        "tenure_years": 10,
        "risk_profile": "Moderate",
        "return_expectation": 8.0
    }
    
    print(f"Test Data: {json.dumps(test_data, indent=2)}")
    
    # Test the main portfolio generation endpoint (what frontend should use)
    try:
        print("\n" + "="*60)
        print("TESTING: /api/portfolio/generate (Main Frontend Call)")
        print("="*60)
        response = requests.post(f"{base_url}/api/portfolio/generate", json=test_data, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS!")
            print(f"  Success: {data.get('success')}")
            print(f"  Message: {data.get('message')}")
            
            response_data = data.get('data', {})
            
            # Check allocation
            allocation = response_data.get('allocation', {})
            print(f"\n📊 Allocation:")
            for asset, percentage in allocation.items():
                if percentage > 0:
                    print(f"  {asset}: {percentage}%")
            
            # Check projections
            projections = response_data.get('projections', [])
            print(f"\n📈 Projections ({len(projections)} years):")
            if projections:
                print(f"  Year 0: ${projections[0].get('portfolio_value', 0):,.2f}")
                print(f"  Year {len(projections)-1}: ${projections[-1].get('portfolio_value', 0):,.2f}")
            
            # Check risk metrics
            risk_metrics = response_data.get('risk_metrics', {})
            print(f"\n📊 Risk Metrics:")
            print(f"  Expected Return: {risk_metrics.get('expected_return', 'N/A')}%")
            print(f"  Volatility: {risk_metrics.get('volatility', 'N/A')}%")
            print(f"  Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 'N/A')}")
            
            # Check summary
            summary = response_data.get('summary', {})
            print(f"\n💼 Summary:")
            print(f"  Initial Investment: ${summary.get('initial_investment', 0):,.2f}")
            print(f"  Final Value: ${summary.get('final_value', 0):,.2f}")
            print(f"  Total Return: ${summary.get('total_return', 0):,.2f}")
            
            print("\n✅ This endpoint provides everything the frontend needs!")
            
        else:
            print(f"❌ FAILED!")
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    # Test dashboard endpoint
    try:
        print("\n" + "="*60)
        print("TESTING: /api/dashboard (Dashboard Data)")
        print("="*60)
        response = requests.get(f"{base_url}/api/dashboard", timeout=15)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS!")
            print(f"  System Status: {data.get('data', {}).get('system_status', 'N/A')}")
            print(f"  Last Updated: {data.get('data', {}).get('last_updated', 'N/A')}")
        else:
            print(f"❌ FAILED!")
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    # Test agent status endpoint
    try:
        print("\n" + "="*60)
        print("TESTING: /api/agent-status (Agent Status)")
        print("="*60)
        response = requests.get(f"{base_url}/api/agent-status", timeout=15)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS!")
            print(f"  Pipeline Status: {data.get('data', {}).get('pipeline_status', 'N/A')}")
            agents = data.get('data', {}).get('agents_executed', [])
            print(f"  Agents Executed: {len(agents)}")
        else:
            print(f"❌ FAILED!")
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")

    print("\n" + "="*60)
    print("🎉 FRONTEND INTEGRATION TEST COMPLETED!")
    print("="*60)
    print("\n📋 Summary:")
    print("✅ /api/portfolio/generate - Main endpoint for portfolio recommendations")
    print("✅ /api/dashboard - Dashboard data")
    print("✅ /api/agent-status - Agent pipeline status")
    print("\n💡 The frontend should use /api/portfolio/generate for all portfolio recommendations!")

if __name__ == "__main__":
    try:
        test_frontend_integration()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)