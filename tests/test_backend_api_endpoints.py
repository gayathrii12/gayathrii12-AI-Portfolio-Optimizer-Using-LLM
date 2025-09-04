"""
Integration tests for Backend API Endpoints (Task 9)

This module tests the four new API endpoints:
- /api/portfolio/allocate
- /api/investment/calculate  
- /api/rebalancing/simulate
- /api/models/predict

Tests include input validation, error handling, and response format validation.
"""

import pytest
import requests
import json
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = "http://localhost:8000"
API_ENDPOINTS = {
    'portfolio_allocate': f"{BASE_URL}/api/portfolio/allocate",
    'investment_calculate': f"{BASE_URL}/api/investment/calculate",
    'rebalancing_simulate': f"{BASE_URL}/api/rebalancing/simulate",
    'models_predict': f"{BASE_URL}/api/models/predict"
}


class TestBackendAPIEndpoints:
    """Test suite for backend API endpoints."""
    
    @pytest.fixture
    def valid_user_input(self) -> Dict[str, Any]:
        """Valid user input for testing."""
        return {
            "investment_amount": 100000.0,
            "investment_type": "lumpsum",
            "tenure_years": 10,
            "risk_profile": "Moderate",
            "return_expectation": 8.0
        }
    
    @pytest.fixture
    def valid_sip_input(self) -> Dict[str, Any]:
        """Valid SIP user input for testing."""
        return {
            "investment_amount": 120000.0,  # 10k per month for 12 months
            "investment_type": "sip",
            "tenure_years": 15,
            "risk_profile": "High",
            "return_expectation": 12.0
        }
    
    @pytest.fixture
    def valid_prediction_request(self) -> Dict[str, Any]:
        """Valid ML prediction request."""
        return {
            "asset_classes": ["sp500", "small_cap", "t_bonds"],
            "horizon": 1,
            "include_confidence": True
        }
    
    def test_portfolio_allocate_endpoint(self, valid_user_input):
        """Test /api/portfolio/allocate endpoint."""
        try:
            response = requests.post(
                API_ENDPOINTS['portfolio_allocate'],
                json=valid_user_input,
                timeout=30
            )
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            
            data = response.json()
            assert data['success'] is True
            assert 'data' in data
            
            # Validate response structure
            response_data = data['data']
            assert 'allocation' in response_data
            assert 'allocation_summary' in response_data
            assert 'risk_metrics' in response_data
            
            # Validate allocation structure
            allocation = response_data['allocation']
            required_assets = ['sp500', 'small_cap', 't_bills', 't_bonds', 'corporate_bonds', 'real_estate', 'gold']
            for asset in required_assets:
                assert asset in allocation
                assert isinstance(allocation[asset], (int, float))
                assert 0 <= allocation[asset] <= 100
            
            # Validate allocation sums to 100%
            total_allocation = sum(allocation.values())
            assert abs(total_allocation - 100.0) < 0.01, f"Allocation should sum to 100%, got {total_allocation}"
            
            # Validate risk metrics
            risk_metrics = response_data['risk_metrics']
            assert 'expected_return' in risk_metrics
            assert 'estimated_volatility' in risk_metrics
            assert 'sharpe_ratio' in risk_metrics
            assert 'risk_level' in risk_metrics
            
            logger.info("✅ Portfolio allocate endpoint test passed")
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API server not available: {e}")
    
    def test_portfolio_allocate_different_risk_profiles(self):
        """Test portfolio allocation with different risk profiles."""
        risk_profiles = ["Low", "Moderate", "High"]
        
        for risk_profile in risk_profiles:
            user_input = {
                "investment_amount": 50000.0,
                "investment_type": "lumpsum",
                "tenure_years": 5,
                "risk_profile": risk_profile,
                "return_expectation": 7.0
            }
            
            try:
                response = requests.post(
                    API_ENDPOINTS['portfolio_allocate'],
                    json=user_input,
                    timeout=30
                )
                
                assert response.status_code == 200
                data = response.json()
                
                allocation_summary = data['data']['allocation_summary']
                
                # Validate risk profile characteristics
                if risk_profile == "Low":
                    assert allocation_summary['bonds'] > allocation_summary['equity']
                elif risk_profile == "High":
                    assert allocation_summary['equity'] > allocation_summary['bonds']
                
                logger.info(f"✅ {risk_profile} risk profile allocation test passed")
                
            except requests.exceptions.RequestException as e:
                pytest.skip(f"API server not available: {e}")
    
    def test_investment_calculate_lumpsum(self, valid_user_input):
        """Test /api/investment/calculate endpoint with lump sum."""
        try:
            response = requests.post(
                API_ENDPOINTS['investment_calculate'],
                json=valid_user_input,
                timeout=30
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['success'] is True
            
            # Validate response structure
            response_data = data['data']
            assert 'projections' in response_data
            assert 'summary' in response_data
            assert 'parameters' in response_data
            
            # Validate projections
            projections = response_data['projections']
            assert len(projections) == valid_user_input['tenure_years']
            
            for i, projection in enumerate(projections):
                assert 'year' in projection
                assert 'portfolio_value' in projection
                assert 'annual_return' in projection
                assert projection['year'] == i + 1
                assert projection['portfolio_value'] > 0
                
                # Portfolio value should generally increase over time
                if i > 0:
                    assert projection['portfolio_value'] >= projections[i-1]['portfolio_value'] * 0.8  # Allow for some volatility
            
            # Validate summary
            summary = response_data['summary']
            assert 'initial_investment' in summary
            assert 'final_value' in summary
            assert 'total_return' in summary
            assert summary['initial_investment'] == valid_user_input['investment_amount']
            
            logger.info("✅ Investment calculate lumpsum test passed")
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API server not available: {e}")
    
    def test_investment_calculate_sip(self, valid_sip_input):
        """Test /api/investment/calculate endpoint with SIP."""
        try:
            response = requests.post(
                API_ENDPOINTS['investment_calculate'],
                json=valid_sip_input,
                timeout=30
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['success'] is True
            
            response_data = data['data']
            projections = response_data['projections']
            
            # Validate SIP-specific behavior
            for projection in projections:
                assert 'annual_contribution' in projection
                # For SIP, should have annual contributions
                if projection['year'] > 0:
                    assert projection['annual_contribution'] > 0
            
            logger.info("✅ Investment calculate SIP test passed")
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API server not available: {e}")
    
    def test_rebalancing_simulate(self, valid_user_input):
        """Test /api/rebalancing/simulate endpoint."""
        # Add rebalancing preferences
        user_input_with_rebalancing = valid_user_input.copy()
        user_input_with_rebalancing['rebalancing_preferences'] = {
            'equity_reduction_rate': 3.0,
            'frequency_years': 3
        }
        
        try:
            response = requests.post(
                API_ENDPOINTS['rebalancing_simulate'],
                json=user_input_with_rebalancing,
                timeout=30
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['success'] is True
            
            # Validate response structure
            response_data = data['data']
            assert 'rebalancing_schedule' in response_data
            assert 'projections_with_rebalancing' in response_data
            assert 'rebalancing_summary' in response_data
            
            # Validate rebalancing schedule
            schedule = response_data['rebalancing_schedule']
            assert len(schedule) > 0
            
            for event in schedule:
                assert 'year' in event
                assert 'allocation' in event
                assert 'equity_percentage' in event
                assert 'bonds_percentage' in event
                
                # Validate allocation sums to 100%
                allocation_total = sum(event['allocation'].values())
                assert abs(allocation_total - 100.0) < 0.01
            
            # Validate equity reduction over time
            initial_equity = schedule[0]['equity_percentage']
            final_equity = schedule[-1]['equity_percentage']
            assert final_equity <= initial_equity, "Equity percentage should decrease or stay same over time"
            
            # Validate rebalancing summary
            summary = response_data['rebalancing_summary']
            assert 'initial_equity_percentage' in summary
            assert 'final_equity_percentage' in summary
            assert 'total_equity_reduction' in summary
            
            logger.info("✅ Rebalancing simulate test passed")
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API server not available: {e}")
    
    def test_models_predict_all_assets(self):
        """Test /api/models/predict endpoint for all assets."""
        request_data = {
            "horizon": 1,
            "include_confidence": True
        }
        
        try:
            response = requests.post(
                API_ENDPOINTS['models_predict'],
                json=request_data,
                timeout=60  # ML predictions might take longer
            )
            
            # Accept both 200 (success) and 503 (models not available)
            assert response.status_code in [200, 503]
            
            if response.status_code == 200:
                data = response.json()
                assert data['success'] is True
                
                response_data = data['data']
                assert 'predictions' in response_data
                assert 'model_information' in response_data
                
                # Validate predictions structure
                predictions = response_data['predictions']
                assert len(predictions) > 0
                
                for asset_class, prediction in predictions.items():
                    if 'predicted_return' in prediction and prediction['predicted_return'] is not None:
                        assert isinstance(prediction['predicted_return'], (int, float))
                        assert 'prediction_horizon_years' in prediction
                        assert 'asset_name' in prediction
                
                # Validate confidence intervals if included
                if 'confidence_intervals' in response_data and response_data['confidence_intervals']:
                    confidence = response_data['confidence_intervals']
                    for asset_class, interval in confidence.items():
                        assert 'lower_95' in interval
                        assert 'upper_95' in interval
                        assert interval['lower_95'] <= interval['upper_95']
                
                logger.info("✅ Models predict all assets test passed")
            else:
                logger.info("⚠️ ML models not available, test skipped")
                
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API server not available: {e}")
    
    def test_models_predict_specific_assets(self, valid_prediction_request):
        """Test /api/models/predict endpoint for specific assets."""
        try:
            response = requests.post(
                API_ENDPOINTS['models_predict'],
                json=valid_prediction_request,
                timeout=60
            )
            
            assert response.status_code in [200, 503]
            
            if response.status_code == 200:
                data = response.json()
                response_data = data['data']
                
                # Validate only requested assets are returned
                predictions = response_data['predictions']
                requested_assets = valid_prediction_request['asset_classes']
                
                for asset in requested_assets:
                    assert asset in predictions
                
                logger.info("✅ Models predict specific assets test passed")
            else:
                logger.info("⚠️ ML models not available, test skipped")
                
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API server not available: {e}")
    
    def test_input_validation_errors(self):
        """Test input validation and error handling."""
        
        # Test invalid investment amount
        invalid_input = {
            "investment_amount": -1000.0,  # Negative amount
            "investment_type": "lumpsum",
            "tenure_years": 10,
            "risk_profile": "Moderate",
            "return_expectation": 8.0
        }
        
        try:
            response = requests.post(
                API_ENDPOINTS['portfolio_allocate'],
                json=invalid_input,
                timeout=30
            )
            
            assert response.status_code == 422  # Validation error
            logger.info("✅ Input validation error test passed")
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API server not available: {e}")
        
        # Test invalid risk profile
        invalid_risk_input = {
            "investment_amount": 50000.0,
            "investment_type": "lumpsum",
            "tenure_years": 10,
            "risk_profile": "Invalid",  # Invalid risk profile
            "return_expectation": 8.0
        }
        
        try:
            response = requests.post(
                API_ENDPOINTS['portfolio_allocate'],
                json=invalid_risk_input,
                timeout=30
            )
            
            assert response.status_code in [400, 422]  # Validation error
            logger.info("✅ Invalid risk profile error test passed")
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API server not available: {e}")
    
    def test_ml_prediction_validation(self):
        """Test ML prediction input validation."""
        
        # Test invalid horizon
        invalid_horizon = {
            "horizon": 15,  # Too long
            "include_confidence": False
        }
        
        try:
            response = requests.post(
                API_ENDPOINTS['models_predict'],
                json=invalid_horizon,
                timeout=30
            )
            
            assert response.status_code == 400  # Validation error
            logger.info("✅ ML prediction validation test passed")
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API server not available: {e}")
    
    def test_response_format_consistency(self, valid_user_input):
        """Test that all endpoints return consistent response format."""
        endpoints_to_test = [
            ('portfolio_allocate', valid_user_input),
            ('investment_calculate', valid_user_input),
            ('rebalancing_simulate', valid_user_input)
        ]
        
        for endpoint_name, test_input in endpoints_to_test:
            try:
                response = requests.post(
                    API_ENDPOINTS[endpoint_name],
                    json=test_input,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Validate standard response format
                    assert 'success' in data
                    assert 'data' in data
                    assert 'message' in data
                    assert 'timestamp' in data
                    assert 'data_source' in data
                    
                    assert data['success'] is True
                    assert isinstance(data['data'], dict)
                    assert isinstance(data['message'], str)
                    
                    logger.info(f"✅ Response format consistency test passed for {endpoint_name}")
                
            except requests.exceptions.RequestException as e:
                pytest.skip(f"API server not available: {e}")


if __name__ == "__main__":
    # Run tests directly
    import sys
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running, starting tests...")
            pytest.main([__file__, "-v"])
        else:
            print("❌ API server returned unexpected status code")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print("❌ API server is not running. Please start the server first:")
        print("   python backend_api_with_agents.py")
        sys.exit(1)