"""
End-to-end integration tests for the complete user flow.

Tests the complete journey from user input to portfolio recommendation display.
"""

import pytest
import requests
import json
from typing import Dict, Any
from models.data_models import UserInputModel


class TestEndToEndUserFlow:
    """Test the complete end-to-end user flow."""
    
    BASE_URL = "http://localhost:8001"
    
    @pytest.fixture
    def sample_user_inputs(self) -> Dict[str, UserInputModel]:
        """Sample user inputs for different scenarios."""
        return {
            "conservative_lumpsum": UserInputModel(
                investment_amount=100000.0,
                investment_type="lumpsum",
                tenure_years=10,
                risk_profile="Low",
                return_expectation=8.0
            ),
            "moderate_sip": UserInputModel(
                investment_amount=50000.0,
                investment_type="sip",
                tenure_years=15,
                risk_profile="Moderate",
                return_expectation=12.0
            ),
            "aggressive_lumpsum": UserInputModel(
                investment_amount=200000.0,
                investment_type="lumpsum",
                tenure_years=20,
                risk_profile="High",
                return_expectation=15.0
            )
        }
    
    def test_complete_user_flow_conservative(self, sample_user_inputs):
        """Test complete flow for conservative investor."""
        user_input = sample_user_inputs["conservative_lumpsum"]
        
        # Step 1: Generate portfolio recommendation
        response = requests.post(
            f"{self.BASE_URL}/api/portfolio/generate",
            json=user_input.model_dump(),
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "success" in data
        assert data["success"] is True
        assert "data" in data
        
        recommendation = data["data"]
        
        # Validate allocation
        assert "allocation" in recommendation
        allocation = recommendation["allocation"]
        assert isinstance(allocation, dict)
        
        # Conservative allocation should have high bonds percentage
        assert allocation["bonds"] >= 40.0  # At least 40% bonds for conservative
        assert allocation["sp500"] + allocation["small_cap"] <= 50.0  # Max 50% equity
        
        # Validate total allocation equals 100%
        total_allocation = sum(allocation.values())
        assert abs(total_allocation - 100.0) < 0.01
        
        # Validate projections
        assert "projections" in recommendation
        projections = recommendation["projections"]
        assert isinstance(projections, list)
        assert len(projections) == user_input.tenure_years + 1
        
        # First projection should be initial investment
        assert projections[0]["year"] == 0
        assert projections[0]["portfolio_value"] == user_input.investment_amount
        assert projections[0]["annual_return"] == 0.0
        
        # Last projection should show growth
        final_projection = projections[-1]
        assert final_projection["year"] == user_input.tenure_years
        assert final_projection["portfolio_value"] > user_input.investment_amount
        
        # Validate risk metrics
        assert "risk_metrics" in recommendation
        risk_metrics = recommendation["risk_metrics"]
        assert "expected_return" in risk_metrics
        assert "volatility" in risk_metrics
        assert "sharpe_ratio" in risk_metrics
        
        # Conservative should have lower volatility
        assert risk_metrics["volatility"] <= 12.0
        
        # Validate summary
        assert "summary" in recommendation
        summary = recommendation["summary"]
        assert summary["initial_investment"] == user_input.investment_amount
        assert summary["risk_profile"] == user_input.risk_profile
        assert summary["tenure_years"] == user_input.tenure_years
        assert summary["final_value"] > summary["initial_investment"]
    
    def test_complete_user_flow_moderate_sip(self, sample_user_inputs):
        """Test complete flow for moderate risk SIP investor."""
        user_input = sample_user_inputs["moderate_sip"]
        
        response = requests.post(
            f"{self.BASE_URL}/api/portfolio/generate",
            json=user_input.model_dump(),
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        recommendation = data["data"]
        
        # Moderate allocation should be balanced
        allocation = recommendation["allocation"]
        equity_percentage = allocation["sp500"] + allocation["small_cap"]
        bonds_percentage = allocation["bonds"]
        
        # Moderate should have balanced equity/bonds
        assert 40.0 <= equity_percentage <= 70.0
        assert 20.0 <= bonds_percentage <= 50.0
        
        # SIP projections should show compounding effect
        projections = recommendation["projections"]
        
        # Growth should be more significant due to SIP compounding
        final_value = projections[-1]["portfolio_value"]
        initial_investment = user_input.investment_amount
        
        # With SIP, final value should be significantly higher due to regular contributions
        growth_multiple = final_value / initial_investment
        assert growth_multiple > 2.0  # At least 2x growth over 15 years with SIP
        
        # Validate moderate risk metrics
        risk_metrics = recommendation["risk_metrics"]
        assert 10.0 <= risk_metrics["volatility"] <= 15.0  # Moderate volatility
        assert risk_metrics["sharpe_ratio"] > 0.5  # Reasonable risk-adjusted return
    
    def test_complete_user_flow_aggressive(self, sample_user_inputs):
        """Test complete flow for aggressive investor."""
        user_input = sample_user_inputs["aggressive_lumpsum"]
        
        response = requests.post(
            f"{self.BASE_URL}/api/portfolio/generate",
            json=user_input.model_dump(),
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        recommendation = data["data"]
        
        # Aggressive allocation should have high equity
        allocation = recommendation["allocation"]
        equity_percentage = allocation["sp500"] + allocation["small_cap"]
        
        assert equity_percentage >= 60.0  # At least 60% equity for aggressive
        assert allocation["bonds"] <= 30.0  # Max 30% bonds
        
        # Aggressive should have higher expected returns and volatility
        risk_metrics = recommendation["risk_metrics"]
        assert risk_metrics["expected_return"] >= 12.0
        assert risk_metrics["volatility"] >= 15.0
        
        # Long-term aggressive investment should show significant growth
        projections = recommendation["projections"]
        final_value = projections[-1]["portfolio_value"]
        growth_multiple = final_value / user_input.investment_amount
        
        # 20 years of aggressive investing should show substantial growth
        assert growth_multiple > 5.0  # At least 5x growth over 20 years
    
    def test_investment_calculation_endpoint(self, sample_user_inputs):
        """Test the investment calculation endpoint separately."""
        user_input = sample_user_inputs["moderate_sip"]
        
        response = requests.post(
            f"{self.BASE_URL}/api/investment/calculate",
            json=user_input.model_dump(),
            timeout=15
        )
        
        assert response.status_code == 200
        data = response.json()
        
        projections = data["data"]
        assert isinstance(projections, list)
        assert len(projections) == user_input.tenure_years + 1
        
        # Validate projection structure
        for projection in projections:
            assert "year" in projection
            assert "portfolio_value" in projection
            assert "annual_return" in projection
            assert "cumulative_return" in projection
            
            # Values should be reasonable
            assert projection["portfolio_value"] >= 0
            assert -100 <= projection["annual_return"] <= 100  # Reasonable return range
    
    def test_rebalancing_simulation_endpoint(self, sample_user_inputs):
        """Test the rebalancing simulation endpoint."""
        user_input = sample_user_inputs["conservative_lumpsum"]
        
        response = requests.post(
            f"{self.BASE_URL}/api/rebalancing/simulate",
            json=user_input.model_dump(),
            timeout=15
        )
        
        assert response.status_code == 200
        data = response.json()
        
        rebalancing_schedule = data["data"]
        assert isinstance(rebalancing_schedule, list)
        assert len(rebalancing_schedule) > 0
        
        # Validate rebalancing structure
        for rebalancing in rebalancing_schedule:
            assert "year" in rebalancing
            assert "equity_allocation" in rebalancing
            assert "bonds_allocation" in rebalancing
            assert "rebalancing_trigger" in rebalancing
            
            # Allocations should sum to 100%
            total = rebalancing["equity_allocation"] + rebalancing["bonds_allocation"]
            assert abs(total - 100.0) < 0.01
    
    def test_model_predictions_endpoint(self):
        """Test the ML model predictions endpoint."""
        response = requests.get(
            f"{self.BASE_URL}/api/models/predict?horizon=10",
            timeout=15
        )
        
        assert response.status_code == 200
        data = response.json()
        
        predictions = data["data"]
        assert isinstance(predictions, dict)
        
        # Validate prediction structure
        required_assets = ["sp500", "small_cap", "bonds", "real_estate", "gold"]
        for asset in required_assets:
            assert asset in predictions
            asset_prediction = predictions[asset]
            
            assert "expected_return" in asset_prediction
            assert "volatility" in asset_prediction
            assert "confidence_interval" in asset_prediction
            
            # Validate reasonable ranges
            assert -10 <= asset_prediction["expected_return"] <= 30
            assert 0 <= asset_prediction["volatility"] <= 50
            assert len(asset_prediction["confidence_interval"]) == 2
        
        assert "horizon_years" in predictions
        assert predictions["horizon_years"] == 10
        assert "market_regime" in predictions
    
    def test_input_validation(self):
        """Test input validation for invalid user inputs."""
        # Test invalid investment amount
        invalid_input = {
            "investment_amount": -1000,  # Negative amount
            "investment_type": "lumpsum",
            "tenure_years": 10,
            "risk_profile": "Moderate",
            "return_expectation": 12.0
        }
        
        response = requests.post(
            f"{self.BASE_URL}/api/portfolio/generate",
            json=invalid_input,
            timeout=15
        )
        
        assert response.status_code == 422  # Validation error
        
        # Test invalid tenure
        invalid_input = {
            "investment_amount": 100000,
            "investment_type": "lumpsum",
            "tenure_years": 100,  # Too long
            "risk_profile": "Moderate",
            "return_expectation": 12.0
        }
        
        response = requests.post(
            f"{self.BASE_URL}/api/portfolio/generate",
            json=invalid_input,
            timeout=15
        )
        
        assert response.status_code == 422  # Validation error
        
        # Test invalid risk profile
        invalid_input = {
            "investment_amount": 100000,
            "investment_type": "lumpsum",
            "tenure_years": 10,
            "risk_profile": "Invalid",  # Invalid risk profile
            "return_expectation": 12.0
        }
        
        response = requests.post(
            f"{self.BASE_URL}/api/portfolio/generate",
            json=invalid_input,
            timeout=15
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_error_handling(self):
        """Test error handling for malformed requests."""
        # Test missing required fields
        incomplete_input = {
            "investment_amount": 100000
            # Missing other required fields
        }
        
        response = requests.post(
            f"{self.BASE_URL}/api/portfolio/generate",
            json=incomplete_input,
            timeout=15
        )
        
        assert response.status_code == 422  # Validation error
        
        # Test invalid JSON
        response = requests.post(
            f"{self.BASE_URL}/api/portfolio/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        assert response.status_code == 422  # JSON decode error
    
    def test_api_response_format(self, sample_user_inputs):
        """Test that all API responses follow the standard format."""
        user_input = sample_user_inputs["moderate_sip"]
        
        endpoints = [
            "/api/portfolio/generate",
            "/api/investment/calculate",
            "/api/rebalancing/simulate"
        ]
        
        for endpoint in endpoints:
            response = requests.post(
                f"{self.BASE_URL}{endpoint}",
                json=user_input.model_dump(),
                timeout=15
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate standard API response format
            assert "success" in data
            assert "data" in data
            assert "message" in data
            assert "timestamp" in data
            
            assert data["success"] is True
            assert isinstance(data["data"], (dict, list))
            assert isinstance(data["message"], str)
            assert isinstance(data["timestamp"], str)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])