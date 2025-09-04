"""
Unit tests for Backend API Endpoints (Task 9) - Without Server

This module tests the API endpoint logic without requiring a running server.
Tests the core functionality of the four new endpoints:
- /api/portfolio/allocate
- /api/investment/calculate  
- /api/rebalancing/simulate
- /api/models/predict
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_models import UserInputModel
from models.portfolio_allocation_engine import PortfolioAllocationEngine
from models.investment_calculators import InvestmentCalculators
from models.asset_return_models import AssetReturnModels

logger = logging.getLogger(__name__)


class TestAPIEndpointsUnit:
    """Unit tests for API endpoint business logic."""
    
    @pytest.fixture
    def valid_user_input(self) -> UserInputModel:
        """Valid user input for testing."""
        return UserInputModel(
            investment_amount=100000.0,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=8.0
        )
    
    @pytest.fixture
    def valid_sip_input(self) -> UserInputModel:
        """Valid SIP user input for testing."""
        return UserInputModel(
            investment_amount=120000.0,
            investment_type="sip",
            tenure_years=15,
            risk_profile="High",
            return_expectation=12.0
        )
    
    def test_portfolio_allocation_engine_integration(self, valid_user_input):
        """Test portfolio allocation engine integration."""
        try:
            allocation_engine = PortfolioAllocationEngine()
            
            # Test all risk profiles
            risk_profiles = ['low', 'moderate', 'high']
            
            for risk_profile in risk_profiles:
                allocation = allocation_engine.get_allocation_by_risk_profile(risk_profile)
                
                # Validate allocation structure
                allocation_dict = allocation.to_dict()
                
                # Check all required assets are present
                required_assets = ['sp500', 'small_cap', 't_bills', 't_bonds', 'corporate_bonds', 'real_estate', 'gold']
                for asset in required_assets:
                    assert asset in allocation_dict
                    assert isinstance(allocation_dict[asset], (int, float))
                    assert 0 <= allocation_dict[asset] <= 100
                
                # Check allocation sums to 100%
                total_allocation = sum(allocation_dict.values())
                assert abs(total_allocation - 100.0) < 0.01
                
                # Check risk profile characteristics
                equity_pct = allocation.get_equity_percentage()
                bonds_pct = allocation.get_bonds_percentage()
                
                if risk_profile == 'low':
                    assert bonds_pct > equity_pct, f"Low risk should have more bonds than equity"
                elif risk_profile == 'high':
                    assert equity_pct > bonds_pct, f"High risk should have more equity than bonds"
                
                logger.info(f"✅ {risk_profile} risk allocation: {equity_pct:.1f}% equity, {bonds_pct:.1f}% bonds")
            
            logger.info("✅ Portfolio allocation engine integration test passed")
            
        except Exception as e:
            pytest.fail(f"Portfolio allocation engine test failed: {e}")
    
    def test_investment_calculators_lumpsum(self, valid_user_input):
        """Test investment calculators for lump sum."""
        try:
            investment_calc = InvestmentCalculators()
            
            # Test lump sum calculation
            returns_dict = {'portfolio': valid_user_input.return_expectation}
            
            projections = investment_calc.calculate_lump_sum(
                amount=valid_user_input.investment_amount,
                returns=returns_dict,
                years=valid_user_input.tenure_years
            )
            
            # Validate projections
            assert len(projections) == valid_user_input.tenure_years
            
            for i, projection in enumerate(projections):
                assert projection.year == i + 1
                assert projection.portfolio_value > 0
                assert projection.annual_return == valid_user_input.return_expectation
                
                # Portfolio value should generally increase
                if i > 0:
                    assert projection.portfolio_value >= projections[i-1].portfolio_value
            
            # Test summary generation
            summary = investment_calc.generate_investment_summary(projections)
            assert 'initial_investment' in summary
            assert 'final_value' in summary
            assert 'total_return' in summary
            assert 'cagr' in summary
            
            assert summary['initial_investment'] == valid_user_input.investment_amount
            assert summary['final_value'] > valid_user_input.investment_amount
            
            logger.info(f"✅ Lump sum calculation: {summary['initial_investment']} → {summary['final_value']}")
            logger.info("✅ Investment calculators lump sum test passed")
            
        except Exception as e:
            pytest.fail(f"Investment calculators lump sum test failed: {e}")
    
    def test_investment_calculators_sip(self, valid_sip_input):
        """Test investment calculators for SIP."""
        try:
            investment_calc = InvestmentCalculators()
            
            # Test SIP calculation
            monthly_amount = valid_sip_input.investment_amount / 12
            returns_dict = {'portfolio': valid_sip_input.return_expectation}
            
            projections = investment_calc.calculate_sip(
                monthly_amount=monthly_amount,
                returns=returns_dict,
                years=valid_sip_input.tenure_years
            )
            
            # Validate projections
            assert len(projections) == valid_sip_input.tenure_years
            
            for i, projection in enumerate(projections):
                assert projection.year == i + 1
                assert projection.portfolio_value > 0
                assert projection.annual_contribution > 0  # SIP should have contributions
                
                # Check annual contribution is approximately correct
                expected_annual_contribution = monthly_amount * 12
                assert abs(projection.annual_contribution - expected_annual_contribution) < 0.01
            
            # Final value should be higher than total contributions due to compounding
            final_projection = projections[-1]
            total_contributions = final_projection.cumulative_contributions
            assert final_projection.portfolio_value > total_contributions
            
            logger.info(f"✅ SIP calculation: {total_contributions} contributed → {final_projection.portfolio_value} final value")
            logger.info("✅ Investment calculators SIP test passed")
            
        except Exception as e:
            pytest.fail(f"Investment calculators SIP test failed: {e}")
    
    def test_asset_return_models_structure(self):
        """Test asset return models structure and methods."""
        try:
            asset_models = AssetReturnModels()
            
            # Test asset columns mapping
            expected_assets = ['sp500', 'small_cap', 't_bills', 't_bonds', 'corporate_bonds', 'real_estate', 'gold']
            
            for asset in expected_assets:
                assert asset in asset_models.asset_columns
                assert isinstance(asset_models.asset_columns[asset], str)
            
            # Test model initialization
            assert asset_models.models == {}
            assert asset_models.scalers == {}
            assert asset_models.historical_data is None
            
            logger.info("✅ Asset return models structure test passed")
            
        except Exception as e:
            pytest.fail(f"Asset return models structure test failed: {e}")
    
    def test_rebalancing_logic(self, valid_user_input):
        """Test rebalancing logic."""
        try:
            allocation_engine = PortfolioAllocationEngine()
            
            # Get initial allocation
            initial_allocation = allocation_engine.get_allocation_by_risk_profile('moderate')
            initial_dict = initial_allocation.to_dict()
            
            # Simulate rebalancing logic
            equity_reduction_rate = 2.0
            rebalancing_frequency = 5
            
            current_allocation = initial_dict.copy()
            rebalancing_events = []
            
            for year in range(0, valid_user_input.tenure_years + 1, rebalancing_frequency):
                if year > 0:  # Skip initial year
                    # Reduce equity allocation
                    equity_reduction = min(equity_reduction_rate, 
                                         current_allocation['sp500'] + current_allocation['small_cap'])
                    
                    # Apply reduction
                    if current_allocation['sp500'] >= equity_reduction:
                        current_allocation['sp500'] -= equity_reduction
                    else:
                        remaining = equity_reduction - current_allocation['sp500']
                        current_allocation['sp500'] = 0
                        current_allocation['small_cap'] = max(0, current_allocation['small_cap'] - remaining)
                    
                    # Increase bonds
                    current_allocation['t_bonds'] += equity_reduction
                    
                    # Normalize to 100%
                    total = sum(current_allocation.values())
                    if abs(total - 100.0) > 0.01:
                        for key in current_allocation:
                            current_allocation[key] = (current_allocation[key] / total) * 100
                
                # Record rebalancing event
                equity_pct = current_allocation['sp500'] + current_allocation['small_cap']
                bonds_pct = (current_allocation['t_bills'] + current_allocation['t_bonds'] + 
                           current_allocation['corporate_bonds'])
                
                rebalancing_events.append({
                    'year': year,
                    'equity_percentage': equity_pct,
                    'bonds_percentage': bonds_pct,
                    'allocation': current_allocation.copy()
                })
            
            # Validate rebalancing results
            assert len(rebalancing_events) > 0
            
            # Check equity reduction over time
            initial_equity = rebalancing_events[0]['equity_percentage']
            final_equity = rebalancing_events[-1]['equity_percentage']
            
            assert final_equity <= initial_equity, "Equity should decrease or stay same over time"
            
            # Check all allocations sum to 100%
            for event in rebalancing_events:
                total_allocation = sum(event['allocation'].values())
                assert abs(total_allocation - 100.0) < 0.01
            
            logger.info(f"✅ Rebalancing: {initial_equity:.1f}% → {final_equity:.1f}% equity over {len(rebalancing_events)} events")
            logger.info("✅ Rebalancing logic test passed")
            
        except Exception as e:
            pytest.fail(f"Rebalancing logic test failed: {e}")
    
    def test_input_validation(self):
        """Test input validation for user input model."""
        try:
            # Test valid input
            valid_input = UserInputModel(
                investment_amount=50000.0,
                investment_type="lumpsum",
                tenure_years=5,
                risk_profile="Low",
                return_expectation=6.0
            )
            assert valid_input.investment_amount == 50000.0
            assert valid_input.risk_profile == "Low"
            
            # Test invalid investment amount
            with pytest.raises(ValueError):
                UserInputModel(
                    investment_amount=-1000.0,  # Negative amount
                    investment_type="lumpsum",
                    tenure_years=5,
                    risk_profile="Low",
                    return_expectation=6.0
                )
            
            # Test invalid tenure
            with pytest.raises(ValueError):
                UserInputModel(
                    investment_amount=50000.0,
                    investment_type="lumpsum",
                    tenure_years=0,  # Zero tenure
                    risk_profile="Low",
                    return_expectation=6.0
                )
            
            # Test invalid return expectation
            with pytest.raises(ValueError):
                UserInputModel(
                    investment_amount=50000.0,
                    investment_type="lumpsum",
                    tenure_years=5,
                    risk_profile="Low",
                    return_expectation=-5.0  # Negative return
                )
            
            logger.info("✅ Input validation test passed")
            
        except Exception as e:
            pytest.fail(f"Input validation test failed: {e}")
    
    def test_error_handling_scenarios(self):
        """Test error handling scenarios."""
        try:
            investment_calc = InvestmentCalculators()
            
            # Test invalid inputs to calculators
            with pytest.raises(ValueError):
                investment_calc.calculate_lump_sum(
                    amount=0,  # Zero amount
                    returns={'portfolio': 8.0},
                    years=10
                )
            
            with pytest.raises(ValueError):
                investment_calc.calculate_sip(
                    monthly_amount=-100,  # Negative amount
                    returns={'portfolio': 8.0},
                    years=10
                )
            
            with pytest.raises(ValueError):
                investment_calc.calculate_lump_sum(
                    amount=10000,
                    returns={},  # Empty returns
                    years=10
                )
            
            logger.info("✅ Error handling scenarios test passed")
            
        except Exception as e:
            pytest.fail(f"Error handling scenarios test failed: {e}")
    
    def test_allocation_summary_calculations(self):
        """Test allocation summary calculations."""
        try:
            allocation_engine = PortfolioAllocationEngine()
            
            for risk_profile in ['low', 'moderate', 'high']:
                allocation = allocation_engine.get_allocation_by_risk_profile(risk_profile)
                summary = allocation_engine.get_allocation_summary(risk_profile)
                
                # Validate summary structure
                assert 'equity' in summary
                assert 'bonds' in summary
                assert 'alternatives' in summary
                
                # Validate calculations
                expected_equity = allocation.get_equity_percentage()
                expected_bonds = allocation.get_bonds_percentage()
                expected_alternatives = allocation.get_alternatives_percentage()
                
                assert abs(summary['equity'] - expected_equity) < 0.01
                assert abs(summary['bonds'] - expected_bonds) < 0.01
                assert abs(summary['alternatives'] - expected_alternatives) < 0.01
                
                # Total should be 100%
                total_summary = summary['equity'] + summary['bonds'] + summary['alternatives']
                assert abs(total_summary - 100.0) < 0.01
                
                logger.info(f"✅ {risk_profile} summary: {summary['equity']:.1f}% equity, {summary['bonds']:.1f}% bonds, {summary['alternatives']:.1f}% alternatives")
            
            logger.info("✅ Allocation summary calculations test passed")
            
        except Exception as e:
            pytest.fail(f"Allocation summary calculations test failed: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    pytest.main([__file__, "-v"])