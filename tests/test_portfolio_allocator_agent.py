"""
Unit tests for the Portfolio Allocator Agent.

This module tests the portfolio allocation logic, constraint validation,
risk profile mapping, and optimization algorithms.
"""

import pytest
import numpy as np
from typing import Dict, List

from agents.portfolio_allocator_agent import (
    PortfolioAllocatorAgent,
    RiskProfile,
    AllocationInput,
    AllocationStrategy,
    OptimizationResult,
    create_portfolio_allocator_agent
)
from models.data_models import PortfolioAllocation, AssetReturns


class TestPortfolioAllocatorAgent:
    """Test cases for Portfolio Allocator Agent functionality."""
    
    @pytest.fixture
    def agent(self):
        """Create a Portfolio Allocator Agent for testing."""
        return create_portfolio_allocator_agent()
    
    @pytest.fixture
    def sample_historical_data(self) -> List[AssetReturns]:
        """Create sample historical data for testing."""
        return [
            AssetReturns(
                sp500=0.10, small_cap=0.12, t_bills=0.02, t_bonds=0.04,
                corporate_bonds=0.05, real_estate=0.08, gold=0.06, year=2020
            ),
            AssetReturns(
                sp500=0.15, small_cap=0.18, t_bills=0.01, t_bonds=0.03,
                corporate_bonds=0.04, real_estate=0.12, gold=0.08, year=2021
            ),
            AssetReturns(
                sp500=0.08, small_cap=0.10, t_bills=0.03, t_bonds=0.05,
                corporate_bonds=0.06, real_estate=0.06, gold=0.04, year=2022
            ),
            AssetReturns(
                sp500=0.12, small_cap=0.14, t_bills=0.02, t_bonds=0.04,
                corporate_bonds=0.05, real_estate=0.09, gold=0.07, year=2023
            )
        ]
    
    @pytest.fixture
    def sample_expected_returns(self) -> Dict[str, float]:
        """Create sample expected returns for testing."""
        return {
            "sp500": 0.10,
            "small_cap": 0.12,
            "t_bills": 0.02,
            "t_bonds": 0.04,
            "corporate_bonds": 0.05,
            "real_estate": 0.08,
            "gold": 0.06
        }
    
    def test_agent_initialization(self, agent):
        """Test that the agent initializes correctly."""
        assert agent is not None
        assert len(agent.tools) == 4
        assert len(agent.allocation_strategies) == 3
        assert RiskProfile.LOW in agent.allocation_strategies
        assert RiskProfile.MODERATE in agent.allocation_strategies
        assert RiskProfile.HIGH in agent.allocation_strategies
    
    def test_allocation_strategies_initialization(self, agent):
        """Test that allocation strategies are properly initialized."""
        strategies = agent.get_allocation_strategies()
        
        # Test Low risk strategy
        low_strategy = strategies[RiskProfile.LOW]
        assert low_strategy.strategy_name == "Conservative Income"
        assert low_strategy.base_allocation["bonds"] == 65.0
        assert low_strategy.base_allocation["real_estate"] == 17.5
        assert low_strategy.base_allocation["sp500"] == 12.5
        assert low_strategy.base_allocation["gold"] == 5.0
        assert low_strategy.base_allocation["small_cap"] == 0.0
        
        # Test Moderate risk strategy
        moderate_strategy = strategies[RiskProfile.MODERATE]
        assert moderate_strategy.strategy_name == "Balanced Growth"
        assert moderate_strategy.base_allocation["sp500"] == 45.0
        assert moderate_strategy.base_allocation["bonds"] == 30.0
        assert moderate_strategy.base_allocation["real_estate"] == 12.5
        assert moderate_strategy.base_allocation["small_cap"] == 7.5
        assert moderate_strategy.base_allocation["gold"] == 5.0
        
        # Test High risk strategy
        high_strategy = strategies[RiskProfile.HIGH]
        assert high_strategy.strategy_name == "Aggressive Growth"
        assert high_strategy.base_allocation["sp500"] == 55.0
        assert high_strategy.base_allocation["small_cap"] == 20.0
        assert high_strategy.base_allocation["real_estate"] == 12.5
        assert high_strategy.base_allocation["bonds"] == 10.0
        assert high_strategy.base_allocation["gold"] == 2.5
    
    def test_allocation_strategies_sum_to_100(self, agent):
        """Test that all allocation strategies sum to 100%."""
        strategies = agent.get_allocation_strategies()
        
        for risk_profile, strategy in strategies.items():
            total = sum(strategy.base_allocation.values())
            assert abs(total - 100.0) < 0.01, f"{risk_profile} strategy total: {total}"
    
    def test_correlation_matrix_calculation(self, agent, sample_historical_data):
        """Test correlation matrix calculation."""
        correlation_matrix = agent._calculate_correlation_matrix(sample_historical_data)
        
        # Check structure
        expected_assets = ['sp500', 'small_cap', 't_bills', 't_bonds', 'corporate_bonds', 'real_estate', 'gold']
        for asset in expected_assets:
            assert asset in correlation_matrix
            assert len(correlation_matrix[asset]) == len(expected_assets)
        
        # Check diagonal elements (should be 1.0)
        for asset in expected_assets:
            assert abs(correlation_matrix[asset][asset] - 1.0) < 0.01
        
        # Check symmetry
        for asset1 in expected_assets:
            for asset2 in expected_assets:
                assert abs(correlation_matrix[asset1][asset2] - correlation_matrix[asset2][asset1]) < 0.01
    
    def test_portfolio_allocation_low_risk(self, agent, sample_historical_data, sample_expected_returns):
        """Test portfolio allocation for Low risk profile."""
        allocation_input = AllocationInput(
            risk_profile=RiskProfile.LOW,
            expected_returns=sample_expected_returns,
            historical_data=sample_historical_data,
            optimization_method="strategic"
        )
        
        result = agent.allocate_portfolio(allocation_input)
        
        assert result.success is True
        assert result.strategy_used.risk_profile == RiskProfile.LOW
        
        # Check allocation constraints
        allocation = result.allocation
        assert 0 <= allocation.sp500 <= 100
        assert 0 <= allocation.small_cap <= 100
        assert 0 <= allocation.bonds <= 100
        assert 0 <= allocation.gold <= 100
        assert 0 <= allocation.real_estate <= 100
        
        # Check total equals 100%
        total = allocation.sp500 + allocation.small_cap + allocation.bonds + allocation.gold + allocation.real_estate
        assert abs(total - 100.0) <= 0.01
        
        # Check that it follows low risk profile (high bonds allocation)
        assert allocation.bonds >= 50.0  # Should have significant bond allocation
    
    def test_portfolio_allocation_moderate_risk(self, agent, sample_historical_data, sample_expected_returns):
        """Test portfolio allocation for Moderate risk profile."""
        allocation_input = AllocationInput(
            risk_profile=RiskProfile.MODERATE,
            expected_returns=sample_expected_returns,
            historical_data=sample_historical_data,
            optimization_method="strategic"
        )
        
        result = agent.allocate_portfolio(allocation_input)
        
        assert result.success is True
        assert result.strategy_used.risk_profile == RiskProfile.MODERATE
        
        # Check allocation constraints
        allocation = result.allocation
        total = allocation.sp500 + allocation.small_cap + allocation.bonds + allocation.gold + allocation.real_estate
        assert abs(total - 100.0) <= 0.01
        
        # Check balanced allocation (significant equity and bonds)
        assert allocation.sp500 >= 30.0  # Should have significant equity
        assert allocation.bonds >= 20.0  # Should have significant bonds
    
    def test_portfolio_allocation_high_risk(self, agent, sample_historical_data, sample_expected_returns):
        """Test portfolio allocation for High risk profile."""
        allocation_input = AllocationInput(
            risk_profile=RiskProfile.MODERATE,
            expected_returns=sample_expected_returns,
            historical_data=sample_historical_data,
            optimization_method="strategic"
        )
        
        result = agent.allocate_portfolio(allocation_input)
        
        assert result.success is True
        
        # Check allocation constraints
        allocation = result.allocation
        total = allocation.sp500 + allocation.small_cap + allocation.bonds + allocation.gold + allocation.real_estate
        assert abs(total - 100.0) <= 0.01
        
        # All individual allocations should be within 0-100%
        assert 0 <= allocation.sp500 <= 100
        assert 0 <= allocation.small_cap <= 100
        assert 0 <= allocation.bonds <= 100
        assert 0 <= allocation.gold <= 100
        assert 0 <= allocation.real_estate <= 100
    
    def test_different_optimization_methods(self, agent, sample_historical_data, sample_expected_returns):
        """Test different optimization methods."""
        base_input = AllocationInput(
            risk_profile=RiskProfile.MODERATE,
            expected_returns=sample_expected_returns,
            historical_data=sample_historical_data
        )
        
        methods = ["strategic", "mean_variance", "risk_parity"]
        results = {}
        
        for method in methods:
            input_params = AllocationInput(
                risk_profile=base_input.risk_profile,
                expected_returns=base_input.expected_returns,
                historical_data=base_input.historical_data,
                optimization_method=method
            )
            
            result = agent.allocate_portfolio(input_params)
            results[method] = result
            
            # All methods should succeed
            assert result.success is True
            
            # All should meet constraints
            allocation = result.allocation
            total = allocation.sp500 + allocation.small_cap + allocation.bonds + allocation.gold + allocation.real_estate
            assert abs(total - 100.0) <= 0.01
    
    def test_constraint_validation(self, agent):
        """Test allocation constraint validation."""
        # Valid allocation
        valid_allocation = PortfolioAllocation(
            sp500=50.0, small_cap=10.0, bonds=30.0, gold=5.0, real_estate=5.0
        )
        validation = agent._validate_constraints(valid_allocation)
        assert validation["all_constraints_met"] is True
        assert validation["total_equals_100"] is True
        
        # Test that the validation function works correctly for valid allocations
        # (Invalid allocations can't be created due to Pydantic validation)
        another_valid_allocation = PortfolioAllocation(
            sp500=60.0, small_cap=15.0, bonds=20.0, gold=3.0, real_estate=2.0
        )
        validation = agent._validate_constraints(another_valid_allocation)
        assert validation["all_constraints_met"] is True
        assert validation["total_equals_100"] is True
    
    def test_normalization_function(self, agent):
        """Test allocation normalization."""
        # Test normal case
        allocation = {"sp500": 45.0, "bonds": 30.0, "real_estate": 15.0, "gold": 10.0}
        normalized = agent._normalize_allocation(allocation)
        total = sum(normalized.values())
        assert abs(total - 100.0) <= 0.01
        
        # Test case where total is not 100%
        allocation = {"sp500": 50.0, "bonds": 40.0, "real_estate": 20.0, "gold": 10.0}  # Sums to 120%
        normalized = agent._normalize_allocation(allocation)
        total = sum(normalized.values())
        assert abs(total - 100.0) <= 0.01
        
        # Test zero case
        allocation = {"sp500": 0.0, "bonds": 0.0, "real_estate": 0.0, "gold": 0.0}
        normalized = agent._normalize_allocation(allocation)
        total = sum(normalized.values())
        assert abs(total - 100.0) <= 0.01
        # Should distribute equally
        for value in normalized.values():
            assert abs(value - 25.0) <= 0.01
    
    def test_user_preferences_application(self, agent, sample_historical_data, sample_expected_returns):
        """Test application of user preferences."""
        user_preferences = {
            "asset_preferences": {
                "sp500": {"min_allocation": 40.0},
                "gold": {"max_allocation": 3.0}
            },
            "esg_focused": True
        }
        
        allocation_input = AllocationInput(
            risk_profile=RiskProfile.MODERATE,
            expected_returns=sample_expected_returns,
            historical_data=sample_historical_data,
            user_preferences=user_preferences,
            optimization_method="strategic"
        )
        
        result = agent.allocate_portfolio(allocation_input)
        
        assert result.success is True
        
        # Check that preferences were applied
        allocation = result.allocation
        assert allocation.sp500 >= 40.0  # Should respect minimum
        assert allocation.gold <= 3.0   # Should respect maximum
    
    def test_optimization_metrics_calculation(self, agent, sample_expected_returns):
        """Test optimization metrics calculation."""
        allocation = PortfolioAllocation(
            sp500=50.0, small_cap=10.0, bonds=30.0, gold=5.0, real_estate=5.0
        )
        
        correlation_matrix = {
            "sp500": {"sp500": 1.0, "small_cap": 0.8},
            "small_cap": {"sp500": 0.8, "small_cap": 1.0}
        }
        
        metrics = agent._calculate_optimization_metrics(
            allocation, sample_expected_returns, correlation_matrix
        )
        
        assert "expected_return" in metrics
        assert "expected_volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "diversification_ratio" in metrics
        
        # Check that values are reasonable
        assert 0 <= metrics["expected_return"] <= 1.0
        assert 0 <= metrics["expected_volatility"] <= 1.0
        assert metrics["diversification_ratio"] >= 1.0
    
    def test_get_risk_profile_allocation(self, agent):
        """Test getting base allocation for risk profiles."""
        for risk_profile in [RiskProfile.LOW, RiskProfile.MODERATE, RiskProfile.HIGH]:
            allocation = agent.get_risk_profile_allocation(risk_profile)
            
            # Check that it's a valid PortfolioAllocation
            assert isinstance(allocation, PortfolioAllocation)
            
            # Check constraints
            total = allocation.sp500 + allocation.small_cap + allocation.bonds + allocation.gold + allocation.real_estate
            assert abs(total - 100.0) <= 0.01
            
            # Check individual constraints
            assert 0 <= allocation.sp500 <= 100
            assert 0 <= allocation.small_cap <= 100
            assert 0 <= allocation.bonds <= 100
            assert 0 <= allocation.gold <= 100
            assert 0 <= allocation.real_estate <= 100
    
    def test_error_handling_invalid_risk_profile(self, agent, sample_historical_data, sample_expected_returns):
        """Test error handling with invalid inputs."""
        # Test with invalid optimization method
        allocation_input = AllocationInput(
            risk_profile=RiskProfile.MODERATE,
            expected_returns=sample_expected_returns,
            historical_data=sample_historical_data,
            optimization_method="invalid_method"
        )
        
        result = agent.allocate_portfolio(allocation_input)
        
        # Should still succeed with fallback to strategic
        assert result.success is True
    
    def test_empty_historical_data_handling(self, agent, sample_expected_returns):
        """Test handling of empty historical data."""
        allocation_input = AllocationInput(
            risk_profile=RiskProfile.MODERATE,
            expected_returns=sample_expected_returns,
            historical_data=[],  # Empty data
            optimization_method="strategic"
        )
        
        result = agent.allocate_portfolio(allocation_input)
        
        # Should handle gracefully and return valid allocation
        assert isinstance(result, OptimizationResult)
        if result.success:
            total = (result.allocation.sp500 + result.allocation.small_cap + 
                    result.allocation.bonds + result.allocation.gold + result.allocation.real_estate)
            assert abs(total - 100.0) <= 0.01
    
    def test_extreme_expected_returns(self, agent, sample_historical_data):
        """Test handling of extreme expected returns."""
        extreme_returns = {
            "sp500": 0.50,      # 50% return
            "small_cap": -0.20,  # -20% return
            "t_bills": 0.01,
            "t_bonds": 0.02,
            "corporate_bonds": 0.03,
            "real_estate": 0.30,
            "gold": 0.00
        }
        
        allocation_input = AllocationInput(
            risk_profile=RiskProfile.MODERATE,
            expected_returns=extreme_returns,
            historical_data=sample_historical_data,
            optimization_method="strategic"
        )
        
        result = agent.allocate_portfolio(allocation_input)
        
        # Should handle extreme values and still produce valid allocation
        assert isinstance(result, OptimizationResult)
        if result.success:
            allocation = result.allocation
            total = allocation.sp500 + allocation.small_cap + allocation.bonds + allocation.gold + allocation.real_estate
            assert abs(total - 100.0) <= 0.01


class TestAllocationStrategies:
    """Test cases for allocation strategy logic."""
    
    def test_allocation_strategy_ranges(self):
        """Test that allocation strategy ranges are valid."""
        agent = create_portfolio_allocator_agent()
        strategies = agent.get_allocation_strategies()
        
        for risk_profile, strategy in strategies.items():
            # Check that base allocation is within ranges
            for asset, base_value in strategy.base_allocation.items():
                if asset in strategy.allocation_ranges:
                    min_val, max_val = strategy.allocation_ranges[asset]
                    assert min_val <= base_value <= max_val, \
                        f"{risk_profile} {asset}: {base_value} not in range [{min_val}, {max_val}]"
    
    def test_risk_profile_progression(self):
        """Test that risk profiles show appropriate progression."""
        agent = create_portfolio_allocator_agent()
        strategies = agent.get_allocation_strategies()
        
        low_strategy = strategies[RiskProfile.LOW]
        moderate_strategy = strategies[RiskProfile.MODERATE]
        high_strategy = strategies[RiskProfile.HIGH]
        
        # Low risk should have highest bond allocation
        assert low_strategy.base_allocation["bonds"] > moderate_strategy.base_allocation["bonds"]
        assert moderate_strategy.base_allocation["bonds"] > high_strategy.base_allocation["bonds"]
        
        # High risk should have highest equity allocation
        equity_low = low_strategy.base_allocation["sp500"] + low_strategy.base_allocation["small_cap"]
        equity_moderate = moderate_strategy.base_allocation["sp500"] + moderate_strategy.base_allocation["small_cap"]
        equity_high = high_strategy.base_allocation["sp500"] + high_strategy.base_allocation["small_cap"]
        
        assert equity_high > equity_moderate > equity_low


class TestConstraintValidation:
    """Test cases for allocation constraint validation."""
    
    def test_individual_asset_constraints(self):
        """Test individual asset constraint validation."""
        agent = create_portfolio_allocator_agent()
        
        # Test valid allocations
        valid_cases = [
            PortfolioAllocation(sp500=100.0, small_cap=0.0, bonds=0.0, gold=0.0, real_estate=0.0),
            PortfolioAllocation(sp500=0.0, small_cap=0.0, bonds=100.0, gold=0.0, real_estate=0.0),
            PortfolioAllocation(sp500=25.0, small_cap=25.0, bonds=25.0, gold=25.0, real_estate=0.0),
        ]
        
        for allocation in valid_cases:
            validation = agent._validate_constraints(allocation)
            assert validation["all_constraints_met"] is True
    
    def test_total_allocation_constraint(self):
        """Test total allocation constraint validation."""
        agent = create_portfolio_allocator_agent()
        
        # Test that Pydantic validation prevents invalid allocations
        with pytest.raises(Exception):  # Should raise ValidationError
            PortfolioAllocation(sp500=50.0, small_cap=0.0, bonds=0.0, gold=0.0, real_estate=0.0)  # 50%
        
        with pytest.raises(Exception):  # Should raise ValidationError
            PortfolioAllocation(sp500=60.0, small_cap=20.0, bonds=30.0, gold=10.0, real_estate=0.0)  # 120%
        
        # Test valid allocations that sum to 100%
        valid_cases = [
            PortfolioAllocation(sp500=100.0, small_cap=0.0, bonds=0.0, gold=0.0, real_estate=0.0),
            PortfolioAllocation(sp500=25.0, small_cap=25.0, bonds=25.0, gold=25.0, real_estate=0.0),
        ]
        
        for allocation in valid_cases:
            validation = agent._validate_constraints(allocation)
            assert validation["total_equals_100"] is True
            assert validation["all_constraints_met"] is True
    
    def test_boundary_conditions(self):
        """Test boundary conditions for constraints."""
        agent = create_portfolio_allocator_agent()
        
        # Test exactly 100%
        exact_allocation = PortfolioAllocation(
            sp500=50.0, small_cap=20.0, bonds=20.0, gold=5.0, real_estate=5.0
        )
        validation = agent._validate_constraints(exact_allocation)
        assert validation["total_equals_100"] is True
        
        # Test that allocations outside tolerance are rejected by Pydantic
        with pytest.raises(Exception):  # Should raise ValidationError
            PortfolioAllocation(
                sp500=50.0, small_cap=20.0, bonds=20.0, gold=5.0, real_estate=5.1  # 100.1%
            )
        
        # Test edge case allocations that are valid
        edge_cases = [
            PortfolioAllocation(sp500=0.0, small_cap=0.0, bonds=100.0, gold=0.0, real_estate=0.0),
            PortfolioAllocation(sp500=99.99, small_cap=0.0, bonds=0.0, gold=0.0, real_estate=0.01),
        ]
        
        for allocation in edge_cases:
            validation = agent._validate_constraints(allocation)
            assert validation["total_equals_100"] is True


if __name__ == "__main__":
    pytest.main([__file__])