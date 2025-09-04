"""
Unit tests for Portfolio Allocation Agent

Tests the portfolio allocation agent functionality including
risk-based allocation and optimization logic.
"""

import pytest
import logging
from unittest.mock import Mock, patch
from typing import Dict, Any

from agents.portfolio_allocation_agent import PortfolioAllocationAgent, PortfolioAllocationInput, PortfolioAllocationOutput


class TestPortfolioAllocationAgent:
    """Test suite for Portfolio Allocation Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create portfolio allocation agent for testing"""
        return PortfolioAllocationAgent()
    
    @pytest.fixture
    def sample_predicted_returns(self):
        """Sample predicted returns for testing"""
        return {
            'sp500': 0.10,
            'small_cap': 0.11,
            't_bills': 0.03,
            't_bonds': 0.05,
            'corporate_bonds': 0.06,
            'real_estate': 0.08,
            'gold': 0.07
        }
    
    @pytest.fixture
    def sample_state_low_risk(self, sample_predicted_returns):
        """Sample state for low risk profile"""
        return {
            'risk_profile': 'low',
            'predicted_returns': sample_predicted_returns,
            'investment_amount': 100000.0,
            'investment_horizon': 10
        }
    
    @pytest.fixture
    def sample_state_moderate_risk(self, sample_predicted_returns):
        """Sample state for moderate risk profile"""
        return {
            'risk_profile': 'moderate',
            'predicted_returns': sample_predicted_returns,
            'investment_amount': 100000.0,
            'investment_horizon': 10
        }
    
    @pytest.fixture
    def sample_state_high_risk(self, sample_predicted_returns):
        """Sample state for high risk profile"""
        return {
            'risk_profile': 'high',
            'predicted_returns': sample_predicted_returns,
            'investment_amount': 100000.0,
            'investment_horizon': 10
        }
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        agent = PortfolioAllocationAgent()
        
        assert agent.name == "portfolio_allocation_agent"
        assert 'low' in agent.base_allocations
        assert 'moderate' in agent.base_allocations
        assert 'high' in agent.base_allocations
        
        # Verify base allocations sum to 100%
        for risk_profile, allocation in agent.base_allocations.items():
            total = sum(allocation.values())
            assert abs(total - 100.0) < 0.01
    
    def test_allocate_portfolio_low_risk(self, agent, sample_state_low_risk):
        """Test portfolio allocation for low risk profile"""
        result = agent.allocate_portfolio(sample_state_low_risk)
        
        # Verify successful completion
        assert result['agent_status'] == 'portfolio_allocation_complete'
        assert 'error' not in result
        
        # Verify allocation structure
        allocation = result['portfolio_allocation']
        assert isinstance(allocation, dict)
        assert abs(sum(allocation.values()) - 100.0) < 0.01
        
        # Verify low risk characteristics (more bonds, less equity)
        equity_allocation = allocation.get('sp500', 0) + allocation.get('small_cap', 0)
        bond_allocation = (allocation.get('t_bills', 0) + 
                          allocation.get('t_bonds', 0) + 
                          allocation.get('corporate_bonds', 0))
        
        assert bond_allocation > equity_allocation  # More bonds for low risk
        assert equity_allocation < 30  # Limited equity exposure
        
        # Verify expected return is calculated
        assert 'expected_portfolio_return' in result
        assert isinstance(result['expected_portfolio_return'], float)
        assert result['expected_portfolio_return'] > 0
        
        # Verify rationale is provided
        assert 'allocation_rationale' in result
        assert 'low' in result['allocation_rationale'].lower()
    
    def test_allocate_portfolio_moderate_risk(self, agent, sample_state_moderate_risk):
        """Test portfolio allocation for moderate risk profile"""
        result = agent.allocate_portfolio(sample_state_moderate_risk)
        
        # Verify successful completion
        assert result['agent_status'] == 'portfolio_allocation_complete'
        
        # Verify allocation structure
        allocation = result['portfolio_allocation']
        assert abs(sum(allocation.values()) - 100.0) < 0.01
        
        # Verify moderate risk characteristics (balanced allocation)
        equity_allocation = allocation.get('sp500', 0) + allocation.get('small_cap', 0)
        bond_allocation = (allocation.get('t_bills', 0) + 
                          allocation.get('t_bonds', 0) + 
                          allocation.get('corporate_bonds', 0))
        
        # Should be more balanced
        assert 25 <= equity_allocation <= 55  # Moderate equity exposure
        assert 30 <= bond_allocation <= 60   # Moderate bond exposure
        
        # Verify rationale mentions moderate/balanced approach
        assert any(keyword in result['allocation_rationale'].lower() 
                  for keyword in ['moderate', 'balanced'])
    
    def test_allocate_portfolio_high_risk(self, agent, sample_state_high_risk):
        """Test portfolio allocation for high risk profile"""
        result = agent.allocate_portfolio(sample_state_high_risk)
        
        # Verify successful completion
        assert result['agent_status'] == 'portfolio_allocation_complete'
        
        # Verify allocation structure
        allocation = result['portfolio_allocation']
        assert abs(sum(allocation.values()) - 100.0) < 0.01
        
        # Verify high risk characteristics (more equity, less bonds)
        equity_allocation = allocation.get('sp500', 0) + allocation.get('small_cap', 0)
        bond_allocation = (allocation.get('t_bills', 0) + 
                          allocation.get('t_bonds', 0) + 
                          allocation.get('corporate_bonds', 0))
        
        assert equity_allocation > bond_allocation  # More equity for high risk
        assert equity_allocation > 50  # Significant equity exposure
        
        # Verify rationale mentions growth/aggressive approach
        assert any(keyword in result['allocation_rationale'].lower() 
                  for keyword in ['growth', 'aggressive'])
    
    def test_allocate_portfolio_invalid_risk_profile(self, agent, sample_predicted_returns):
        """Test handling of invalid risk profile"""
        state = {
            'risk_profile': 'invalid_profile',
            'predicted_returns': sample_predicted_returns,
            'investment_amount': 100000.0,
            'investment_horizon': 10
        }
        
        result = agent.allocate_portfolio(state)
        
        # Should default to moderate and complete successfully
        assert result['agent_status'] == 'portfolio_allocation_complete'
        assert result.get('risk_level') == 'moderate'
    
    def test_allocate_portfolio_no_predicted_returns(self, agent):
        """Test allocation without predicted returns"""
        state = {
            'risk_profile': 'moderate',
            'predicted_returns': {},
            'investment_amount': 100000.0,
            'investment_horizon': 10
        }
        
        result = agent.allocate_portfolio(state)
        
        # Should complete successfully with base allocation
        assert result['agent_status'] == 'portfolio_allocation_complete'
        assert 'portfolio_allocation' in result
    
    def test_optimize_allocation(self, agent, sample_predicted_returns):
        """Test allocation optimization based on predicted returns"""
        base_allocation = agent.base_allocations['moderate'].copy()
        
        # Test optimization
        optimized = agent._optimize_allocation(
            base_allocation, sample_predicted_returns, 'moderate', 10
        )
        
        # Verify optimization maintains reasonable bounds
        assert isinstance(optimized, dict)
        assert len(optimized) == len(base_allocation)
        
        # All weights should be non-negative
        assert all(weight >= 0 for weight in optimized.values())
        
        # Should favor assets with higher predicted returns (within bounds)
        # Small cap has highest return (11%), should get some boost
        assert optimized['small_cap'] >= base_allocation['small_cap'] * 0.9
    
    def test_optimize_allocation_no_returns(self, agent):
        """Test optimization without predicted returns"""
        base_allocation = agent.base_allocations['moderate'].copy()
        
        optimized = agent._optimize_allocation(base_allocation, {}, 'moderate', 10)
        
        # Should return base allocation unchanged
        assert optimized == base_allocation
    
    def test_get_weight_bounds(self, agent):
        """Test weight bounds for different assets and risk profiles"""
        # Test low risk bounds
        min_weight, max_weight = agent._get_weight_bounds('sp500', 'low')
        assert min_weight == 5.0
        assert max_weight == 25.0
        
        # Test high risk bounds
        min_weight, max_weight = agent._get_weight_bounds('sp500', 'high')
        assert min_weight == 25.0
        assert max_weight == 60.0
        
        # Test bond bounds for low risk
        min_weight, max_weight = agent._get_weight_bounds('t_bonds', 'low')
        assert min_weight >= 15.0  # Should allow significant bond allocation
        
        # Test unknown asset
        min_weight, max_weight = agent._get_weight_bounds('unknown_asset', 'moderate')
        assert min_weight == 0.0
        assert max_weight == 100.0
    
    def test_validate_allocation(self, agent):
        """Test allocation validation and normalization"""
        # Test normal allocation
        allocation = {'sp500': 50.0, 'bonds': 30.0, 'other': 20.0}
        validated = agent._validate_allocation(allocation)
        
        assert abs(sum(validated.values()) - 100.0) < 0.01
        
        # Test allocation that doesn't sum to 100%
        allocation = {'sp500': 60.0, 'bonds': 30.0, 'other': 15.0}  # Sums to 105%
        validated = agent._validate_allocation(allocation)
        
        assert abs(sum(validated.values()) - 100.0) < 0.01
        
        # Test allocation with negative values
        allocation = {'sp500': 60.0, 'bonds': -10.0, 'other': 50.0}
        validated = agent._validate_allocation(allocation)
        
        assert all(weight >= 0 for weight in validated.values())
        assert abs(sum(validated.values()) - 100.0) < 0.01
        
        # Test zero allocation
        allocation = {'sp500': 0.0, 'bonds': 0.0, 'other': 0.0}
        validated = agent._validate_allocation(allocation)
        
        # Should return equal weights as fallback
        assert len(validated) == 3
        assert abs(sum(validated.values()) - 100.0) < 0.01
    
    def test_calculate_expected_return(self, agent, sample_predicted_returns):
        """Test expected portfolio return calculation"""
        allocation = {
            'sp500': 40.0,
            'small_cap': 20.0,
            't_bills': 20.0,
            't_bonds': 20.0
        }
        
        expected_return = agent._calculate_expected_return(allocation, sample_predicted_returns)
        
        # Calculate manually for verification
        manual_calculation = (
            0.40 * 0.10 +  # SP500
            0.20 * 0.11 +  # Small Cap
            0.20 * 0.03 +  # T-Bills
            0.20 * 0.05    # T-Bonds
        )
        
        assert abs(expected_return - manual_calculation) < 0.001
    
    def test_calculate_expected_return_no_predictions(self, agent):
        """Test expected return calculation without predictions"""
        allocation = {'sp500': 50.0, 'bonds': 50.0}
        
        expected_return = agent._calculate_expected_return(allocation, {})
        
        # Should return default 6%
        assert expected_return == 0.06
    
    def test_calculate_expected_return_partial_predictions(self, agent):
        """Test expected return calculation with partial predictions"""
        allocation = {
            'sp500': 50.0,
            'bonds': 30.0,
            'unknown_asset': 20.0
        }
        
        predictions = {'sp500': 0.10, 'bonds': 0.05}  # Missing unknown_asset
        
        expected_return = agent._calculate_expected_return(allocation, predictions)
        
        # Should calculate based on available predictions only
        expected = 0.50 * 0.10 + 0.30 * 0.05  # Only SP500 and bonds
        assert abs(expected_return - expected) < 0.001
    
    def test_generate_allocation_rationale(self, agent):
        """Test allocation rationale generation"""
        allocation = {
            'sp500': 40.0,
            'small_cap': 20.0,
            't_bills': 15.0,
            't_bonds': 15.0,
            'corporate_bonds': 10.0
        }
        
        rationale = agent._generate_allocation_rationale(
            allocation, 'moderate', 0.08, 10
        )
        
        # Verify rationale content
        assert isinstance(rationale, str)
        assert len(rationale) > 0
        assert 'moderate' in rationale.lower()
        assert '10-year' in rationale
        assert 'sp500' in rationale  # Should mention top allocation
        assert '8.00%' in rationale or '0.08' in rationale  # Expected return
    
    def test_create_runnable(self, agent):
        """Test creation of LangChain runnable"""
        runnable = agent.create_runnable()
        
        # Verify runnable is created
        assert runnable is not None
        
        # Verify runnable can be invoked
        sample_state = {
            'risk_profile': 'moderate',
            'predicted_returns': {'sp500': 0.10, 't_bills': 0.03},
            'investment_amount': 50000.0,
            'investment_horizon': 5
        }
        
        result = runnable.invoke(sample_state)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'portfolio_allocation' in result
    
    def test_allocation_edge_cases(self, agent):
        """Test edge cases in allocation"""
        # Test with minimal state
        result = agent.allocate_portfolio({})
        assert result['agent_status'] == 'portfolio_allocation_complete'
        
        # Test with None values
        state = {
            'risk_profile': None,
            'predicted_returns': None,
            'investment_amount': None,
            'investment_horizon': None
        }
        result = agent.allocate_portfolio(state)
        assert result['agent_status'] == 'portfolio_allocation_complete'
    
    def test_allocation_consistency(self, agent, sample_predicted_returns):
        """Test allocation consistency across multiple runs"""
        state = {
            'risk_profile': 'moderate',
            'predicted_returns': sample_predicted_returns,
            'investment_amount': 100000.0,
            'investment_horizon': 10
        }
        
        # Run allocation multiple times
        results = []
        for _ in range(5):
            result = agent.allocate_portfolio(state.copy())
            results.append(result['portfolio_allocation'])
        
        # Results should be consistent (same inputs -> same outputs)
        for i in range(1, len(results)):
            for asset in results[0]:
                assert abs(results[0][asset] - results[i][asset]) < 0.01
    
    def test_allocation_logging(self, agent, sample_state_moderate_risk, caplog):
        """Test logging during allocation process"""
        with caplog.at_level(logging.INFO):
            agent.allocate_portfolio(sample_state_moderate_risk)
        
        # Verify logging messages
        assert any("Starting portfolio allocation" in record.message for record in caplog.records)
        assert any("Portfolio allocation completed" in record.message for record in caplog.records)


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    pytest.main([__file__, "-v"])