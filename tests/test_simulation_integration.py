"""
Integration tests for the Simulation Agent with sample investment parameters.

These tests verify the complete simulation pipeline with realistic scenarios
and validate integration with other system components.
"""

import pytest
import numpy as np
from typing import List

from agents.simulation_agent import SimulationAgent, SimulationInput
from models.data_models import (
    UserInputModel,
    PortfolioAllocation,
    AssetReturns,
    ProjectionResult
)


class TestSimulationIntegration:
    """Integration tests for complete simulation workflows."""
    
    def setup_method(self):
        """Set up test fixtures with realistic data."""
        self.agent = SimulationAgent()
        
        # Sample historical data (simplified)
        self.historical_data = [
            AssetReturns(sp500=10.5, small_cap=12.3, t_bills=2.1, t_bonds=4.8, 
                        corporate_bonds=5.9, real_estate=8.7, gold=6.2, year=2020),
            AssetReturns(sp500=28.7, small_cap=31.3, t_bills=0.4, t_bonds=7.5, 
                        corporate_bonds=9.9, real_estate=7.8, gold=24.4, year=2021),
            AssetReturns(sp500=-18.1, small_cap=-20.4, t_bills=1.5, t_bonds=-13.0, 
                        corporate_bonds=-15.8, real_estate=-25.9, gold=-0.3, year=2022),
            AssetReturns(sp500=26.3, small_cap=16.9, t_bills=5.0, t_bonds=5.5, 
                        corporate_bonds=8.5, real_estate=11.5, gold=13.1, year=2023),
        ]
        
        # Expected returns based on historical averages
        self.expected_returns = {
            'sp500': 0.095,
            'small_cap': 0.11,
            't_bills': 0.025,
            't_bonds': 0.048,
            'corporate_bonds': 0.057,
            'real_estate': 0.085,
            'gold': 0.065
        }
    
    def test_young_professional_aggressive_portfolio(self):
        """Test simulation for young professional with aggressive portfolio."""
        # Scenario: 25-year-old starting career, high risk tolerance
        user_input = UserInputModel(
            investment_amount=10000,  # Monthly SIP
            investment_type="sip",
            tenure_years=30,
            risk_profile="High",
            return_expectation=12.0
        )
        
        # Aggressive allocation
        aggressive_allocation = PortfolioAllocation(
            sp500=60.0,
            small_cap=25.0,
            bonds=5.0,
            real_estate=8.0,
            gold=2.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=aggressive_allocation,
            expected_returns=self.expected_returns,
            simulation_runs=1000
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        # Validate results
        assert result.success is True
        assert len(result.projections) == 30
        assert result.total_invested == 10000 * 12 * 30  # 3.6M invested
        assert result.final_value > result.total_invested  # Should grow significantly
        assert result.cagr > 5.0  # Should have good returns with aggressive allocation
        
        # Check simulation statistics
        assert result.simulation_statistics["simulation_runs"] == 1000
        assert result.simulation_statistics["probability_positive"] > 0.7  # High probability of positive returns
        
        # Validate year-over-year growth
        values = [p.portfolio_value for p in result.projections]
        assert all(values[i] <= values[i+1] for i in range(len(values)-1))  # Monotonic growth for SIP
    
    def test_mid_career_balanced_portfolio(self):
        """Test simulation for mid-career professional with balanced portfolio."""
        # Scenario: 40-year-old with moderate risk tolerance
        user_input = UserInputModel(
            investment_amount=500000,  # Lumpsum investment
            investment_type="lumpsum",
            tenure_years=20,
            risk_profile="Moderate",
            return_expectation=8.0
        )
        
        # Balanced allocation
        balanced_allocation = PortfolioAllocation(
            sp500=45.0,
            small_cap=10.0,
            bonds=30.0,
            real_estate=10.0,
            gold=5.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=balanced_allocation,
            expected_returns=self.expected_returns,
            simulation_runs=500
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        # Validate results
        assert result.success is True
        assert len(result.projections) == 20
        assert result.total_invested == 500000
        assert result.final_value > 500000
        assert 6.0 < result.cagr < 10.0  # Moderate returns
        
        # Check that final value is reasonable for 20-year investment
        expected_range_low = 500000 * (1.06 ** 20)  # 6% CAGR
        expected_range_high = 500000 * (1.10 ** 20)  # 10% CAGR
        assert expected_range_low <= result.final_value <= expected_range_high * 1.2  # Allow some variance
    
    def test_pre_retirement_conservative_portfolio(self):
        """Test simulation for pre-retirement conservative portfolio."""
        # Scenario: 55-year-old nearing retirement, low risk tolerance
        user_input = UserInputModel(
            investment_amount=1000000,  # Lumpsum investment
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Low",
            return_expectation=5.0,
            withdrawal_preferences={
                "annual_withdrawal": 40000,
                "start_year": 5
            }
        )
        
        # Conservative allocation
        conservative_allocation = PortfolioAllocation(
            sp500=20.0,
            small_cap=0.0,
            bonds=65.0,
            real_estate=10.0,
            gold=5.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=conservative_allocation,
            expected_returns=self.expected_returns,
            simulation_runs=500
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        # Validate results
        assert result.success is True
        assert len(result.projections) == 10
        assert result.total_invested == 1000000
        assert result.cagr > 0  # Should have positive returns
        assert result.cagr < 8.0  # Conservative returns
        
        # Check withdrawal impact
        assert result.withdrawal_impact is not None
        assert result.withdrawal_impact["total_withdrawals"] > 0
        assert result.withdrawal_impact["withdrawal_strategy"] == "annual"
    
    def test_retirement_income_focused_portfolio(self):
        """Test simulation for retirement income-focused portfolio."""
        # Scenario: Retiree needing regular income
        user_input = UserInputModel(
            investment_amount=800000,  # Lumpsum retirement corpus
            investment_type="lumpsum",
            tenure_years=25,  # Long retirement period
            risk_profile="Low",
            return_expectation=4.0,
            withdrawal_preferences={
                "annual_withdrawal": 50000,  # 6.25% withdrawal rate
                "start_year": 1
            }
        )
        
        # Income-focused allocation
        income_allocation = PortfolioAllocation(
            sp500=15.0,
            small_cap=0.0,
            bonds=70.0,
            real_estate=10.0,
            gold=5.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=income_allocation,
            expected_returns=self.expected_returns,
            simulation_runs=500
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        # Validate results
        assert result.success is True
        assert len(result.projections) == 25
        assert result.withdrawal_impact is not None
        assert result.withdrawal_impact["total_withdrawals"] == 50000 * 25  # 25 years of withdrawals
        
        # Portfolio should still have value after withdrawals
        assert result.final_value > 0
    
    def test_education_savings_plan(self):
        """Test simulation for education savings plan."""
        # Scenario: Parent saving for child's education
        user_input = UserInputModel(
            investment_amount=1500,  # Monthly SIP
            investment_type="sip",
            tenure_years=15,  # Child is 3, college at 18
            risk_profile="Moderate",
            return_expectation=8.0
        )
        
        # Education-focused allocation
        education_allocation = PortfolioAllocation(
            sp500=50.0,
            small_cap=15.0,
            bonds=25.0,
            real_estate=5.0,
            gold=5.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=education_allocation,
            expected_returns=self.expected_returns,
            simulation_runs=500
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        # Validate results
        assert result.success is True
        assert len(result.projections) == 15
        assert result.total_invested == 1500 * 12 * 15  # 270,000
        assert result.final_value > result.total_invested
        
        # Should accumulate reasonable amount for education
        assert result.final_value > 400000  # Should grow significantly
    
    def test_house_down_payment_savings(self):
        """Test simulation for house down payment savings."""
        # Scenario: Young couple saving for house down payment
        user_input = UserInputModel(
            investment_amount=3000,  # Monthly SIP
            investment_type="sip",
            tenure_years=5,  # Short-term goal
            risk_profile="Moderate",
            return_expectation=6.0
        )
        
        # Moderate allocation for short-term goal
        moderate_allocation = PortfolioAllocation(
            sp500=35.0,
            small_cap=5.0,
            bonds=45.0,
            real_estate=10.0,
            gold=5.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=moderate_allocation,
            expected_returns=self.expected_returns,
            simulation_runs=500
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        # Validate results
        assert result.success is True
        assert len(result.projections) == 5
        assert result.total_invested == 3000 * 12 * 5  # 180,000
        assert result.final_value > result.total_invested
        
        # Should have reasonable growth for short-term goal
        assert result.cagr > 3.5  # Should beat inflation
    
    def test_wealth_building_high_net_worth(self):
        """Test simulation for high net worth wealth building."""
        # Scenario: High net worth individual building wealth
        user_input = UserInputModel(
            investment_amount=2000000,  # Large lumpsum
            investment_type="lumpsum",
            tenure_years=15,
            risk_profile="High",
            return_expectation=10.0
        )
        
        # Growth-focused allocation
        growth_allocation = PortfolioAllocation(
            sp500=55.0,
            small_cap=20.0,
            bonds=10.0,
            real_estate=12.0,
            gold=3.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=growth_allocation,
            expected_returns=self.expected_returns,
            simulation_runs=1000
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        # Validate results
        assert result.success is True
        assert len(result.projections) == 15
        assert result.total_invested == 2000000
        assert result.final_value > 4000000  # Should double at minimum
        assert result.cagr > 7.0  # Should have strong returns
        
        # Check simulation statistics for risk analysis
        assert result.simulation_statistics["std_deviation"] > 0.05  # Should show volatility
        assert result.simulation_statistics["percentile_5"] < result.simulation_statistics["percentile_95"]
    
    def test_market_crash_scenario_resilience(self):
        """Test portfolio resilience during market crash scenarios."""
        # Scenario: Investment during volatile market conditions
        user_input = UserInputModel(
            investment_amount=100000,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=7.0
        )
        
        # Diversified allocation for resilience
        resilient_allocation = PortfolioAllocation(
            sp500=30.0,
            small_cap=5.0,
            bonds=40.0,
            real_estate=15.0,
            gold=10.0
        )
        
        # Simulate with lower expected returns (market stress)
        stressed_returns = {
            'sp500': 0.06,  # Reduced from normal
            'small_cap': 0.08,
            't_bills': 0.02,
            't_bonds': 0.04,
            'corporate_bonds': 0.05,
            'real_estate': 0.06,
            'gold': 0.08  # Gold performs better in stress
        }
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=resilient_allocation,
            expected_returns=stressed_returns,
            simulation_runs=1000
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        # Validate results
        assert result.success is True
        assert result.final_value > 100000  # Should still grow despite stress
        assert result.cagr > 3.0  # Should beat inflation even in stress
        
        # Diversification should provide some protection
        assert result.simulation_statistics["probability_positive"] > 0.6
    
    def test_inflation_hedging_portfolio(self):
        """Test portfolio designed for inflation hedging."""
        # Scenario: High inflation environment
        user_input = UserInputModel(
            investment_amount=250000,
            investment_type="lumpsum",
            tenure_years=12,
            risk_profile="Moderate",
            return_expectation=9.0
        )
        
        # Inflation-hedging allocation
        inflation_hedge_allocation = PortfolioAllocation(
            sp500=35.0,
            small_cap=10.0,
            bonds=20.0,  # Reduced bonds in high inflation
            real_estate=25.0,  # Increased real estate
            gold=10.0  # Increased gold
        )
        
        # Higher expected returns to account for inflation
        inflation_adjusted_returns = {
            'sp500': 0.11,
            'small_cap': 0.13,
            't_bills': 0.04,
            't_bonds': 0.05,
            'corporate_bonds': 0.06,
            'real_estate': 0.10,  # Real estate benefits from inflation
            'gold': 0.08  # Gold hedge against inflation
        }
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=inflation_hedge_allocation,
            expected_returns=inflation_adjusted_returns,
            simulation_runs=500
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        # Validate results
        assert result.success is True
        assert result.cagr > 6.0  # Should beat inflation
        assert result.final_value > 250000 * 1.5  # Should grow significantly
    
    def test_simulation_consistency_across_runs(self):
        """Test that simulation results are consistent across multiple runs."""
        user_input = UserInputModel(
            investment_amount=100000,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=8.0
        )
        
        allocation = PortfolioAllocation(
            sp500=50.0,
            small_cap=10.0,
            bonds=30.0,
            real_estate=5.0,
            gold=5.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=allocation,
            expected_returns=self.expected_returns,
            simulation_runs=100
        )
        
        # Run simulation multiple times
        results = []
        for _ in range(5):
            result = self.agent.simulate_portfolio(simulation_input)
            results.append(result)
        
        # All runs should succeed
        assert all(r.success for r in results)
        
        # Results should be similar (deterministic base calculation)
        final_values = [r.final_value for r in results]
        cagrs = [r.cagr for r in results]
        
        # Base projections should be identical (deterministic)
        assert len(set(final_values)) == 1  # All final values should be the same
        assert len(set(cagrs)) == 1  # All CAGRs should be the same
        
        # Monte Carlo stats may vary slightly but should be in reasonable range
        mean_returns = [r.simulation_statistics["mean_return"] for r in results]
        assert max(mean_returns) - min(mean_returns) < 0.01  # Small variance due to random seed


class TestSimulationEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = SimulationAgent()
        
        self.basic_allocation = PortfolioAllocation(
            sp500=60.0,
            small_cap=10.0,
            bonds=20.0,
            real_estate=5.0,
            gold=5.0
        )
        
        self.basic_returns = {
            'sp500': 0.08,
            'small_cap': 0.10,
            't_bills': 0.02,
            't_bonds': 0.04,
            'corporate_bonds': 0.05,
            'real_estate': 0.07,
            'gold': 0.05
        }
    
    def test_minimum_investment_amount(self):
        """Test simulation with minimum investment amount."""
        user_input = UserInputModel(
            investment_amount=1,  # Minimum amount
            investment_type="lumpsum",
            tenure_years=1,
            risk_profile="Moderate",
            return_expectation=5.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=self.basic_allocation,
            expected_returns=self.basic_returns
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        assert result.success is True
        assert result.final_value > 1
        assert result.total_invested == 1
    
    def test_maximum_tenure(self):
        """Test simulation with maximum tenure."""
        user_input = UserInputModel(
            investment_amount=10000,
            investment_type="lumpsum",
            tenure_years=50,  # Maximum tenure
            risk_profile="Moderate",
            return_expectation=8.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=self.basic_allocation,
            expected_returns=self.basic_returns
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        assert result.success is True
        assert len(result.projections) == 50
        assert result.final_value > 100000  # Should grow significantly over 50 years
    
    def test_zero_expected_returns_all_assets(self):
        """Test simulation with zero expected returns for all assets."""
        zero_returns = {asset: 0.0 for asset in self.basic_returns.keys()}
        
        user_input = UserInputModel(
            investment_amount=50000,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=0.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=self.basic_allocation,
            expected_returns=zero_returns
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        assert result.success is True
        assert result.final_value == 50000  # No growth
        assert result.cagr == 0.0
        assert result.cumulative_return == 0.0
    
    def test_extreme_allocation_single_asset(self):
        """Test simulation with 100% allocation to single asset."""
        single_asset_allocation = PortfolioAllocation(
            sp500=100.0,
            small_cap=0.0,
            bonds=0.0,
            real_estate=0.0,
            gold=0.0
        )
        
        user_input = UserInputModel(
            investment_amount=75000,
            investment_type="lumpsum",
            tenure_years=8,
            risk_profile="High",
            return_expectation=10.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=single_asset_allocation,
            expected_returns=self.basic_returns
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        assert result.success is True
        # Portfolio return should equal SP500 return
        expected_portfolio_return = self.basic_returns['sp500']
        calculated_return = self.agent._calculate_portfolio_return(
            single_asset_allocation, self.basic_returns
        )
        assert abs(calculated_return - expected_portfolio_return) < 0.001


if __name__ == "__main__":
    pytest.main([__file__])