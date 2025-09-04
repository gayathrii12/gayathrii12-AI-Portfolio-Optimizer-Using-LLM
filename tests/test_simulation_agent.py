"""
Unit tests for the Simulation Agent.

Tests cover lumpsum projections, SIP calculations, withdrawal processing,
CAGR calculations, and Monte Carlo simulation functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from agents.simulation_agent import (
    SimulationAgent,
    SimulationInput,
    SimulationResult,
    InvestmentType,
    CalculateLumpsumProjectionTool,
    CalculateSIPProjectionTool,
    ProcessWithdrawalScheduleTool,
    CalculateCAGRTool
)
from models.data_models import (
    UserInputModel,
    PortfolioAllocation,
    ProjectionResult
)


class TestSimulationAgent:
    """Test cases for SimulationAgent class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = SimulationAgent()
        
        # Sample user input for lumpsum investment
        self.lumpsum_user_input = UserInputModel(
            investment_amount=100000,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=8.0
        )
        
        # Sample user input for SIP investment
        self.sip_user_input = UserInputModel(
            investment_amount=5000,  # Monthly SIP amount
            investment_type="sip",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=8.0
        )
        
        # Sample portfolio allocation
        self.portfolio_allocation = PortfolioAllocation(
            sp500=45.0,
            small_cap=10.0,
            bonds=30.0,
            real_estate=10.0,
            gold=5.0
        )
        
        # Sample expected returns
        self.expected_returns = {
            'sp500': 0.10,
            'small_cap': 0.12,
            't_bonds': 0.05,
            'corporate_bonds': 0.06,
            'real_estate': 0.09,
            'gold': 0.07
        }
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = SimulationAgent()
        
        assert agent.llm is None
        assert len(agent.tools) == 4
        assert agent.prompt is not None
        assert agent.agent_executor is None
        
        # Test tool names
        tool_names = [tool.name for tool in agent.tools]
        expected_tools = [
            "calculate_lumpsum_projection",
            "calculate_sip_projection", 
            "process_withdrawal_schedule",
            "calculate_cagr"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names
    
    def test_simulate_portfolio_lumpsum_success(self):
        """Test successful lumpsum portfolio simulation."""
        simulation_input = SimulationInput(
            user_input=self.lumpsum_user_input,
            portfolio_allocation=self.portfolio_allocation,
            expected_returns=self.expected_returns
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        assert result.success is True
        assert len(result.projections) == 10  # 10 years
        assert result.final_value > 100000  # Should grow
        assert result.total_invested == 100000
        assert result.cagr > 0
        assert result.cumulative_return > 0
        assert result.error_message is None
        
        # Check projection structure
        first_projection = result.projections[0]
        assert first_projection.year == 1
        assert first_projection.portfolio_value > 100000
        assert first_projection.annual_return > 0
        assert first_projection.cumulative_return > 0
    
    def test_simulate_portfolio_sip_success(self):
        """Test successful SIP portfolio simulation."""
        simulation_input = SimulationInput(
            user_input=self.sip_user_input,
            portfolio_allocation=self.portfolio_allocation,
            expected_returns=self.expected_returns
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        assert result.success is True
        assert len(result.projections) == 10  # 10 years
        assert result.final_value > 0
        assert result.total_invested == 5000 * 12 * 10  # Monthly * 12 * years
        assert result.cagr > 0
        assert result.cumulative_return > 0
        
        # SIP should show gradual growth
        values = [p.portfolio_value for p in result.projections]
        assert all(values[i] <= values[i+1] for i in range(len(values)-1))  # Monotonic growth
    
    def test_simulate_portfolio_with_withdrawals(self):
        """Test portfolio simulation with withdrawal schedule."""
        user_input_with_withdrawals = UserInputModel(
            investment_amount=100000,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=8.0,
            withdrawal_preferences={
                "annual_withdrawal": 5000,
                "start_year": 5
            }
        )
        
        simulation_input = SimulationInput(
            user_input=user_input_with_withdrawals,
            portfolio_allocation=self.portfolio_allocation,
            expected_returns=self.expected_returns
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        assert result.success is True
        assert result.withdrawal_impact is not None
        assert result.withdrawal_impact["total_withdrawals"] > 0
        assert result.withdrawal_impact["withdrawal_strategy"] == "annual"
    
    def test_calculate_portfolio_return(self):
        """Test portfolio return calculation."""
        portfolio_return = self.agent._calculate_portfolio_return(
            self.portfolio_allocation,
            self.expected_returns
        )
        
        # Expected calculation:
        # 45% * 10% + 10% * 12% + 30% * 5.5% + 10% * 9% + 5% * 7%
        # = 4.5% + 1.2% + 1.65% + 0.9% + 0.35% = 8.6%
        expected_return = 0.086
        assert abs(portfolio_return - expected_return) < 0.001
    
    def test_calculate_lumpsum_projections(self):
        """Test lumpsum projection calculations."""
        projections = self.agent._calculate_lumpsum_projections(
            investment_amount=100000,
            tenure_years=5,
            annual_return=0.08
        )
        
        assert len(projections) == 5
        
        # Test compound growth
        expected_values = [
            108000,   # Year 1: 100000 * 1.08
            116640,   # Year 2: 108000 * 1.08
            125971.2, # Year 3: 116640 * 1.08
            136048.9, # Year 4: 125971.2 * 1.08
            146932.8  # Year 5: 136048.9 * 1.08
        ]
        
        for i, projection in enumerate(projections):
            assert projection.year == i + 1
            assert abs(projection.portfolio_value - expected_values[i]) < 1  # Allow small rounding differences
            assert projection.annual_return == 8.0  # 8% annual return
    
    def test_calculate_sip_projections(self):
        """Test SIP projection calculations."""
        projections = self.agent._calculate_sip_projections(
            monthly_investment=1000,
            tenure_years=2,
            annual_return=0.12  # 12% annual = 1% monthly
        )
        
        assert len(projections) == 2
        
        # First year should have 12 months of investments with compound growth
        first_year = projections[0]
        assert first_year.year == 1
        assert first_year.portfolio_value > 12000  # More than just 12 * 1000 due to compounding
        
        # Second year should be higher
        second_year = projections[1]
        assert second_year.year == 2
        assert second_year.portfolio_value > first_year.portfolio_value
    
    def test_calculate_final_metrics_lumpsum(self):
        """Test final metrics calculation for lumpsum investment."""
        # Create sample projections that represent 8% annual growth
        # Year 1: 100000 * 1.08 = 108000
        # Year 2: 100000 * 1.08^2 = 116640
        projections = [
            ProjectionResult(year=1, portfolio_value=108000, annual_return=8.0, cumulative_return=8.0),
            ProjectionResult(year=2, portfolio_value=116640, annual_return=8.0, cumulative_return=16.64)
        ]
        
        # Create a 2-year lumpsum user input for this test
        test_user_input = UserInputModel(
            investment_amount=100000,
            investment_type="lumpsum",
            tenure_years=2,  # 2 years to match projections
            risk_profile="Moderate",
            return_expectation=8.0
        )
        
        metrics = self.agent._calculate_final_metrics(
            projections=projections,
            withdrawal_impact=None,
            user_input=test_user_input
        )
        
        assert metrics["final_value"] == 116640
        assert metrics["total_invested"] == 100000
        # CAGR should be 8% for lumpsum: (116640/100000)^(1/2) - 1 = 0.08
        assert abs(metrics["cagr"] - 8.0) < 0.1  # Should be close to 8%
        assert abs(metrics["cumulative_return"] - 16.64) < 0.1
    
    def test_calculate_final_metrics_sip(self):
        """Test final metrics calculation for SIP investment."""
        # Create realistic SIP projections for 2 years
        projections = [
            ProjectionResult(year=1, portfolio_value=62000, annual_return=8.0, cumulative_return=3.33),
            ProjectionResult(year=2, portfolio_value=130000, annual_return=8.0, cumulative_return=8.33)
        ]
        
        # Create a 2-year SIP user input for this test
        test_sip_user_input = UserInputModel(
            investment_amount=5000,  # Monthly SIP amount
            investment_type="sip",
            tenure_years=2,  # 2 years to match projections
            risk_profile="Moderate",
            return_expectation=8.0
        )
        
        metrics = self.agent._calculate_final_metrics(
            projections=projections,
            withdrawal_impact=None,
            user_input=test_sip_user_input
        )
        
        assert metrics["final_value"] == 130000
        assert metrics["total_invested"] == 5000 * 12 * 2  # Monthly * 12 * 2 years = 120000
        assert metrics["cagr"] >= 0  # Should be positive
        assert metrics["cumulative_return"] >= 0  # Should be positive (130000-120000)/120000 * 100 = 8.33%
    
    def test_run_monte_carlo_simulation(self):
        """Test Monte Carlo simulation."""
        simulation_input = SimulationInput(
            user_input=self.lumpsum_user_input,
            portfolio_allocation=self.portfolio_allocation,
            expected_returns=self.expected_returns,
            simulation_runs=100
        )
        
        stats = self.agent._run_monte_carlo_simulation(simulation_input)
        
        assert "mean_return" in stats
        assert "median_return" in stats
        assert "std_deviation" in stats
        assert "percentile_5" in stats
        assert "percentile_95" in stats
        assert "probability_positive" in stats
        assert stats["simulation_runs"] == 100
        
        # Mean should be close to expected portfolio return
        expected_return = self.agent._calculate_portfolio_return(
            self.portfolio_allocation,
            self.expected_returns
        )
        assert abs(stats["mean_return"] - expected_return) < 0.02  # Within 2%
    
    def test_process_withdrawal_schedule_none(self):
        """Test withdrawal processing with no withdrawals."""
        result = self.agent._process_withdrawal_schedule([], None)
        assert result is None
    
    def test_process_withdrawal_schedule_annual(self):
        """Test withdrawal processing with annual withdrawals."""
        projections = [
            ProjectionResult(year=i, portfolio_value=100000 + i*10000, 
                           annual_return=8.0, cumulative_return=i*8.0)
            for i in range(1, 6)
        ]
        
        withdrawal_prefs = {
            "annual_withdrawal": 5000,
            "start_year": 3
        }
        
        result = self.agent._process_withdrawal_schedule(projections, withdrawal_prefs)
        
        assert result is not None
        assert result["withdrawal_strategy"] == "annual"
        assert result["total_withdrawals"] > 0
        assert result["impact_on_final_value"] < 0  # Negative impact
    
    def test_simulation_input_validation(self):
        """Test simulation input validation."""
        # Valid input
        valid_input = SimulationInput(
            user_input=self.lumpsum_user_input,
            portfolio_allocation=self.portfolio_allocation,
            expected_returns=self.expected_returns,
            simulation_runs=500
        )
        assert valid_input.simulation_runs == 500
        
        # Test simulation runs bounds
        with pytest.raises(ValueError):
            SimulationInput(
                user_input=self.lumpsum_user_input,
                portfolio_allocation=self.portfolio_allocation,
                expected_returns=self.expected_returns,
                simulation_runs=50  # Below minimum
            )
        
        with pytest.raises(ValueError):
            SimulationInput(
                user_input=self.lumpsum_user_input,
                portfolio_allocation=self.portfolio_allocation,
                expected_returns=self.expected_returns,
                simulation_runs=15000  # Above maximum
            )
    
    def test_error_handling(self):
        """Test error handling in simulation."""
        # Create valid input but mock an internal method to raise an exception
        simulation_input = SimulationInput(
            user_input=self.lumpsum_user_input,
            portfolio_allocation=self.portfolio_allocation,
            expected_returns={}  # Empty returns
        )
        
        # Mock the internal method to raise an exception
        with patch.object(self.agent, '_calculate_base_projections', side_effect=Exception("Test error")):
            result = self.agent.simulate_portfolio(simulation_input)
            
            assert result.success is False
            assert result.error_message is not None
            assert "Test error" in result.error_message
            assert result.final_value == 0.0
            assert result.total_invested == 0.0


class TestSimulationTools:
    """Test cases for individual simulation tools."""
    
    def test_calculate_lumpsum_projection_tool(self):
        """Test lumpsum projection tool."""
        tool = CalculateLumpsumProjectionTool()
        
        result = tool._run(
            investment_amount=100000,
            tenure_years=5,
            expected_returns="{'sp500': 0.10}",
            allocation="{'sp500': 100}"
        )
        
        assert "Error:" not in result
        assert "projections" in result
        assert "final_value" in result
        assert "total_invested" in result
    
    def test_calculate_sip_projection_tool(self):
        """Test SIP projection tool."""
        tool = CalculateSIPProjectionTool()
        
        result = tool._run(
            monthly_investment=5000,
            tenure_years=3,
            expected_returns="{'sp500': 0.10}",
            allocation="{'sp500': 100}"
        )
        
        assert "Error:" not in result
        assert "projections" in result
        assert "monthly_investment" in result
        assert "calculation_method" in result
    
    def test_process_withdrawal_schedule_tool(self):
        """Test withdrawal schedule processing tool."""
        tool = ProcessWithdrawalScheduleTool()
        
        # Test with no withdrawals
        result = tool._run(
            base_projections="[]",
            withdrawal_preferences="None"
        )
        
        assert "Error:" not in result
        assert "withdrawal_strategy" in result
        
        # Test with withdrawal preferences
        result = tool._run(
            base_projections="[]",
            withdrawal_preferences="{'annual_withdrawal': 5000}"
        )
        
        assert "Error:" not in result
        assert "total_withdrawals" in result
    
    def test_calculate_cagr_tool(self):
        """Test CAGR calculation tool."""
        tool = CalculateCAGRTool()
        
        result = tool._run(
            initial_value=100000,
            final_value=150000,
            tenure_years=5,
            total_invested=100000
        )
        
        assert "Error:" not in result
        assert "cagr" in result
        assert "cumulative_return" in result
        assert "absolute_return" in result
    
    def test_tool_error_handling(self):
        """Test error handling in tools."""
        tool = CalculateCAGRTool()
        
        # Test with invalid inputs that should cause division by zero
        result = tool._run(
            initial_value=0,
            final_value=100000,
            tenure_years=0,
            total_invested=0
        )
        
        # Should handle gracefully and return metrics with zeros
        assert "Error:" not in result or "cagr" in result


class TestSimulationScenarios:
    """Test various simulation scenarios and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = SimulationAgent()
        
        self.portfolio_allocation = PortfolioAllocation(
            sp500=50.0,
            small_cap=20.0,
            bonds=20.0,
            real_estate=5.0,
            gold=5.0
        )
        
        self.expected_returns = {
            'sp500': 0.10,
            'small_cap': 0.12,
            't_bonds': 0.04,
            'corporate_bonds': 0.06,
            'real_estate': 0.08,
            'gold': 0.06
        }
    
    def test_high_risk_portfolio_simulation(self):
        """Test simulation with high-risk portfolio."""
        user_input = UserInputModel(
            investment_amount=50000,
            investment_type="lumpsum",
            tenure_years=15,
            risk_profile="High",
            return_expectation=12.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=self.portfolio_allocation,
            expected_returns=self.expected_returns
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        assert result.success is True
        assert result.cagr > 8.0  # Should have decent returns
        assert len(result.projections) == 15
    
    def test_long_term_sip_simulation(self):
        """Test long-term SIP simulation."""
        user_input = UserInputModel(
            investment_amount=2000,  # Monthly SIP
            investment_type="sip",
            tenure_years=25,
            risk_profile="Moderate",
            return_expectation=10.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=self.portfolio_allocation,
            expected_returns=self.expected_returns
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        assert result.success is True
        assert result.total_invested == 2000 * 12 * 25  # 600,000
        assert result.final_value > result.total_invested  # Should grow
        assert len(result.projections) == 25
    
    def test_conservative_portfolio_simulation(self):
        """Test simulation with conservative portfolio."""
        conservative_allocation = PortfolioAllocation(
            sp500=20.0,
            small_cap=0.0,
            bonds=70.0,
            real_estate=5.0,
            gold=5.0
        )
        
        user_input = UserInputModel(
            investment_amount=100000,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Low",
            return_expectation=5.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=conservative_allocation,
            expected_returns=self.expected_returns
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        assert result.success is True
        # Conservative portfolio should have lower but positive returns
        assert 0 < result.cagr < 8.0
        assert result.final_value > 100000
    
    def test_short_term_investment(self):
        """Test short-term investment simulation."""
        user_input = UserInputModel(
            investment_amount=25000,
            investment_type="lumpsum",
            tenure_years=2,
            risk_profile="Moderate",
            return_expectation=6.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=self.portfolio_allocation,
            expected_returns=self.expected_returns
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        assert result.success is True
        assert len(result.projections) == 2
        assert result.final_value > 25000
        assert result.cagr > 0
    
    def test_zero_expected_returns(self):
        """Test simulation with zero expected returns."""
        zero_returns = {asset: 0.0 for asset in self.expected_returns.keys()}
        
        user_input = UserInputModel(
            investment_amount=100000,
            investment_type="lumpsum",
            tenure_years=5,
            risk_profile="Moderate",
            return_expectation=0.0
        )
        
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=self.portfolio_allocation,
            expected_returns=zero_returns
        )
        
        result = self.agent.simulate_portfolio(simulation_input)
        
        assert result.success is True
        assert result.final_value == 100000  # No growth
        assert result.cagr == 0.0
        assert result.cumulative_return == 0.0


if __name__ == "__main__":
    pytest.main([__file__])