"""
Unit tests for terminal output generator functionality.

Tests formatting functions for portfolio allocations, projections,
risk metrics, and explanatory text generation.
"""

import pytest
from utils.terminal_output import TerminalOutputGenerator
from models.data_models import (
    PortfolioAllocation,
    ProjectionResult,
    RiskMetrics,
    UserInputModel
)


class TestTerminalOutputGenerator:
    """Test cases for TerminalOutputGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = TerminalOutputGenerator()
        
        # Sample user input
        self.user_input = UserInputModel(
            investment_amount=100000.0,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=8.0
        )
        
        # Sample portfolio allocation
        self.allocation = PortfolioAllocation(
            sp500=40.0,
            small_cap=10.0,
            bonds=30.0,
            real_estate=15.0,
            gold=5.0
        )
        
        # Sample projections
        self.projections = [
            ProjectionResult(
                year=1,
                portfolio_value=108000.0,
                annual_return=8.0,
                cumulative_return=8.0
            ),
            ProjectionResult(
                year=2,
                portfolio_value=116640.0,
                annual_return=8.0,
                cumulative_return=16.64
            ),
            ProjectionResult(
                year=3,
                portfolio_value=125971.2,
                annual_return=8.0,
                cumulative_return=25.97
            )
        ]
        
        # Sample risk metrics
        self.risk_metrics = RiskMetrics(
            alpha=1.5,
            beta=0.85,
            volatility=12.5,
            sharpe_ratio=0.75,
            max_drawdown=-15.2
        )
    
    def test_format_portfolio_allocation_basic(self):
        """Test basic portfolio allocation formatting."""
        result = self.generator.format_portfolio_allocation(
            self.allocation, self.user_input
        )
        
        # Check header presence
        assert "PORTFOLIO ALLOCATION RECOMMENDATION" in result
        assert "Risk Profile: Moderate" in result
        assert "Investment Amount: $100,000.00" in result
        assert "Investment Type: LUMPSUM" in result
        assert "Investment Tenure: 10 years" in result
        
        # Check allocation table
        assert "S&P 500" in result
        assert "40.0%" in result
        assert "40,000.00" in result
        assert "US Small Cap" in result
        assert "10.0%" in result
        assert "10,000.00" in result
        assert "Bonds (Combined)" in result
        assert "30.0%" in result
        assert "30,000.00" in result
        
        # Check total
        assert "TOTAL" in result
        assert "100.0%" in result
    
    def test_format_portfolio_allocation_sip(self):
        """Test portfolio allocation formatting for SIP investment."""
        sip_input = UserInputModel(
            investment_amount=50000.0,
            investment_type="sip",
            tenure_years=5,
            risk_profile="High",
            return_expectation=12.0
        )
        
        result = self.generator.format_portfolio_allocation(
            self.allocation, sip_input
        )
        
        assert "Investment Type: SIP" in result
        assert "Risk Profile: High" in result
        assert "$50,000.00" in result
        assert "Investment Tenure: 5 years" in result
    
    def test_format_portfolio_allocation_zero_allocations(self):
        """Test allocation formatting with zero allocations."""
        zero_allocation = PortfolioAllocation(
            sp500=60.0,
            small_cap=0.0,
            bonds=40.0,
            real_estate=0.0,
            gold=0.0
        )
        
        result = self.generator.format_portfolio_allocation(
            zero_allocation, self.user_input
        )
        
        # Should show non-zero allocations
        assert "S&P 500" in result
        assert "60.0%" in result
        assert "Bonds (Combined)" in result
        assert "40.0%" in result
        
        # Should not show zero allocations in detail
        lines = result.split('\n')
        allocation_lines = [line for line in lines if '$' in line and 'TOTAL' not in line and 'Investment Amount:' not in line]
        assert len(allocation_lines) == 2  # Only S&P 500 and Bonds
    
    def test_format_year_by_year_projections_basic(self):
        """Test basic year-by-year projections formatting."""
        result = self.generator.format_year_by_year_projections(
            self.projections, self.user_input
        )
        
        # Check header
        assert "PORTFOLIO GROWTH PROJECTIONS" in result
        
        # Check CAGR calculation
        assert "Overall CAGR:" in result
        
        # Check table headers
        assert "Year" in result
        assert "Portfolio Value" in result
        assert "Annual Return" in result
        assert "Cumulative Return" in result
        
        # Check starting value
        assert "$100,000.00" in result
        assert "0.0%" in result
        
        # Check projection values
        assert "$108,000.00" in result
        assert "8.00%" in result
        assert "$116,640.00" in result
        assert "16.64%" in result
        assert "$125,971.20" in result
        assert "25.97%" in result
    
    def test_format_year_by_year_projections_single_year(self):
        """Test projections formatting with single year."""
        single_projection = [self.projections[0]]
        
        result = self.generator.format_year_by_year_projections(
            single_projection, self.user_input
        )
        
        # Should not show CAGR for single year
        assert "Overall CAGR:" not in result
        
        # Should still show the projection
        assert "$108,000.00" in result
        assert "8.00%" in result
    
    def test_format_risk_metrics_basic(self):
        """Test basic risk metrics formatting."""
        result = self.generator.format_risk_metrics(self.risk_metrics)
        
        # Check header
        assert "RISK ANALYSIS & BENCHMARK COMPARISON" in result
        assert "Benchmark: S&P 500" in result
        
        # Check metrics values
        assert "+1.50%" in result  # Alpha
        assert "0.85" in result    # Beta
        assert "12.50%" in result  # Volatility
        assert "0.75" in result    # Sharpe ratio
        assert "-15.20%" in result # Max drawdown
        
        # Check interpretations
        assert "Outperforming" in result  # Alpha > 0
        assert "Similar volatility" in result  # Beta = 0.85
        assert "Good risk-adj return" in result  # Sharpe > 0.5
    
    def test_format_risk_metrics_custom_benchmark(self):
        """Test risk metrics formatting with custom benchmark."""
        result = self.generator.format_risk_metrics(
            self.risk_metrics, 
            benchmark_name="NASDAQ 100"
        )
        
        assert "Benchmark: NASDAQ 100" in result
    
    def test_format_risk_metrics_interpretations(self):
        """Test different risk metric interpretations."""
        # Test negative alpha
        negative_alpha_metrics = RiskMetrics(
            alpha=-2.0,
            beta=1.5,
            volatility=25.0,
            sharpe_ratio=-0.2,
            max_drawdown=-30.0
        )
        
        result = self.generator.format_risk_metrics(negative_alpha_metrics)
        
        assert "Underperforming" in result  # Alpha < 0
        assert "More volatile" in result    # Beta > 1.2
        assert "Poor risk-adj return" in result  # Sharpe < 0
    
    def test_generate_explanation_moderate_risk(self):
        """Test explanation generation for moderate risk profile."""
        result = self.generator.generate_explanation(
            self.allocation, self.user_input, self.risk_metrics
        )
        
        # Check header
        assert "PORTFOLIO STRATEGY EXPLANATION" in result
        
        # Check risk profile explanation
        assert "balanced allocation" in result
        assert "moderate growth" in result
        
        # Check asset allocation rationale
        assert "Asset Allocation Rationale:" in result
        assert "Stocks (S&P 500 + Small Cap)" in result
        assert "50.0%" in result  # 40% + 10%
        assert "Bonds" in result
        assert "30.0%" in result
        
        # Check risk considerations
        assert "Risk Considerations:" in result
        assert "Moderate volatility" in result  # 12.5% volatility
    
    def test_generate_explanation_low_risk(self):
        """Test explanation generation for low risk profile."""
        low_risk_input = UserInputModel(
            investment_amount=100000.0,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Low",
            return_expectation=5.0
        )
        
        result = self.generator.generate_explanation(
            self.allocation, low_risk_input, self.risk_metrics
        )
        
        assert "conservative allocation" in result
        assert "capital preservation" in result
        assert "lower volatility" in result
    
    def test_generate_explanation_high_risk(self):
        """Test explanation generation for high risk profile."""
        high_risk_input = UserInputModel(
            investment_amount=100000.0,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="High",
            return_expectation=12.0
        )
        
        result = self.generator.generate_explanation(
            self.allocation, high_risk_input, self.risk_metrics
        )
        
        assert "aggressive allocation" in result
        assert "higher returns" in result
        assert "higher volatility" in result
    
    def test_generate_explanation_sip_vs_lumpsum(self):
        """Test explanation differences for SIP vs lumpsum."""
        sip_input = UserInputModel(
            investment_amount=100000.0,
            investment_type="sip",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=8.0
        )
        
        sip_result = self.generator.generate_explanation(
            self.allocation, sip_input, self.risk_metrics
        )
        
        lumpsum_result = self.generator.generate_explanation(
            self.allocation, self.user_input, self.risk_metrics
        )
        
        # SIP should mention dollar-cost averaging
        assert "SIP Investment Benefits:" in sip_result
        assert "Dollar-cost averaging" in sip_result
        assert "Regular investments" in sip_result
        
        # Lumpsum should mention timing considerations
        assert "Lumpsum Investment Considerations:" in lumpsum_result
        assert "Immediate market exposure" in lumpsum_result
        assert "market timing" in lumpsum_result
    
    def test_generate_complete_report(self):
        """Test complete report generation."""
        result = self.generator.generate_complete_report(
            self.allocation, self.projections, self.risk_metrics, self.user_input
        )
        
        # Check all sections are present
        assert "PORTFOLIO ALLOCATION RECOMMENDATION" in result
        assert "PORTFOLIO GROWTH PROJECTIONS" in result
        assert "RISK ANALYSIS & BENCHMARK COMPARISON" in result
        assert "PORTFOLIO STRATEGY EXPLANATION" in result
        assert "IMPORTANT DISCLAIMER" in result
        
        # Check disclaimer content
        assert "educational purposes only" in result
        assert "does not constitute" in result and "financial advice" in result
        assert "Past performance does not guarantee future results" in result
        assert "consult with a qualified financial advisor" in result
    
    def test_terminal_width_consistency(self):
        """Test that all outputs respect terminal width."""
        result = self.generator.generate_complete_report(
            self.allocation, self.projections, self.risk_metrics, self.user_input
        )
        
        lines = result.split('\n')
        for line in lines:
            # Allow some flexibility for table formatting
            if '=' in line and len(line.strip()) > 0:
                assert len(line) <= self.generator.width + 5, f"Line too long: {line}"
    
    def test_formatting_with_large_numbers(self):
        """Test formatting with large investment amounts."""
        large_input = UserInputModel(
            investment_amount=10000000.0,  # 10 million
            investment_type="lumpsum",
            tenure_years=20,
            risk_profile="High",
            return_expectation=10.0
        )
        
        large_projections = [
            ProjectionResult(
                year=1,
                portfolio_value=11000000.0,
                annual_return=10.0,
                cumulative_return=10.0
            )
        ]
        
        allocation_result = self.generator.format_portfolio_allocation(
            self.allocation, large_input
        )
        
        projection_result = self.generator.format_year_by_year_projections(
            large_projections, large_input
        )
        
        # Check proper comma formatting for large numbers
        assert "$10,000,000.00" in allocation_result
        assert "$4,000,000.00" in allocation_result  # 40% of 10M
        assert "$11,000,000.00" in projection_result
    
    def test_formatting_with_small_numbers(self):
        """Test formatting with small investment amounts."""
        small_input = UserInputModel(
            investment_amount=1000.0,
            investment_type="sip",
            tenure_years=5,
            risk_profile="Low",
            return_expectation=6.0
        )
        
        result = self.generator.format_portfolio_allocation(
            self.allocation, small_input
        )
        
        # Check proper formatting for small numbers
        assert "$1,000.00" in result
        assert "400.00" in result  # 40% of 1000
        assert "300.00" in result  # 30% of 1000