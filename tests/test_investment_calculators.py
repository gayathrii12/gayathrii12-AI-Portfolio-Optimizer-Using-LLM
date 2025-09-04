"""
Unit tests for Investment Calculators module.

Tests cover all investment calculation scenarios:
- Lump sum investments with compound growth
- SIP (Systematic Investment Plan) calculations
- SWP (Systematic Withdrawal Plan) calculations
- Edge cases and error handling
"""

import pytest
import math
from models.investment_calculators import InvestmentCalculators, InvestmentProjection


class TestInvestmentCalculators:
    """Test suite for InvestmentCalculators class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = InvestmentCalculators()
        self.sample_returns = {
            'sp500': 10.0,
            'bonds': 5.0,
            'gold': 3.0
        }
    
    def test_lump_sum_basic_calculation(self):
        """Test basic lump sum calculation with compound growth."""
        amount = 100000
        returns = {'sp500': 10.0}  # 10% annual return
        years = 5
        
        projections = self.calculator.calculate_lump_sum(amount, returns, years)
        
        assert len(projections) == years
        assert projections[0].portfolio_value == pytest.approx(110000, rel=1e-2)  # Year 1: 100k * 1.1
        assert projections[4].portfolio_value == pytest.approx(161051, rel=1e-2)  # Year 5: 100k * 1.1^5
        
        # Verify projection structure
        for i, projection in enumerate(projections):
            assert projection.year == i + 1
            assert projection.annual_contribution == (amount if i == 0 else 0)
            assert projection.annual_withdrawal == 0
            assert projection.cumulative_contributions == amount
            assert projection.cumulative_withdrawals == 0
    
    def test_lump_sum_weighted_returns(self):
        """Test lump sum calculation with multiple asset returns."""
        amount = 50000
        returns = {
            'sp500': 12.0,  # 12% return
            'bonds': 4.0,   # 4% return
            'gold': 2.0     # 2% return
        }
        years = 3
        
        # Expected average return: (12 + 4 + 2) / 3 = 6%
        expected_average_return = (12.0 + 4.0 + 2.0) / 3
        
        projections = self.calculator.calculate_lump_sum(amount, returns, years)
        
        assert len(projections) == years
        # Year 1: 50000 * 1.06 = 53000
        assert projections[0].portfolio_value == pytest.approx(53000, rel=1e-2)
        assert projections[0].annual_return == pytest.approx(expected_average_return, rel=1e-2)
    
    def test_sip_basic_calculation(self):
        """Test basic SIP calculation with monthly investments."""
        monthly_amount = 5000
        returns = {'sp500': 12.0}  # 12% annual return, 1% monthly
        years = 2
        
        projections = self.calculator.calculate_sip(monthly_amount, returns, years)
        
        assert len(projections) == years
        
        # Verify annual contributions
        assert projections[0].annual_contribution == monthly_amount * 12
        assert projections[1].annual_contribution == monthly_amount * 12
        
        # Verify cumulative contributions
        assert projections[0].cumulative_contributions == monthly_amount * 12
        assert projections[1].cumulative_contributions == monthly_amount * 24
        
        # Portfolio should grow with monthly compounding
        assert projections[0].portfolio_value > monthly_amount * 12  # Growth in year 1
        assert projections[1].portfolio_value > monthly_amount * 24  # Growth in year 2
    
    def test_sip_compound_growth(self):
        """Test SIP with compound growth over multiple years."""
        monthly_amount = 1000
        returns = {'bonds': 6.0}  # 6% annual return
        years = 5
        
        projections = self.calculator.calculate_sip(monthly_amount, returns, years)
        
        # Verify growth pattern
        values = [p.portfolio_value for p in projections]
        for i in range(1, len(values)):
            # Each year should have higher value than previous
            assert values[i] > values[i-1]
        
        # Final value should be significantly more than total contributions due to compounding
        total_contributions = monthly_amount * 12 * years
        final_value = projections[-1].portfolio_value
        assert final_value > total_contributions
        
        # Verify cumulative contributions
        assert projections[-1].cumulative_contributions == total_contributions
    
    def test_swp_basic_calculation(self):
        """Test basic SWP calculation with monthly withdrawals."""
        initial_amount = 500000
        monthly_withdrawal = 2000
        returns = {'bonds': 8.0}  # 8% annual return
        years = 3
        
        projections = self.calculator.calculate_swp(initial_amount, monthly_withdrawal, returns, years)
        
        assert len(projections) == years
        
        # Verify annual withdrawals
        expected_annual_withdrawal = monthly_withdrawal * 12
        assert projections[0].annual_withdrawal == expected_annual_withdrawal
        
        # Verify cumulative withdrawals
        assert projections[0].cumulative_withdrawals == expected_annual_withdrawal
        assert projections[2].cumulative_withdrawals == expected_annual_withdrawal * 3
        
        # Portfolio value should be different from initial amount
        # With 8% return and 4.8% withdrawal rate, it should actually grow
        assert projections[0].portfolio_value != initial_amount
        
        # But should still have positive value if returns > withdrawals
        assert projections[-1].portfolio_value > 0
    
    def test_swp_portfolio_exhaustion(self):
        """Test SWP when withdrawals exceed portfolio growth."""
        initial_amount = 100000
        monthly_withdrawal = 5000  # High withdrawal rate
        returns = {'bonds': 3.0}   # Low return
        years = 5
        
        projections = self.calculator.calculate_swp(initial_amount, monthly_withdrawal, returns, years)
        
        # Portfolio should be exhausted before 5 years
        assert len(projections) <= years
        
        # Final projection should have zero or very low value
        final_projection = projections[-1]
        assert final_projection.portfolio_value <= 1000  # Allow for small remaining amount
    
    def test_swp_sustainable_withdrawal(self):
        """Test SWP with sustainable withdrawal rate."""
        initial_amount = 1000000
        monthly_withdrawal = 3000  # 3.6% annual withdrawal rate
        returns = {'sp500': 8.0}   # 8% annual return
        years = 10
        
        projections = self.calculator.calculate_swp(initial_amount, monthly_withdrawal, returns, years)
        
        assert len(projections) == years
        
        # Portfolio should maintain positive value throughout
        for projection in projections:
            assert projection.portfolio_value > 0
        
        # Final value should still be substantial
        assert projections[-1].portfolio_value > 500000
    
    def test_investment_summary_lump_sum(self):
        """Test investment summary generation for lump sum."""
        amount = 100000
        returns = {'sp500': 10.0}
        years = 5
        
        projections = self.calculator.calculate_lump_sum(amount, returns, years)
        summary = self.calculator.generate_investment_summary(projections)
        
        assert summary['initial_investment'] == amount
        assert summary['total_contributions'] == amount
        assert summary['total_withdrawals'] == 0
        assert summary['investment_years'] == years
        assert summary['final_value'] > amount  # Should have grown
        assert summary['total_return'] > 0
        assert summary['total_return_percentage'] > 0
        assert summary['cagr'] > 0
    
    def test_investment_summary_sip(self):
        """Test investment summary generation for SIP."""
        monthly_amount = 5000
        returns = {'sp500': 12.0}
        years = 3
        
        projections = self.calculator.calculate_sip(monthly_amount, returns, years)
        summary = self.calculator.generate_investment_summary(projections)
        
        expected_contributions = monthly_amount * 12 * years
        assert summary['total_contributions'] == expected_contributions
        assert summary['total_withdrawals'] == 0
        assert summary['investment_years'] == years
        assert summary['final_value'] > expected_contributions  # Should have grown
        assert summary['cagr'] > 0
    
    def test_investment_summary_swp(self):
        """Test investment summary generation for SWP."""
        initial_amount = 500000
        monthly_withdrawal = 2000
        returns = {'bonds': 6.0}
        years = 5
        
        projections = self.calculator.calculate_swp(initial_amount, monthly_withdrawal, returns, years)
        summary = self.calculator.generate_investment_summary(projections)
        
        expected_withdrawals = monthly_withdrawal * 12 * years
        assert summary['total_contributions'] == initial_amount
        assert summary['total_withdrawals'] == expected_withdrawals
        assert summary['investment_years'] == years
    
    def test_error_handling_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Test negative amounts
        with pytest.raises(ValueError, match="Investment amount must be positive"):
            self.calculator.calculate_lump_sum(-1000, self.sample_returns, 5)
        
        with pytest.raises(ValueError, match="Monthly investment amount must be positive"):
            self.calculator.calculate_sip(-500, self.sample_returns, 3)
        
        with pytest.raises(ValueError, match="Initial investment amount must be positive"):
            self.calculator.calculate_swp(-100000, 1000, self.sample_returns, 5)
        
        # Test zero/negative years
        with pytest.raises(ValueError, match="Investment years must be positive"):
            self.calculator.calculate_lump_sum(10000, self.sample_returns, 0)
        
        # Test empty returns
        with pytest.raises(ValueError, match="Returns dictionary cannot be empty"):
            self.calculator.calculate_lump_sum(10000, {}, 5)
        
        # Test extremely negative returns
        with pytest.raises(ValueError, match="Returns cannot be less than -100%"):
            self.calculator.calculate_lump_sum(10000, {'sp500': -150}, 5)
    
    def test_zero_returns_scenario(self):
        """Test calculations with zero returns."""
        amount = 50000
        returns = {'cash': 0.0}  # Zero return
        years = 3
        
        projections = self.calculator.calculate_lump_sum(amount, returns, years)
        
        # With zero returns, value should remain constant
        for projection in projections:
            assert projection.portfolio_value == amount
            assert projection.annual_return == 0.0
    
    def test_high_returns_scenario(self):
        """Test calculations with high returns."""
        amount = 10000
        returns = {'growth_stock': 25.0}  # 25% annual return
        years = 4
        
        projections = self.calculator.calculate_lump_sum(amount, returns, years)
        
        # Verify exponential growth
        expected_final = amount * (1.25 ** years)
        assert projections[-1].portfolio_value == pytest.approx(expected_final, rel=1e-2)
    
    def test_mixed_allocation_returns(self):
        """Test calculations with mixed positive and negative returns."""
        amount = 100000
        returns = {
            'stocks': 15.0,     # 15% return
            'bonds': 5.0,       # 5% return  
            'commodities': -2.0 # -2% return (loss)
        }
        years = 2
        
        # Expected average return: (15 + 5 + (-2)) / 3 = 6%
        expected_average_return = (15.0 + 5.0 + (-2.0)) / 3
        
        projections = self.calculator.calculate_lump_sum(amount, returns, years)
        
        # Should still grow despite negative component
        assert projections[-1].portfolio_value > amount
        assert projections[0].annual_return == pytest.approx(expected_average_return, rel=1e-2)
    
    def test_projection_data_integrity(self):
        """Test that projection data maintains integrity across calculations."""
        monthly_amount = 2000
        returns = {'balanced': 8.0}
        years = 4
        
        projections = self.calculator.calculate_sip(monthly_amount, returns, years)
        
        # Verify year sequence
        for i, projection in enumerate(projections):
            assert projection.year == i + 1
        
        # Verify cumulative values are monotonic
        cumulative_contributions = [p.cumulative_contributions for p in projections]
        assert all(cumulative_contributions[i] <= cumulative_contributions[i+1] 
                  for i in range(len(cumulative_contributions)-1))
        
        # Verify portfolio values are positive
        for projection in projections:
            assert projection.portfolio_value >= 0
    
    def test_edge_case_single_year(self):
        """Test calculations for single year investment."""
        amount = 25000
        returns = {'sp500': 12.0}
        years = 1
        
        projections = self.calculator.calculate_lump_sum(amount, returns, years)
        
        assert len(projections) == 1
        assert projections[0].year == 1
        assert projections[0].portfolio_value == pytest.approx(amount * 1.12, rel=1e-2)
        assert projections[0].cumulative_contributions == amount
    
    def test_large_numbers_handling(self):
        """Test calculations with large investment amounts."""
        amount = 10_000_000  # 10 million
        returns = {'diversified': 7.0}
        years = 10
        
        projections = self.calculator.calculate_lump_sum(amount, returns, years)
        
        # Should handle large numbers without overflow
        assert projections[-1].portfolio_value > amount
        assert projections[-1].portfolio_value < 1_000_000_000_000  # Less than 1 trillion
    
    def test_precision_handling(self):
        """Test that calculations maintain reasonable precision."""
        amount = 12345.67
        returns = {'precise': 7.89}
        years = 3
        
        projections = self.calculator.calculate_lump_sum(amount, returns, years)
        
        # Values should be rounded to 2 decimal places
        for projection in projections:
            assert projection.portfolio_value == round(projection.portfolio_value, 2)