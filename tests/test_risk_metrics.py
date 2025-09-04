"""
Unit tests for the Risk Metrics Calculation Module.

This module tests all risk metric calculations including Alpha, Beta, volatility,
Sharpe ratio, and maximum drawdown calculations against known financial formulas.
"""

import unittest
import pandas as pd
import numpy as np
from typing import List

from models.data_models import AssetReturns, PortfolioAllocation, RiskMetrics
from utils.risk_metrics import RiskMetricsCalculator, calculate_portfolio_risk_metrics, validate_risk_metrics


class TestRiskMetricsCalculator(unittest.TestCase):
    """Test cases for RiskMetricsCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = RiskMetricsCalculator(risk_free_rate=0.02)
        
        # Create sample historical data
        self.sample_data = [
            AssetReturns(year=2020, sp500=18.4, small_cap=19.96, t_bills=0.37, 
                        t_bonds=8.0, corporate_bonds=9.89, real_estate=2.12, gold=24.43),
            AssetReturns(year=2021, sp500=28.71, small_cap=14.82, t_bills=0.05, 
                        t_bonds=-2.32, corporate_bonds=-1.04, real_estate=41.34, gold=-3.64),
            AssetReturns(year=2022, sp500=-18.11, small_cap=-20.44, t_bills=1.46, 
                        t_bonds=-12.99, corporate_bonds=-15.76, real_estate=-25.09, gold=-0.01),
            AssetReturns(year=2023, sp500=26.29, small_cap=16.93, t_bills=4.65, 
                        t_bonds=5.53, corporate_bonds=8.52, real_estate=13.59, gold=13.09),
            AssetReturns(year=2024, sp500=12.5, small_cap=8.2, t_bills=3.2, 
                        t_bonds=2.1, corporate_bonds=4.5, real_estate=6.8, gold=5.2)
        ]
        
        # Sample portfolio allocation
        self.sample_allocation = PortfolioAllocation(
            sp500=50.0,
            small_cap=20.0,
            bonds=20.0,
            gold=5.0,
            real_estate=5.0
        )
    
    def test_calculate_portfolio_returns(self):
        """Test portfolio returns calculation."""
        portfolio_returns = self.calculator.calculate_portfolio_returns(
            self.sample_data, self.sample_allocation
        )
        
        # Check that we get the right number of returns
        self.assertEqual(len(portfolio_returns), 5)
        
        # Check that returns are in decimal format (not percentage)
        self.assertTrue(all(abs(ret) < 1.0 for ret in portfolio_returns))
        
        # Manually calculate expected return for first year
        expected_2020 = (
            18.4 * 0.5 +  # SP500: 50%
            19.96 * 0.2 +  # Small cap: 20%
            (8.0 + 9.89) / 2 * 0.2 +  # Bonds: 20%
            24.43 * 0.05 +  # Gold: 5%
            2.12 * 0.05  # Real estate: 5%
        ) / 100
        
        self.assertAlmostEqual(portfolio_returns.iloc[0], expected_2020, places=4)
    
    def test_calculate_alpha(self):
        """Test Alpha calculation."""
        # Create simple test data
        portfolio_returns = pd.Series([0.10, 0.15, -0.05, 0.20, 0.08])
        benchmark_returns = pd.Series([0.08, 0.12, -0.03, 0.18, 0.06])
        
        alpha = self.calculator.calculate_alpha(portfolio_returns, benchmark_returns)
        
        # Alpha should be a reasonable value
        self.assertIsInstance(alpha, float)
        self.assertTrue(-50 <= alpha <= 50)  # Within reasonable range
        
        # Test with identical returns (should give Alpha close to 0)
        identical_returns = pd.Series([0.10, 0.15, -0.05, 0.20, 0.08])
        alpha_identical = self.calculator.calculate_alpha(identical_returns, identical_returns)
        self.assertAlmostEqual(alpha_identical, 0.0, places=2)
    
    def test_calculate_beta(self):
        """Test Beta calculation."""
        # Create test data with known relationship
        benchmark_returns = pd.Series([0.10, 0.15, -0.05, 0.20, 0.08])
        # Portfolio with 1.5x volatility should have Beta ≈ 1.5
        portfolio_returns = benchmark_returns * 1.5
        
        beta = self.calculator.calculate_beta(portfolio_returns, benchmark_returns)
        
        # Beta should be close to 1.5
        self.assertAlmostEqual(beta, 1.5, places=1)
        
        # Test with identical returns (should give Beta = 1.0)
        beta_identical = self.calculator.calculate_beta(benchmark_returns, benchmark_returns)
        self.assertAlmostEqual(beta_identical, 1.0, places=2)
        
        # Test with zero variance benchmark
        zero_variance = pd.Series([0.05, 0.05, 0.05, 0.05, 0.05])
        beta_zero = self.calculator.calculate_beta(portfolio_returns, zero_variance)
        self.assertEqual(beta_zero, 1.0)  # Should default to 1.0
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        # Test with known data
        returns = pd.Series([0.10, 0.15, -0.05, 0.20, 0.08])
        
        volatility = self.calculator.calculate_volatility(returns)
        
        # Calculate expected volatility manually
        expected_volatility = returns.std(ddof=1) * 100
        
        self.assertAlmostEqual(volatility, expected_volatility, places=4)
        
        # Test with constant returns (should give volatility = 0)
        constant_returns = pd.Series([0.05, 0.05, 0.05, 0.05, 0.05])
        volatility_zero = self.calculator.calculate_volatility(constant_returns)
        self.assertEqual(volatility_zero, 0.0)
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Test with known data
        returns = pd.Series([0.10, 0.15, -0.05, 0.20, 0.08])
        
        sharpe_ratio = self.calculator.calculate_sharpe_ratio(returns)
        
        # Calculate expected Sharpe ratio manually
        avg_return = returns.mean()
        volatility = returns.std(ddof=1)
        expected_sharpe = (avg_return - self.calculator.risk_free_rate) / volatility
        
        self.assertAlmostEqual(sharpe_ratio, expected_sharpe, places=4)
        
        # Test with zero volatility
        constant_returns = pd.Series([0.05, 0.05, 0.05, 0.05, 0.05])
        sharpe_zero = self.calculator.calculate_sharpe_ratio(constant_returns)
        self.assertEqual(sharpe_zero, 0.0)
    
    def test_calculate_maximum_drawdown(self):
        """Test maximum drawdown calculation."""
        # Create test data with known drawdown
        # Returns that create a 20% drawdown
        returns = pd.Series([0.10, 0.20, -0.15, -0.10, 0.05])
        
        max_drawdown = self.calculator.calculate_maximum_drawdown(returns)
        
        # Should be negative
        self.assertLess(max_drawdown, 0)
        
        # Test with only positive returns (should have minimal drawdown)
        positive_returns = pd.Series([0.05, 0.10, 0.08, 0.12, 0.06])
        max_drawdown_positive = self.calculator.calculate_maximum_drawdown(positive_returns)
        self.assertLessEqual(max_drawdown_positive, 0)
        
        # Test with severe crash
        crash_returns = pd.Series([0.20, 0.15, -0.40, -0.20, 0.10])
        max_drawdown_crash = self.calculator.calculate_maximum_drawdown(crash_returns)
        self.assertLess(max_drawdown_crash, -30)  # Should be significant drawdown
    
    def test_calculate_all_metrics(self):
        """Test calculation of all metrics together."""
        risk_metrics = self.calculator.calculate_all_metrics(
            self.sample_data, self.sample_allocation
        )
        
        # Check that we get a RiskMetrics object
        self.assertIsInstance(risk_metrics, RiskMetrics)
        
        # Check that all metrics are calculated
        self.assertIsInstance(risk_metrics.alpha, float)
        self.assertIsInstance(risk_metrics.beta, float)
        self.assertIsInstance(risk_metrics.volatility, float)
        self.assertIsInstance(risk_metrics.sharpe_ratio, float)
        self.assertIsInstance(risk_metrics.max_drawdown, float)
        
        # Check reasonable ranges
        self.assertTrue(-50 <= risk_metrics.alpha <= 50)
        self.assertTrue(0 <= risk_metrics.beta <= 5)
        self.assertTrue(0 <= risk_metrics.volatility <= 100)
        self.assertTrue(-10 <= risk_metrics.sharpe_ratio <= 10)
        self.assertTrue(-100 <= risk_metrics.max_drawdown <= 0)
    
    def test_validate_metrics(self):
        """Test risk metrics validation."""
        # Test with normal metrics
        normal_metrics = RiskMetrics(
            alpha=2.5,
            beta=1.2,
            volatility=15.0,
            sharpe_ratio=0.8,
            max_drawdown=-12.5
        )
        
        is_valid, warnings = self.calculator.validate_metrics(normal_metrics)
        self.assertTrue(is_valid)
        self.assertEqual(len(warnings), 0)
        
        # Test with extreme metrics
        extreme_metrics = RiskMetrics(
            alpha=25.0,  # Too high
            beta=4.0,    # Very high
            volatility=60.0,  # Very high
            sharpe_ratio=5.0,  # Unusually high
            max_drawdown=-90.0  # Extremely high
        )
        
        is_valid, warnings = self.calculator.validate_metrics(extreme_metrics)
        self.assertTrue(is_valid)  # Still valid, just warnings
        self.assertGreater(len(warnings), 0)
        
        # Test with invalid metrics (this should raise ValidationError due to Pydantic validation)
        with self.assertRaises(Exception):  # Pydantic will raise ValidationError
            invalid_metrics = RiskMetrics(
                alpha=0.0,
                beta=1.0,
                volatility=10.0,
                sharpe_ratio=0.5,
                max_drawdown=5.0  # Should be negative - Pydantic will catch this
            )
        
        # Test validation with edge case that passes Pydantic but triggers warnings
        edge_case_metrics = RiskMetrics(
            alpha=0.0,
            beta=1.0,
            volatility=10.0,
            sharpe_ratio=0.5,
            max_drawdown=0.0  # Exactly zero - valid but unusual
        )
        
        is_valid, warnings = self.calculator.validate_metrics(edge_case_metrics)
        self.assertTrue(is_valid)  # Still valid since 0.0 is allowed
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with insufficient data
        minimal_data = [
            AssetReturns(year=2023, sp500=10.0, small_cap=12.0, t_bills=2.0, 
                        t_bonds=5.0, corporate_bonds=6.0, real_estate=8.0, gold=7.0)
        ]
        
        risk_metrics = self.calculator.calculate_all_metrics(
            minimal_data, self.sample_allocation
        )
        
        # Should return default values without crashing
        self.assertIsInstance(risk_metrics, RiskMetrics)
        
        # Test with empty data
        empty_returns = pd.Series([])
        alpha = self.calculator.calculate_alpha(empty_returns, empty_returns)
        self.assertEqual(alpha, 0.0)
        
        beta = self.calculator.calculate_beta(empty_returns, empty_returns)
        self.assertEqual(beta, 1.0)
        
        volatility = self.calculator.calculate_volatility(empty_returns)
        self.assertEqual(volatility, 0.0)
        
        sharpe = self.calculator.calculate_sharpe_ratio(empty_returns)
        self.assertEqual(sharpe, 0.0)
        
        drawdown = self.calculator.calculate_maximum_drawdown(empty_returns)
        self.assertEqual(drawdown, 0.0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = [
            AssetReturns(year=2020, sp500=18.4, small_cap=19.96, t_bills=0.37, 
                        t_bonds=8.0, corporate_bonds=9.89, real_estate=2.12, gold=24.43),
            AssetReturns(year=2021, sp500=28.71, small_cap=14.82, t_bills=0.05, 
                        t_bonds=-2.32, corporate_bonds=-1.04, real_estate=41.34, gold=-3.64),
            AssetReturns(year=2022, sp500=-18.11, small_cap=-20.44, t_bills=1.46, 
                        t_bonds=-12.99, corporate_bonds=-15.76, real_estate=-25.09, gold=-0.01)
        ]
        
        self.sample_allocation = PortfolioAllocation(
            sp500=60.0,
            small_cap=15.0,
            bonds=15.0,
            gold=5.0,
            real_estate=5.0
        )
    
    def test_calculate_portfolio_risk_metrics(self):
        """Test the convenience function for calculating all metrics."""
        risk_metrics = calculate_portfolio_risk_metrics(
            self.sample_data, 
            self.sample_allocation,
            risk_free_rate=0.025
        )
        
        self.assertIsInstance(risk_metrics, RiskMetrics)
        
        # Test with different benchmark
        risk_metrics_gold = calculate_portfolio_risk_metrics(
            self.sample_data, 
            self.sample_allocation,
            benchmark_asset='gold'
        )
        
        self.assertIsInstance(risk_metrics_gold, RiskMetrics)
        # Beta should be different when using different benchmark
        self.assertNotEqual(risk_metrics.beta, risk_metrics_gold.beta)
    
    def test_validate_risk_metrics_function(self):
        """Test the convenience function for validating metrics."""
        test_metrics = RiskMetrics(
            alpha=1.5,
            beta=1.1,
            volatility=12.0,
            sharpe_ratio=0.6,
            max_drawdown=-8.5
        )
        
        is_valid, warnings = validate_risk_metrics(test_metrics)
        
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(warnings, list)


class TestFinancialFormulas(unittest.TestCase):
    """Test cases to validate against known financial formulas."""
    
    def test_beta_formula_validation(self):
        """Validate Beta calculation against known formula."""
        # Create test data with known correlation
        np.random.seed(42)  # For reproducible results
        
        # Generate correlated returns
        benchmark = np.random.normal(0.08, 0.15, 100)
        portfolio = 0.5 * benchmark + np.random.normal(0.02, 0.10, 100)
        
        calculator = RiskMetricsCalculator()
        beta = calculator.calculate_beta(pd.Series(portfolio), pd.Series(benchmark))
        
        # Calculate Beta manually using formula
        covariance = np.cov(portfolio, benchmark)[0, 1]
        benchmark_variance = np.var(benchmark, ddof=1)
        expected_beta = covariance / benchmark_variance
        
        self.assertAlmostEqual(beta, expected_beta, places=3)
    
    def test_sharpe_ratio_formula_validation(self):
        """Validate Sharpe ratio calculation against known formula."""
        returns = pd.Series([0.12, 0.08, 0.15, -0.03, 0.20, 0.05, 0.18])
        risk_free_rate = 0.03
        
        calculator = RiskMetricsCalculator(risk_free_rate=risk_free_rate)
        sharpe = calculator.calculate_sharpe_ratio(returns)
        
        # Calculate manually
        excess_return = returns.mean() - risk_free_rate
        volatility = returns.std(ddof=1)
        expected_sharpe = excess_return / volatility
        
        self.assertAlmostEqual(sharpe, expected_sharpe, places=4)
    
    def test_maximum_drawdown_formula_validation(self):
        """Validate maximum drawdown calculation."""
        # Create returns with known drawdown pattern
        returns = pd.Series([0.10, 0.05, -0.15, -0.10, -0.05, 0.20, 0.15])
        
        calculator = RiskMetricsCalculator()
        max_drawdown = calculator.calculate_maximum_drawdown(returns)
        
        # Calculate manually
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        expected_max_drawdown = drawdowns.min() * 100
        
        self.assertAlmostEqual(max_drawdown, expected_max_drawdown, places=4)
    
    def test_alpha_formula_validation(self):
        """Validate Alpha calculation against CAPM formula."""
        portfolio_returns = pd.Series([0.12, 0.08, 0.15, -0.03, 0.20])
        benchmark_returns = pd.Series([0.10, 0.06, 0.12, -0.05, 0.18])
        risk_free_rate = 0.02
        
        calculator = RiskMetricsCalculator(risk_free_rate=risk_free_rate)
        alpha = calculator.calculate_alpha(portfolio_returns, benchmark_returns)
        
        # Calculate manually using CAPM
        beta = calculator.calculate_beta(portfolio_returns, benchmark_returns)
        portfolio_return = portfolio_returns.mean()
        benchmark_return = benchmark_returns.mean()
        
        expected_alpha = (portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))) * 100
        
        self.assertAlmostEqual(alpha, expected_alpha, places=3)


if __name__ == '__main__':
    unittest.main()