"""
Risk Metrics Calculation Demo

This script demonstrates how to use the risk metrics calculation module
to analyze portfolio risk characteristics including Alpha, Beta, volatility,
Sharpe ratio, and maximum drawdown.
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_models import AssetReturns, PortfolioAllocation, RiskMetrics
from utils.risk_metrics import RiskMetricsCalculator, calculate_portfolio_risk_metrics, validate_risk_metrics


def create_sample_data():
    """Create sample historical data for demonstration."""
    return [
        AssetReturns(year=2019, sp500=31.49, small_cap=25.52, t_bills=2.27, 
                    t_bonds=6.86, corporate_bonds=13.79, real_estate=22.99, gold=18.31),
        AssetReturns(year=2020, sp500=18.40, small_cap=19.96, t_bills=0.37, 
                    t_bonds=8.00, corporate_bonds=9.89, real_estate=2.12, gold=24.43),
        AssetReturns(year=2021, sp500=28.71, small_cap=14.82, t_bills=0.05, 
                    t_bonds=-2.32, corporate_bonds=-1.04, real_estate=41.34, gold=-3.64),
        AssetReturns(year=2022, sp500=-18.11, small_cap=-20.44, t_bills=1.46, 
                    t_bonds=-12.99, corporate_bonds=-15.76, real_estate=-25.09, gold=-0.01),
        AssetReturns(year=2023, sp500=26.29, small_cap=16.93, t_bills=4.65, 
                    t_bonds=5.53, corporate_bonds=8.52, real_estate=13.59, gold=13.09),
        AssetReturns(year=2024, sp500=12.50, small_cap=8.20, t_bills=3.20, 
                    t_bonds=2.10, corporate_bonds=4.50, real_estate=6.80, gold=5.20)
    ]


def demo_conservative_portfolio():
    """Demonstrate risk metrics for a conservative portfolio."""
    print("=" * 60)
    print("CONSERVATIVE PORTFOLIO RISK ANALYSIS")
    print("=" * 60)
    
    # Conservative allocation
    conservative_allocation = PortfolioAllocation(
        sp500=30.0,
        small_cap=10.0,
        bonds=50.0,
        gold=5.0,
        real_estate=5.0
    )
    
    print(f"Portfolio Allocation:")
    print(f"  S&P 500: {conservative_allocation.sp500}%")
    print(f"  Small Cap: {conservative_allocation.small_cap}%")
    print(f"  Bonds: {conservative_allocation.bonds}%")
    print(f"  Gold: {conservative_allocation.gold}%")
    print(f"  Real Estate: {conservative_allocation.real_estate}%")
    print()
    
    # Calculate risk metrics
    historical_data = create_sample_data()
    risk_metrics = calculate_portfolio_risk_metrics(
        historical_data, 
        conservative_allocation,
        risk_free_rate=0.025
    )
    
    print("Risk Metrics:")
    print(f"  Alpha: {risk_metrics.alpha:.2f}%")
    print(f"  Beta: {risk_metrics.beta:.2f}")
    print(f"  Volatility: {risk_metrics.volatility:.2f}%")
    print(f"  Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
    print(f"  Maximum Drawdown: {risk_metrics.max_drawdown:.2f}%")
    print()
    
    # Validate metrics
    is_valid, warnings = validate_risk_metrics(risk_metrics)
    print(f"Metrics Valid: {is_valid}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("No validation warnings")
    
    return risk_metrics


def demo_aggressive_portfolio():
    """Demonstrate risk metrics for an aggressive portfolio."""
    print("=" * 60)
    print("AGGRESSIVE PORTFOLIO RISK ANALYSIS")
    print("=" * 60)
    
    # Aggressive allocation
    aggressive_allocation = PortfolioAllocation(
        sp500=60.0,
        small_cap=25.0,
        bonds=5.0,
        gold=5.0,
        real_estate=5.0
    )
    
    print(f"Portfolio Allocation:")
    print(f"  S&P 500: {aggressive_allocation.sp500}%")
    print(f"  Small Cap: {aggressive_allocation.small_cap}%")
    print(f"  Bonds: {aggressive_allocation.bonds}%")
    print(f"  Gold: {aggressive_allocation.gold}%")
    print(f"  Real Estate: {aggressive_allocation.real_estate}%")
    print()
    
    # Calculate risk metrics
    historical_data = create_sample_data()
    risk_metrics = calculate_portfolio_risk_metrics(
        historical_data, 
        aggressive_allocation,
        risk_free_rate=0.025
    )
    
    print("Risk Metrics:")
    print(f"  Alpha: {risk_metrics.alpha:.2f}%")
    print(f"  Beta: {risk_metrics.beta:.2f}")
    print(f"  Volatility: {risk_metrics.volatility:.2f}%")
    print(f"  Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
    print(f"  Maximum Drawdown: {risk_metrics.max_drawdown:.2f}%")
    print()
    
    # Validate metrics
    is_valid, warnings = validate_risk_metrics(risk_metrics)
    print(f"Metrics Valid: {is_valid}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("No validation warnings")
    
    return risk_metrics


def demo_individual_calculations():
    """Demonstrate individual risk metric calculations."""
    print("=" * 60)
    print("INDIVIDUAL RISK METRIC CALCULATIONS")
    print("=" * 60)
    
    # Create calculator
    calculator = RiskMetricsCalculator(risk_free_rate=0.025)
    
    # Sample portfolio allocation
    allocation = PortfolioAllocation(
        sp500=50.0,
        small_cap=20.0,
        bonds=20.0,
        gold=5.0,
        real_estate=5.0
    )
    
    historical_data = create_sample_data()
    
    # Calculate portfolio returns
    portfolio_returns = calculator.calculate_portfolio_returns(historical_data, allocation)
    print(f"Portfolio Returns by Year:")
    for i, year_data in enumerate(historical_data):
        print(f"  {year_data.year}: {portfolio_returns.iloc[i]:.4f} ({portfolio_returns.iloc[i]*100:.2f}%)")
    print()
    
    # Get benchmark returns (S&P 500)
    df = calculator._convert_to_dataframe(historical_data)
    benchmark_returns = df['sp500'] / 100
    
    # Calculate individual metrics
    print("Individual Metric Calculations:")
    
    alpha = calculator.calculate_alpha(portfolio_returns, benchmark_returns)
    print(f"  Alpha: {alpha:.4f}%")
    
    beta = calculator.calculate_beta(portfolio_returns, benchmark_returns)
    print(f"  Beta: {beta:.4f}")
    
    volatility = calculator.calculate_volatility(portfolio_returns)
    print(f"  Volatility: {volatility:.4f}%")
    
    sharpe_ratio = calculator.calculate_sharpe_ratio(portfolio_returns)
    print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
    
    max_drawdown = calculator.calculate_maximum_drawdown(portfolio_returns)
    print(f"  Maximum Drawdown: {max_drawdown:.4f}%")
    print()


def compare_portfolios():
    """Compare risk metrics between different portfolios."""
    print("=" * 60)
    print("PORTFOLIO COMPARISON")
    print("=" * 60)
    
    historical_data = create_sample_data()
    
    # Define different portfolios
    portfolios = {
        "Conservative": PortfolioAllocation(sp500=20.0, small_cap=5.0, bonds=65.0, gold=5.0, real_estate=5.0),
        "Moderate": PortfolioAllocation(sp500=40.0, small_cap=15.0, bonds=35.0, gold=5.0, real_estate=5.0),
        "Aggressive": PortfolioAllocation(sp500=60.0, small_cap=25.0, bonds=5.0, gold=5.0, real_estate=5.0),
        "Growth": PortfolioAllocation(sp500=70.0, small_cap=20.0, bonds=0.0, gold=5.0, real_estate=5.0)
    }
    
    print(f"{'Portfolio':<12} {'Alpha':<8} {'Beta':<6} {'Vol':<8} {'Sharpe':<8} {'MaxDD':<8}")
    print("-" * 60)
    
    for name, allocation in portfolios.items():
        risk_metrics = calculate_portfolio_risk_metrics(historical_data, allocation)
        print(f"{name:<12} {risk_metrics.alpha:>6.2f}% {risk_metrics.beta:>5.2f} "
              f"{risk_metrics.volatility:>6.2f}% {risk_metrics.sharpe_ratio:>6.2f} "
              f"{risk_metrics.max_drawdown:>6.2f}%")
    
    print()


def main():
    """Run all risk metrics demonstrations."""
    print("Risk Metrics Calculation Module Demo")
    print("====================================")
    print()
    
    try:
        # Demo conservative portfolio
        conservative_metrics = demo_conservative_portfolio()
        print()
        
        # Demo aggressive portfolio
        aggressive_metrics = demo_aggressive_portfolio()
        print()
        
        # Demo individual calculations
        demo_individual_calculations()
        
        # Compare portfolios
        compare_portfolios()
        
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("The risk metrics module successfully calculated:")
        print("✓ Alpha - excess return relative to benchmark")
        print("✓ Beta - volatility correlation with benchmark")
        print("✓ Volatility - standard deviation of returns")
        print("✓ Sharpe Ratio - risk-adjusted return metric")
        print("✓ Maximum Drawdown - largest peak-to-trough decline")
        print()
        print("All calculations follow standard financial formulas and")
        print("include comprehensive validation and error handling.")
        
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()