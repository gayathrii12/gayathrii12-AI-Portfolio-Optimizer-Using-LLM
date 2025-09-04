"""
Demo script for the Portfolio Allocator Agent.

This script demonstrates how to use the Portfolio Allocator Agent to generate
optimized portfolio allocations based on different risk profiles, expected returns,
and historical data.
"""

import sys
import os
from typing import Dict, List

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.portfolio_allocator_agent import (
    PortfolioAllocatorAgent,
    RiskProfile,
    AllocationInput,
    create_portfolio_allocator_agent
)
from models.data_models import AssetReturns


def create_sample_data() -> List[AssetReturns]:
    """Create sample historical data for demonstration."""
    return [
        AssetReturns(
            sp500=0.10, small_cap=0.12, t_bills=0.02, t_bonds=0.04,
            corporate_bonds=0.05, real_estate=0.08, gold=0.06, year=2019
        ),
        AssetReturns(
            sp500=0.18, small_cap=0.20, t_bills=0.01, t_bonds=0.08,
            corporate_bonds=0.07, real_estate=0.05, gold=0.25, year=2020
        ),
        AssetReturns(
            sp500=0.27, small_cap=0.15, t_bills=0.01, t_bonds=-0.02,
            corporate_bonds=0.02, real_estate=0.40, gold=-0.04, year=2021
        ),
        AssetReturns(
            sp500=-0.19, small_cap=-0.21, t_bills=0.02, t_bonds=-0.13,
            corporate_bonds=-0.15, real_estate=-0.25, gold=0.01, year=2022
        ),
        AssetReturns(
            sp500=0.24, small_cap=0.16, t_bills=0.05, t_bonds=0.05,
            corporate_bonds=0.08, real_estate=0.11, gold=0.13, year=2023
        )
    ]


def create_sample_expected_returns() -> Dict[str, float]:
    """Create sample expected returns for demonstration."""
    return {
        "sp500": 0.095,
        "small_cap": 0.11,
        "t_bills": 0.03,
        "t_bonds": 0.045,
        "corporate_bonds": 0.055,
        "real_estate": 0.085,
        "gold": 0.065
    }


def print_allocation_result(result, risk_profile: str):
    """Print formatted allocation results."""
    print(f"\n{'='*60}")
    print(f"PORTFOLIO ALLOCATION RESULTS - {risk_profile.upper()} RISK")
    print(f"{'='*60}")
    
    if not result.success:
        print(f"❌ Allocation failed: {result.error_message}")
        return
    
    print("✅ Allocation successful!")
    print(f"\nStrategy: {result.strategy_used.strategy_name}")
    print(f"Description: {result.strategy_used.description}")
    
    # Print allocation
    allocation = result.allocation
    print(f"\n📊 PORTFOLIO ALLOCATION:")
    print(f"  S&P 500:        {allocation.sp500:6.2f}%")
    print(f"  Small Cap:      {allocation.small_cap:6.2f}%")
    print(f"  Bonds:          {allocation.bonds:6.2f}%")
    print(f"  Real Estate:    {allocation.real_estate:6.2f}%")
    print(f"  Gold:           {allocation.gold:6.2f}%")
    print(f"  {'─'*25}")
    total = allocation.sp500 + allocation.small_cap + allocation.bonds + allocation.real_estate + allocation.gold
    print(f"  Total:          {total:6.2f}%")
    
    # Print optimization metrics
    if result.optimization_metrics:
        metrics = result.optimization_metrics
        print(f"\n📈 OPTIMIZATION METRICS:")
        print(f"  Expected Return:    {metrics.get('expected_return', 0)*100:6.2f}%")
        print(f"  Expected Volatility: {metrics.get('expected_volatility', 0)*100:6.2f}%")
        print(f"  Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):6.2f}")
        print(f"  Diversification:    {metrics.get('diversification_ratio', 0):6.2f}")
    
    # Print constraint validation
    if result.constraint_validation:
        validation = result.constraint_validation
        print(f"\n✅ CONSTRAINT VALIDATION:")
        print(f"  All constraints met: {validation.get('all_constraints_met', False)}")
        print(f"  Total equals 100%:   {validation.get('total_equals_100', False)}")


def demonstrate_risk_profiles():
    """Demonstrate allocations for different risk profiles."""
    print("🚀 Portfolio Allocator Agent Demo")
    print("=" * 50)
    
    # Create agent
    agent = create_portfolio_allocator_agent()
    
    # Create sample data
    historical_data = create_sample_data()
    expected_returns = create_sample_expected_returns()
    
    print(f"\n📊 Using {len(historical_data)} years of historical data")
    print("Expected Returns:")
    for asset, return_val in expected_returns.items():
        print(f"  {asset:15}: {return_val*100:6.2f}%")
    
    # Test each risk profile
    risk_profiles = [RiskProfile.LOW, RiskProfile.MODERATE, RiskProfile.HIGH]
    
    for risk_profile in risk_profiles:
        allocation_input = AllocationInput(
            risk_profile=risk_profile,
            expected_returns=expected_returns,
            historical_data=historical_data,
            optimization_method="strategic"
        )
        
        result = agent.allocate_portfolio(allocation_input)
        print_allocation_result(result, risk_profile.value)


def demonstrate_optimization_methods():
    """Demonstrate different optimization methods."""
    print(f"\n{'='*60}")
    print("OPTIMIZATION METHODS COMPARISON")
    print(f"{'='*60}")
    
    agent = create_portfolio_allocator_agent()
    historical_data = create_sample_data()
    expected_returns = create_sample_expected_returns()
    
    optimization_methods = ["strategic", "mean_variance", "risk_parity"]
    
    for method in optimization_methods:
        print(f"\n🔧 {method.upper()} OPTIMIZATION")
        print("─" * 40)
        
        allocation_input = AllocationInput(
            risk_profile=RiskProfile.MODERATE,
            expected_returns=expected_returns,
            historical_data=historical_data,
            optimization_method=method
        )
        
        result = agent.allocate_portfolio(allocation_input)
        
        if result.success:
            allocation = result.allocation
            print(f"S&P 500: {allocation.sp500:6.2f}%  |  Small Cap: {allocation.small_cap:6.2f}%")
            print(f"Bonds:   {allocation.bonds:6.2f}%  |  Real Estate: {allocation.real_estate:6.2f}%")
            print(f"Gold:    {allocation.gold:6.2f}%")
            
            if result.optimization_metrics:
                metrics = result.optimization_metrics
                print(f"Expected Return: {metrics.get('expected_return', 0)*100:5.2f}%")
                print(f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):5.2f}")
        else:
            print(f"❌ Failed: {result.error_message}")


def demonstrate_user_preferences():
    """Demonstrate allocation with user preferences."""
    print(f"\n{'='*60}")
    print("USER PREFERENCES DEMONSTRATION")
    print(f"{'='*60}")
    
    agent = create_portfolio_allocator_agent()
    historical_data = create_sample_data()
    expected_returns = create_sample_expected_returns()
    
    # Test with user preferences
    user_preferences = {
        "asset_preferences": {
            "sp500": {"min_allocation": 50.0},  # Minimum 50% in S&P 500
            "gold": {"max_allocation": 2.0}     # Maximum 2% in gold
        },
        "esg_focused": True  # ESG-focused investing
    }
    
    print("\n🎯 User Preferences:")
    print("  • Minimum 50% in S&P 500")
    print("  • Maximum 2% in Gold")
    print("  • ESG-focused investing")
    
    allocation_input = AllocationInput(
        risk_profile=RiskProfile.MODERATE,
        expected_returns=expected_returns,
        historical_data=historical_data,
        user_preferences=user_preferences,
        optimization_method="strategic"
    )
    
    result = agent.allocate_portfolio(allocation_input)
    
    if result.success:
        allocation = result.allocation
        print(f"\n📊 CUSTOMIZED ALLOCATION:")
        print(f"  S&P 500:        {allocation.sp500:6.2f}% {'✅' if allocation.sp500 >= 50.0 else '❌'}")
        print(f"  Small Cap:      {allocation.small_cap:6.2f}%")
        print(f"  Bonds:          {allocation.bonds:6.2f}%")
        print(f"  Real Estate:    {allocation.real_estate:6.2f}%")
        print(f"  Gold:           {allocation.gold:6.2f}% {'✅' if allocation.gold <= 2.0 else '❌'}")
    else:
        print(f"❌ Failed: {result.error_message}")


def demonstrate_correlation_analysis():
    """Demonstrate correlation matrix calculation."""
    print(f"\n{'='*60}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*60}")
    
    agent = create_portfolio_allocator_agent()
    historical_data = create_sample_data()
    
    correlation_matrix = agent._calculate_correlation_matrix(historical_data)
    
    print("\n📊 Asset Correlation Matrix:")
    print("(Values range from -1 to +1, where +1 = perfect positive correlation)")
    
    assets = ['sp500', 'small_cap', 't_bills', 't_bonds', 'corporate_bonds', 'real_estate', 'gold']
    
    # Print header
    print(f"\n{'Asset':<12}", end="")
    for asset in assets:
        print(f"{asset[:8]:>8}", end="")
    print()
    
    # Print correlation values
    for asset1 in assets:
        print(f"{asset1:<12}", end="")
        for asset2 in assets:
            corr_value = correlation_matrix.get(asset1, {}).get(asset2, 0.0)
            print(f"{corr_value:8.2f}", end="")
        print()
    
    # Highlight key insights
    print(f"\n🔍 Key Insights:")
    sp500_small_cap_corr = correlation_matrix.get('sp500', {}).get('small_cap', 0.0)
    sp500_bonds_corr = correlation_matrix.get('sp500', {}).get('t_bonds', 0.0)
    gold_sp500_corr = correlation_matrix.get('gold', {}).get('sp500', 0.0)
    
    print(f"  • S&P 500 vs Small Cap correlation: {sp500_small_cap_corr:.2f} (High - similar asset classes)")
    print(f"  • S&P 500 vs Bonds correlation: {sp500_bonds_corr:.2f} (Low - good diversification)")
    print(f"  • Gold vs S&P 500 correlation: {gold_sp500_corr:.2f} (Low - hedge potential)")


def main():
    """Run the complete demonstration."""
    try:
        # Demonstrate basic functionality
        demonstrate_risk_profiles()
        
        # Demonstrate optimization methods
        demonstrate_optimization_methods()
        
        # Demonstrate user preferences
        demonstrate_user_preferences()
        
        # Demonstrate correlation analysis
        demonstrate_correlation_analysis()
        
        print(f"\n{'='*60}")
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("The Portfolio Allocator Agent can:")
        print("  • Map risk profiles to appropriate allocations")
        print("  • Apply multiple optimization methods")
        print("  • Incorporate user preferences and constraints")
        print("  • Calculate correlation matrices for diversification")
        print("  • Validate all allocation constraints")
        print("  • Provide detailed optimization metrics")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()