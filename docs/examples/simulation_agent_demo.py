"""
Demo script for the Simulation Agent.

This script demonstrates the capabilities of the Simulation Agent including
lumpsum projections, SIP calculations, withdrawal processing, and various
investment scenarios.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.simulation_agent import SimulationAgent, SimulationInput
from models.data_models import UserInputModel, PortfolioAllocation
import json


def print_separator(title: str):
    """Print a formatted separator with title."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_simulation_results(result, scenario_name: str):
    """Print formatted simulation results."""
    print(f"\n📊 {scenario_name} Results:")
    print("-" * 40)
    
    if not result.success:
        print(f"❌ Simulation failed: {result.error_message}")
        return
    
    print(f"✅ Simulation Status: Success")
    print(f"💰 Total Invested: ${result.total_invested:,.2f}")
    print(f"🎯 Final Portfolio Value: ${result.final_value:,.2f}")
    print(f"📈 CAGR: {result.cagr:.2f}%")
    print(f"📊 Cumulative Return: {result.cumulative_return:.2f}%")
    print(f"💵 Absolute Gain: ${result.final_value - result.total_invested:,.2f}")
    
    # Print withdrawal impact if present
    if result.withdrawal_impact:
        print(f"\n💸 Withdrawal Impact:")
        print(f"   Total Withdrawals: ${result.withdrawal_impact.get('total_withdrawals', 0):,.2f}")
        print(f"   Strategy: {result.withdrawal_impact.get('withdrawal_strategy', 'N/A')}")
    
    # Print Monte Carlo statistics
    if result.simulation_statistics:
        stats = result.simulation_statistics
        print(f"\n🎲 Monte Carlo Statistics ({stats.get('simulation_runs', 0)} runs):")
        print(f"   Mean Return: {stats.get('mean_return', 0)*100:.2f}%")
        print(f"   Volatility: {stats.get('std_deviation', 0)*100:.2f}%")
        print(f"   5th Percentile: {stats.get('percentile_5', 0)*100:.2f}%")
        print(f"   95th Percentile: {stats.get('percentile_95', 0)*100:.2f}%")
        print(f"   Probability of Positive Returns: {stats.get('probability_positive', 0)*100:.1f}%")
    
    # Print first few and last few projections
    if result.projections:
        print(f"\n📅 Year-by-Year Projections (First 5 and Last 5 years):")
        print("Year | Portfolio Value | Annual Return | Cumulative Return")
        print("-" * 55)
        
        # Show first 5 years
        for proj in result.projections[:5]:
            print(f"{proj.year:4d} | ${proj.portfolio_value:13,.2f} | {proj.annual_return:11.2f}% | {proj.cumulative_return:15.2f}%")
        
        # Show ellipsis if more than 10 years
        if len(result.projections) > 10:
            print("  ...")
        
        # Show last 5 years if more than 5 years total
        if len(result.projections) > 5:
            for proj in result.projections[-5:]:
                print(f"{proj.year:4d} | ${proj.portfolio_value:13,.2f} | {proj.annual_return:11.2f}% | {proj.cumulative_return:15.2f}%")


def demo_lumpsum_investment():
    """Demonstrate lumpsum investment simulation."""
    print_separator("LUMPSUM INVESTMENT SIMULATION")
    
    # Create simulation agent
    agent = SimulationAgent()
    
    # Define user input for lumpsum investment
    user_input = UserInputModel(
        investment_amount=100000,  # $100,000 lumpsum
        investment_type="lumpsum",
        tenure_years=15,
        risk_profile="Moderate",
        return_expectation=8.0
    )
    
    # Define balanced portfolio allocation
    portfolio_allocation = PortfolioAllocation(
        sp500=45.0,
        small_cap=10.0,
        bonds=30.0,
        real_estate=10.0,
        gold=5.0
    )
    
    # Define expected returns
    expected_returns = {
        'sp500': 0.095,
        'small_cap': 0.11,
        't_bills': 0.025,
        't_bonds': 0.048,
        'corporate_bonds': 0.057,
        'real_estate': 0.085,
        'gold': 0.065
    }
    
    # Create simulation input
    simulation_input = SimulationInput(
        user_input=user_input,
        portfolio_allocation=portfolio_allocation,
        expected_returns=expected_returns,
        simulation_runs=1000
    )
    
    # Run simulation
    result = agent.simulate_portfolio(simulation_input)
    
    # Print results
    print_simulation_results(result, "Lumpsum Investment ($100K for 15 years)")


def demo_sip_investment():
    """Demonstrate SIP investment simulation."""
    print_separator("SIP INVESTMENT SIMULATION")
    
    # Create simulation agent
    agent = SimulationAgent()
    
    # Define user input for SIP investment
    user_input = UserInputModel(
        investment_amount=5000,  # $5,000 monthly SIP
        investment_type="sip",
        tenure_years=20,
        risk_profile="High",
        return_expectation=10.0
    )
    
    # Define aggressive portfolio allocation for long-term SIP
    portfolio_allocation = PortfolioAllocation(
        sp500=55.0,
        small_cap=20.0,
        bonds=15.0,
        real_estate=8.0,
        gold=2.0
    )
    
    # Define expected returns
    expected_returns = {
        'sp500': 0.10,
        'small_cap': 0.12,
        't_bills': 0.025,
        't_bonds': 0.05,
        'corporate_bonds': 0.06,
        'real_estate': 0.09,
        'gold': 0.07
    }
    
    # Create simulation input
    simulation_input = SimulationInput(
        user_input=user_input,
        portfolio_allocation=portfolio_allocation,
        expected_returns=expected_returns,
        simulation_runs=1000
    )
    
    # Run simulation
    result = agent.simulate_portfolio(simulation_input)
    
    # Print results
    print_simulation_results(result, "SIP Investment ($5K monthly for 20 years)")


def demo_retirement_planning():
    """Demonstrate retirement planning with withdrawals."""
    print_separator("RETIREMENT PLANNING SIMULATION")
    
    # Create simulation agent
    agent = SimulationAgent()
    
    # Define user input for retirement planning
    user_input = UserInputModel(
        investment_amount=800000,  # $800K retirement corpus
        investment_type="lumpsum",
        tenure_years=25,  # 25-year retirement period
        risk_profile="Low",
        return_expectation=5.0,
        withdrawal_preferences={
            "annual_withdrawal": 45000,  # $45K annual withdrawal
            "start_year": 1
        }
    )
    
    # Define conservative portfolio allocation for retirement
    portfolio_allocation = PortfolioAllocation(
        sp500=25.0,
        small_cap=0.0,
        bonds=60.0,
        real_estate=10.0,
        gold=5.0
    )
    
    # Define conservative expected returns
    expected_returns = {
        'sp500': 0.08,
        'small_cap': 0.10,
        't_bills': 0.02,
        't_bonds': 0.045,
        'corporate_bonds': 0.055,
        'real_estate': 0.07,
        'gold': 0.06
    }
    
    # Create simulation input
    simulation_input = SimulationInput(
        user_input=user_input,
        portfolio_allocation=portfolio_allocation,
        expected_returns=expected_returns,
        simulation_runs=500
    )
    
    # Run simulation
    result = agent.simulate_portfolio(simulation_input)
    
    # Print results
    print_simulation_results(result, "Retirement Planning ($800K with $45K annual withdrawals)")


def demo_education_savings():
    """Demonstrate education savings simulation."""
    print_separator("EDUCATION SAVINGS SIMULATION")
    
    # Create simulation agent
    agent = SimulationAgent()
    
    # Define user input for education savings
    user_input = UserInputModel(
        investment_amount=1200,  # $1,200 monthly SIP
        investment_type="sip",
        tenure_years=15,  # Child is 3, college at 18
        risk_profile="Moderate",
        return_expectation=8.0
    )
    
    # Define moderate portfolio allocation for education savings
    portfolio_allocation = PortfolioAllocation(
        sp500=50.0,
        small_cap=15.0,
        bonds=25.0,
        real_estate=7.0,
        gold=3.0
    )
    
    # Define expected returns
    expected_returns = {
        'sp500': 0.095,
        'small_cap': 0.11,
        't_bills': 0.025,
        't_bonds': 0.048,
        'corporate_bonds': 0.057,
        'real_estate': 0.085,
        'gold': 0.065
    }
    
    # Create simulation input
    simulation_input = SimulationInput(
        user_input=user_input,
        portfolio_allocation=portfolio_allocation,
        expected_returns=expected_returns,
        simulation_runs=500
    )
    
    # Run simulation
    result = agent.simulate_portfolio(simulation_input)
    
    # Print results
    print_simulation_results(result, "Education Savings ($1.2K monthly for 15 years)")


def demo_wealth_building():
    """Demonstrate high net worth wealth building."""
    print_separator("WEALTH BUILDING SIMULATION")
    
    # Create simulation agent
    agent = SimulationAgent()
    
    # Define user input for wealth building
    user_input = UserInputModel(
        investment_amount=1500000,  # $1.5M lumpsum
        investment_type="lumpsum",
        tenure_years=12,
        risk_profile="High",
        return_expectation=11.0
    )
    
    # Define growth-focused portfolio allocation
    portfolio_allocation = PortfolioAllocation(
        sp500=60.0,
        small_cap=25.0,
        bonds=5.0,
        real_estate=8.0,
        gold=2.0
    )
    
    # Define higher expected returns for growth focus
    expected_returns = {
        'sp500': 0.105,
        'small_cap': 0.125,
        't_bills': 0.025,
        't_bonds': 0.05,
        'corporate_bonds': 0.06,
        'real_estate': 0.09,
        'gold': 0.07
    }
    
    # Create simulation input
    simulation_input = SimulationInput(
        user_input=user_input,
        portfolio_allocation=portfolio_allocation,
        expected_returns=expected_returns,
        simulation_runs=1000
    )
    
    # Run simulation
    result = agent.simulate_portfolio(simulation_input)
    
    # Print results
    print_simulation_results(result, "Wealth Building ($1.5M for 12 years)")


def demo_portfolio_comparison():
    """Demonstrate comparison between different portfolio allocations."""
    print_separator("PORTFOLIO ALLOCATION COMPARISON")
    
    # Create simulation agent
    agent = SimulationAgent()
    
    # Common user input
    user_input = UserInputModel(
        investment_amount=200000,
        investment_type="lumpsum",
        tenure_years=10,
        risk_profile="Moderate",
        return_expectation=8.0
    )
    
    # Expected returns
    expected_returns = {
        'sp500': 0.095,
        'small_cap': 0.11,
        't_bills': 0.025,
        't_bonds': 0.048,
        'corporate_bonds': 0.057,
        'real_estate': 0.085,
        'gold': 0.065
    }
    
    # Define different portfolio allocations
    portfolios = {
        "Conservative": PortfolioAllocation(
            sp500=20.0, small_cap=0.0, bonds=65.0, real_estate=10.0, gold=5.0
        ),
        "Moderate": PortfolioAllocation(
            sp500=45.0, small_cap=10.0, bonds=30.0, real_estate=10.0, gold=5.0
        ),
        "Aggressive": PortfolioAllocation(
            sp500=60.0, small_cap=25.0, bonds=5.0, real_estate=8.0, gold=2.0
        )
    }
    
    results = {}
    
    # Run simulations for each portfolio
    for name, allocation in portfolios.items():
        simulation_input = SimulationInput(
            user_input=user_input,
            portfolio_allocation=allocation,
            expected_returns=expected_returns,
            simulation_runs=500
        )
        
        result = agent.simulate_portfolio(simulation_input)
        results[name] = result
    
    # Print comparison
    print("\n📊 Portfolio Comparison Summary:")
    print("-" * 80)
    print(f"{'Portfolio':<12} {'Final Value':<15} {'CAGR':<8} {'Cum. Return':<12} {'Risk (Vol.)':<12}")
    print("-" * 80)
    
    for name, result in results.items():
        if result.success:
            volatility = result.simulation_statistics.get('std_deviation', 0) * 100
            print(f"{name:<12} ${result.final_value:<14,.0f} {result.cagr:<7.2f}% {result.cumulative_return:<11.2f}% {volatility:<11.2f}%")
        else:
            print(f"{name:<12} {'Failed':<15} {'N/A':<8} {'N/A':<12} {'N/A':<12}")


def demo_monte_carlo_analysis():
    """Demonstrate Monte Carlo risk analysis."""
    print_separator("MONTE CARLO RISK ANALYSIS")
    
    # Create simulation agent
    agent = SimulationAgent()
    
    # Define user input
    user_input = UserInputModel(
        investment_amount=300000,
        investment_type="lumpsum",
        tenure_years=8,
        risk_profile="Moderate",
        return_expectation=8.0
    )
    
    # Define portfolio allocation
    portfolio_allocation = PortfolioAllocation(
        sp500=50.0,
        small_cap=15.0,
        bonds=25.0,
        real_estate=7.0,
        gold=3.0
    )
    
    # Define expected returns
    expected_returns = {
        'sp500': 0.095,
        'small_cap': 0.11,
        't_bills': 0.025,
        't_bonds': 0.048,
        'corporate_bonds': 0.057,
        'real_estate': 0.085,
        'gold': 0.065
    }
    
    # Create simulation input with high number of runs
    simulation_input = SimulationInput(
        user_input=user_input,
        portfolio_allocation=portfolio_allocation,
        expected_returns=expected_returns,
        simulation_runs=5000  # High number for detailed analysis
    )
    
    # Run simulation
    result = agent.simulate_portfolio(simulation_input)
    
    if result.success:
        stats = result.simulation_statistics
        
        print(f"\n🎲 Detailed Monte Carlo Analysis (5,000 simulations):")
        print("-" * 50)
        print(f"Expected Portfolio Return: {stats.get('mean_return', 0)*100:.2f}%")
        print(f"Portfolio Volatility: {stats.get('std_deviation', 0)*100:.2f}%")
        print(f"Median Return: {stats.get('median_return', 0)*100:.2f}%")
        print(f"\nReturn Distribution:")
        print(f"  5th Percentile (Worst 5%): {stats.get('percentile_5', 0)*100:.2f}%")
        print(f"  95th Percentile (Best 5%): {stats.get('percentile_95', 0)*100:.2f}%")
        print(f"  Range: {(stats.get('percentile_95', 0) - stats.get('percentile_5', 0))*100:.2f}%")
        print(f"\nRisk Metrics:")
        print(f"  Probability of Positive Returns: {stats.get('probability_positive', 0)*100:.1f}%")
        print(f"  Probability of Negative Returns: {(1-stats.get('probability_positive', 0))*100:.1f}%")
        
        # Calculate value at risk
        initial_value = user_input.investment_amount
        percentile_5_value = initial_value * (1 + stats.get('percentile_5', 0)) ** user_input.tenure_years
        percentile_95_value = initial_value * (1 + stats.get('percentile_95', 0)) ** user_input.tenure_years
        
        print(f"\nValue Projections:")
        print(f"  Expected Final Value: ${result.final_value:,.2f}")
        print(f"  5th Percentile Value: ${percentile_5_value:,.2f}")
        print(f"  95th Percentile Value: ${percentile_95_value:,.2f}")
        print(f"  Potential Loss (5% scenario): ${initial_value - percentile_5_value:,.2f}")
        print(f"  Potential Gain (95% scenario): ${percentile_95_value - initial_value:,.2f}")


def main():
    """Run all simulation demos."""
    print("🚀 Financial Returns Optimizer - Simulation Agent Demo")
    print("This demo showcases various investment simulation scenarios")
    
    try:
        # Run all demo scenarios
        demo_lumpsum_investment()
        demo_sip_investment()
        demo_retirement_planning()
        demo_education_savings()
        demo_wealth_building()
        demo_portfolio_comparison()
        demo_monte_carlo_analysis()
        
        print_separator("DEMO COMPLETED")
        print("✅ All simulation scenarios completed successfully!")
        print("\n📝 Key Takeaways:")
        print("• Lumpsum investments benefit from immediate compound growth")
        print("• SIP investments provide rupee cost averaging benefits")
        print("• Portfolio allocation significantly impacts risk and returns")
        print("• Monte Carlo analysis helps understand potential outcomes")
        print("• Withdrawal strategies affect long-term portfolio sustainability")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()