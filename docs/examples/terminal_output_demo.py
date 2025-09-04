"""
Demo script for terminal output generator functionality.

This script demonstrates how to use the TerminalOutputGenerator
to create human-readable financial reports.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.terminal_output import TerminalOutputGenerator
from models.data_models import (
    PortfolioAllocation,
    ProjectionResult,
    RiskMetrics,
    UserInputModel
)


def main():
    """Demonstrate terminal output generation with sample data."""
    
    # Create sample user input
    user_input = UserInputModel(
        investment_amount=250000.0,
        investment_type="lumpsum",
        tenure_years=15,
        risk_profile="Moderate",
        return_expectation=9.0
    )
    
    # Create sample portfolio allocation
    allocation = PortfolioAllocation(
        sp500=45.0,
        small_cap=15.0,
        bonds=25.0,
        real_estate=10.0,
        gold=5.0
    )
    
    # Create sample projections (simplified growth)
    projections = []
    portfolio_value = user_input.investment_amount
    
    for year in range(1, user_input.tenure_years + 1):
        annual_return = 8.5 + (year % 3) * 1.5  # Varying returns
        portfolio_value *= (1 + annual_return / 100)
        cumulative_return = ((portfolio_value / user_input.investment_amount) - 1) * 100
        
        projections.append(ProjectionResult(
            year=year,
            portfolio_value=portfolio_value,
            annual_return=annual_return,
            cumulative_return=cumulative_return
        ))
    
    # Create sample risk metrics
    risk_metrics = RiskMetrics(
        alpha=2.1,
        beta=0.92,
        volatility=14.8,
        sharpe_ratio=0.68,
        max_drawdown=-18.5
    )
    
    # Initialize terminal output generator
    generator = TerminalOutputGenerator()
    
    print("=" * 80)
    print("TERMINAL OUTPUT GENERATOR DEMO".center(80))
    print("=" * 80)
    print()
    
    # Demonstrate individual sections
    print("1. Portfolio Allocation Section:")
    print("-" * 40)
    allocation_output = generator.format_portfolio_allocation(allocation, user_input)
    print(allocation_output)
    print()
    
    print("2. Year-by-Year Projections Section (first 5 years):")
    print("-" * 40)
    projections_output = generator.format_year_by_year_projections(
        projections[:5], user_input
    )
    print(projections_output)
    print()
    
    print("3. Risk Metrics Section:")
    print("-" * 40)
    risk_output = generator.format_risk_metrics(risk_metrics)
    print(risk_output)
    print()
    
    print("4. Strategy Explanation Section:")
    print("-" * 40)
    explanation_output = generator.generate_explanation(
        allocation, user_input, risk_metrics
    )
    print(explanation_output)
    print()
    
    print("5. Complete Report:")
    print("-" * 40)
    complete_report = generator.generate_complete_report(
        allocation, projections, risk_metrics, user_input
    )
    print(complete_report)


if __name__ == "__main__":
    main()