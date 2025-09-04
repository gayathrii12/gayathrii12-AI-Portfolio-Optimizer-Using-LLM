"""
Demo script for visualization data preparation module.

This script demonstrates how to use the VisualizationDataPreparator
to format data for React chart components.
"""

import sys
import os
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_models import PortfolioAllocation, ProjectionResult, RiskMetrics
from utils.visualization_data import VisualizationDataPreparator


def main():
    """Demonstrate visualization data preparation functionality."""
    print("=== Visualization Data Preparation Demo ===\n")
    
    # Initialize the preparator
    preparator = VisualizationDataPreparator()
    
    # Sample data
    allocation = PortfolioAllocation(
        sp500=50.0,
        small_cap=20.0,
        bonds=20.0,
        real_estate=8.0,
        gold=2.0
    )
    
    projections = [
        ProjectionResult(year=1, portfolio_value=110000.0, annual_return=10.0, cumulative_return=10.0),
        ProjectionResult(year=2, portfolio_value=121000.0, annual_return=10.0, cumulative_return=21.0),
        ProjectionResult(year=3, portfolio_value=133100.0, annual_return=10.0, cumulative_return=33.1),
        ProjectionResult(year=5, portfolio_value=161051.0, annual_return=10.0, cumulative_return=61.05)
    ]
    
    risk_metrics = RiskMetrics(
        alpha=2.5,
        beta=1.2,
        volatility=18.0,
        sharpe_ratio=0.8,
        max_drawdown=-25.0
    )
    
    initial_investment = 100000.0
    
    # 1. Pie Chart Data
    print("1. PIE CHART DATA (Portfolio Allocation)")
    pie_data = preparator.prepare_pie_chart_data(allocation)
    for point in pie_data:
        print(f"   {point.name}: {point.percentage} (Color: {point.color})")
    print()
    
    # 2. Line Chart Data
    print("2. LINE CHART DATA (Portfolio Value Over Time)")
    line_data = preparator.prepare_line_chart_data(projections, initial_investment)
    for point in line_data:
        print(f"   Year {point.year}: {point.formatted_value} "
              f"(Return: {point.cumulative_return or 0:.1f}%)")
    print()
    
    # 3. Comparison Chart Data
    print("3. COMPARISON CHART DATA (Portfolio vs S&P 500)")
    comparison_data = preparator.prepare_comparison_chart_data(projections, initial_investment)
    for point in comparison_data:
        print(f"   Year {point.year}: Portfolio ${point.portfolio_value:,.0f} vs "
              f"Benchmark ${point.benchmark_value:,.0f} "
              f"(Outperformance: {point.outperformance:+.1f}%)")
    print()
    
    # 4. Risk Visualization Data
    print("4. RISK VISUALIZATION DATA")
    risk_data = preparator.prepare_risk_visualization_data(risk_metrics)
    print(f"   Risk Score: {risk_data['risk_score']}/100 ({risk_data['risk_level']} Risk)")
    print("   Risk Metrics:")
    for metric in risk_data['portfolio_metrics']:
        print(f"     {metric['metric']}: {metric['value']} "
              f"(Benchmark: {metric['benchmark']})")
    print()
    
    # 5. Data Validation
    print("5. DATA VALIDATION")
    pie_data_dict = [point.model_dump() for point in pie_data]
    line_data_dict = [point.model_dump() for point in line_data]
    comparison_data_dict = [point.model_dump() for point in comparison_data]
    
    validations = [
        ("Pie Chart", preparator.validate_chart_data(pie_data_dict, "pie")),
        ("Line Chart", preparator.validate_chart_data(line_data_dict, "line")),
        ("Comparison Chart", preparator.validate_chart_data(comparison_data_dict, "comparison")),
        ("Risk Chart", preparator.validate_chart_data(risk_data, "risk"))
    ]
    
    for chart_type, is_valid in validations:
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"   {chart_type}: {status}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()