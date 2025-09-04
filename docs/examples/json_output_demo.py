"""
Demo script for JSON output generator functionality.

This script demonstrates how to use the JSONOutputGenerator to create
React-compatible JSON output for portfolio analysis results.
"""

import json
from utils.json_output import JSONOutputGenerator
from models.data_models import (
    UserInputModel,
    PortfolioAllocation,
    ProjectionResult,
    RiskMetrics
)


def main():
    """Demonstrate JSON output generator functionality."""
    print("=" * 80)
    print("JSON OUTPUT GENERATOR DEMO".center(80))
    print("=" * 80)
    print()
    
    # Initialize the JSON output generator
    json_generator = JSONOutputGenerator()
    print("✓ JSONOutputGenerator initialized")
    
    # Create sample data
    print("\n1. Creating sample data...")
    
    user_input = UserInputModel(
        investment_amount=250000.0,
        investment_type="sip",
        tenure_years=15,
        risk_profile="Moderate",
        return_expectation=12.0,
        rebalancing_preferences={"frequency": "annual", "threshold": 5.0}
    )
    print(f"   User Input: ${user_input.investment_amount:,.2f} {user_input.investment_type} for {user_input.tenure_years} years")
    
    allocation = PortfolioAllocation(
        sp500=40.0,
        small_cap=20.0,
        bonds=25.0,
        real_estate=10.0,
        gold=5.0
    )
    print(f"   Allocation: {allocation.sp500}% S&P500, {allocation.small_cap}% Small Cap, {allocation.bonds}% Bonds")
    
    projections = [
        ProjectionResult(year=1, portfolio_value=275000.0, annual_return=10.0, cumulative_return=10.0),
        ProjectionResult(year=5, portfolio_value=402500.0, annual_return=8.5, cumulative_return=61.0),
        ProjectionResult(year=10, portfolio_value=647500.0, annual_return=9.2, cumulative_return=159.0),
        ProjectionResult(year=15, portfolio_value=1025000.0, annual_return=9.6, cumulative_return=310.0)
    ]
    print(f"   Projections: {len(projections)} years of data")
    
    risk_metrics = RiskMetrics(
        alpha=1.8,
        beta=0.92,
        volatility=14.5,
        sharpe_ratio=0.78,
        max_drawdown=-18.2
    )
    print(f"   Risk Metrics: Alpha={risk_metrics.alpha}%, Beta={risk_metrics.beta}")
    
    # Generate individual JSON components
    print("\n2. Generating individual JSON components...")
    
    allocation_json = json_generator.generate_allocation_json(allocation)
    print("   ✓ Allocation JSON generated")
    
    projections_json = json_generator.generate_projections_json(projections, user_input)
    print("   ✓ Projections JSON generated")
    
    benchmark_json = json_generator.generate_benchmark_json(projections)
    print("   ✓ Benchmark JSON generated")
    
    risk_metrics_json = json_generator.generate_risk_metrics_json(risk_metrics)
    print("   ✓ Risk Metrics JSON generated")
    
    # Generate complete JSON output
    print("\n3. Generating complete JSON output...")
    
    complete_json = json_generator.generate_complete_json(
        allocation, projections, risk_metrics, user_input
    )
    print("   ✓ Complete JSON structure generated")
    
    # Export to different formats
    print("\n4. Exporting to different formats...")
    
    json_string = json_generator.export_to_json_string(complete_json, indent=2)
    json_dict = json_generator.export_to_json_dict(complete_json)
    
    print(f"   ✓ JSON string exported ({len(json_string)} characters)")
    print(f"   ✓ JSON dictionary exported ({len(json_dict)} keys)")
    
    # Validate JSON schema
    print("\n5. Validating JSON schema...")
    
    is_valid = json_generator.validate_json_schema(json_dict)
    print(f"   ✓ JSON schema validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Display sample JSON structure
    print("\n6. Sample JSON Output Structure:")
    print("-" * 50)
    
    # Show allocation section
    print("Allocation JSON:")
    allocation_sample = json.dumps(json_dict["allocation"], indent=2)
    print(allocation_sample)
    print()
    
    # Show first few projections
    print("Projections JSON (first 2 entries):")
    projections_sample = json.dumps(json_dict["projections"][:2], indent=2)
    print(projections_sample)
    print()
    
    # Show risk metrics
    print("Risk Metrics JSON:")
    risk_sample = json.dumps(json_dict["risk_metrics"], indent=2)
    print(risk_sample)
    print()
    
    # Show visualization data sample
    print("Visualization Data JSON (pie chart sample):")
    viz_sample = json.dumps(json_dict["visualization_data"]["pie_chart_data"][:2], indent=2)
    print(viz_sample)
    print()
    
    # Display React integration information
    print("\n7. React Integration Information:")
    print("-" * 50)
    
    print("React Component Usage Examples:")
    print()
    print("// Pie Chart Component")
    print("const allocationData = jsonOutput.allocation;")
    print("const pieData = jsonOutput.visualization_data.pie_chart_data;")
    print()
    print("// Line Chart Component")
    print("const lineData = jsonOutput.visualization_data.line_chart_data;")
    print("const projections = jsonOutput.projections;")
    print()
    print("// Risk Metrics Display")
    print("const riskMetrics = jsonOutput.risk_metrics;")
    print("const alpha = riskMetrics.alpha;")
    print("const beta = riskMetrics.beta;")
    print()
    
    # Show JSON schema information
    print("\n8. JSON Schema Information:")
    print("-" * 50)
    
    schema = json_generator.get_json_schema()
    print(f"Schema version: {json_dict['metadata']['schema_version']}")
    print(f"Schema properties: {len(schema['properties'])} main sections")
    print("Main sections:", list(schema['properties'].keys()))
    print()
    
    # Performance and size information
    print("\n9. Performance Information:")
    print("-" * 50)
    
    print(f"JSON string size: {len(json_string):,} characters")
    print(f"JSON dict size: {len(str(json_dict)):,} characters")
    print(f"Allocation data points: {len(json_dict['allocation'])}")
    print(f"Projection data points: {len(json_dict['projections'])}")
    print(f"Visualization pie chart items: {len(json_dict['visualization_data']['pie_chart_data'])}")
    print(f"Visualization line chart items: {len(json_dict['visualization_data']['line_chart_data'])}")
    print()
    
    # Save sample output to file
    print("\n10. Saving sample output...")
    
    try:
        with open("output/sample_json_output.json", "w") as f:
            f.write(json_string)
        print("   ✓ Sample JSON saved to output/sample_json_output.json")
    except FileNotFoundError:
        print("   ⚠ Output directory not found, skipping file save")
    
    print("\n" + "=" * 80)
    print("JSON OUTPUT GENERATOR DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()