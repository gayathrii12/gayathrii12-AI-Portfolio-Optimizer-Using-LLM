#!/usr/bin/env python3
"""
Demonstration script for the Financial Returns Optimizer data models.

This script shows how to use the Pydantic models for validation and
demonstrates various validation scenarios.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_models import (
    UserInputModel,
    AssetReturns,
    PortfolioAllocation,
    ProjectionResult,
    RiskMetrics,
    ErrorResponse
)
from pydantic import ValidationError
import json


def demo_user_input_model():
    """Demonstrate UserInputModel validation."""
    print("=== UserInputModel Demo ===")
    
    # Valid user input
    try:
        user_input = UserInputModel(
            investment_amount=100000.0,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=12.0,
            rebalancing_preferences={"frequency": "annual", "threshold": 5.0}
        )
        print("✓ Valid user input created successfully")
        print(f"  Investment: ${user_input.investment_amount:,.2f} ({user_input.investment_type})")
        print(f"  Tenure: {user_input.tenure_years} years")
        print(f"  Risk Profile: {user_input.risk_profile}")
        print(f"  Expected Return: {user_input.return_expectation}%")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}")
    
    # Invalid user input
    try:
        invalid_input = UserInputModel(
            investment_amount=-1000.0,  # Invalid negative amount
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=12.0
        )
        print("✗ Should have failed validation")
    except ValidationError as e:
        print("✓ Correctly caught invalid negative investment amount")
    
    print()


def demo_asset_returns_model():
    """Demonstrate AssetReturns validation."""
    print("=== AssetReturns Demo ===")
    
    # Valid asset returns
    try:
        returns = AssetReturns(
            sp500=12.5,
            small_cap=15.2,
            t_bills=2.1,
            t_bonds=4.8,
            corporate_bonds=6.3,
            real_estate=8.9,
            gold=3.2,
            year=2023
        )
        print("✓ Valid asset returns created successfully")
        print(f"  S&P 500: {returns.sp500}%")
        print(f"  Small Cap: {returns.small_cap}%")
        print(f"  Real Estate: {returns.real_estate}%")
        print(f"  Year: {returns.year}")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}")
    
    # Invalid asset returns
    try:
        invalid_returns = AssetReturns(
            sp500=-150.0,  # Invalid: cannot lose more than 100%
            small_cap=15.2,
            t_bills=2.1,
            t_bonds=4.8,
            corporate_bonds=6.3,
            real_estate=8.9,
            gold=3.2,
            year=2023
        )
        print("✗ Should have failed validation")
    except ValidationError as e:
        print("✓ Correctly caught invalid return below -100%")
    
    print()


def demo_portfolio_allocation_model():
    """Demonstrate PortfolioAllocation validation."""
    print("=== PortfolioAllocation Demo ===")
    
    # Valid allocation
    try:
        allocation = PortfolioAllocation(
            sp500=40.0,
            small_cap=20.0,
            bonds=25.0,
            gold=5.0,
            real_estate=10.0
        )
        print("✓ Valid portfolio allocation created successfully")
        print(f"  S&P 500: {allocation.sp500}%")
        print(f"  Small Cap: {allocation.small_cap}%")
        print(f"  Bonds: {allocation.bonds}%")
        print(f"  Gold: {allocation.gold}%")
        print(f"  Real Estate: {allocation.real_estate}%")
        total = allocation.sp500 + allocation.small_cap + allocation.bonds + allocation.gold + allocation.real_estate
        print(f"  Total: {total}%")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}")
    
    # Invalid allocation (doesn't sum to 100%)
    try:
        invalid_allocation = PortfolioAllocation(
            sp500=40.0,
            small_cap=20.0,
            bonds=25.0,
            gold=5.0,
            real_estate=15.0  # Total = 105%
        )
        print("✗ Should have failed validation")
    except ValidationError as e:
        print("✓ Correctly caught allocation not summing to 100%")
    
    print()


def demo_projection_result_model():
    """Demonstrate ProjectionResult validation."""
    print("=== ProjectionResult Demo ===")
    
    # Valid projection result
    try:
        result = ProjectionResult(
            year=5,
            portfolio_value=150000.0,
            annual_return=12.5,
            cumulative_return=50.0
        )
        print("✓ Valid projection result created successfully")
        print(f"  Year: {result.year}")
        print(f"  Portfolio Value: ${result.portfolio_value:,.2f}")
        print(f"  Annual Return: {result.annual_return}%")
        print(f"  Cumulative Return: {result.cumulative_return}%")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}")
    
    print()


def demo_risk_metrics_model():
    """Demonstrate RiskMetrics validation."""
    print("=== RiskMetrics Demo ===")
    
    # Valid risk metrics
    try:
        metrics = RiskMetrics(
            alpha=2.5,
            beta=1.2,
            volatility=15.8,
            sharpe_ratio=1.4,
            max_drawdown=-12.3
        )
        print("✓ Valid risk metrics created successfully")
        print(f"  Alpha: {metrics.alpha}%")
        print(f"  Beta: {metrics.beta}")
        print(f"  Volatility: {metrics.volatility}%")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio}")
        print(f"  Max Drawdown: {metrics.max_drawdown}%")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}")
    
    # Invalid risk metrics
    try:
        invalid_metrics = RiskMetrics(
            alpha=2.5,
            beta=1.2,
            volatility=15.8,
            sharpe_ratio=1.4,
            max_drawdown=5.0  # Invalid: drawdown must be negative
        )
        print("✗ Should have failed validation")
    except ValidationError as e:
        print("✓ Correctly caught positive max drawdown")
    
    print()


def demo_json_serialization():
    """Demonstrate JSON serialization of models."""
    print("=== JSON Serialization Demo ===")
    
    # Create a valid user input
    user_input = UserInputModel(
        investment_amount=50000.0,
        investment_type="sip",
        tenure_years=15,
        risk_profile="High",
        return_expectation=15.0
    )
    
    # Serialize to JSON
    json_data = user_input.model_dump()
    json_string = json.dumps(json_data, indent=2)
    
    print("✓ Model serialized to JSON:")
    print(json_string)
    
    # Deserialize from JSON
    reconstructed = UserInputModel(**json_data)
    print("✓ Model reconstructed from JSON successfully")
    print(f"  Original amount: ${user_input.investment_amount}")
    print(f"  Reconstructed amount: ${reconstructed.investment_amount}")
    
    print()


if __name__ == "__main__":
    print("Financial Returns Optimizer - Data Models Demonstration")
    print("=" * 60)
    print()
    
    demo_user_input_model()
    demo_asset_returns_model()
    demo_portfolio_allocation_model()
    demo_projection_result_model()
    demo_risk_metrics_model()
    demo_json_serialization()
    
    print("Demo completed successfully! ✓")