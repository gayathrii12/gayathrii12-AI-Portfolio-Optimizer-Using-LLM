"""
Portfolio Allocation Engine Demo

Demonstrates the functionality of the Portfolio Allocation Engine
with different risk profiles and validation features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.portfolio_allocation_engine import PortfolioAllocationEngine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Demonstrate Portfolio Allocation Engine functionality."""
    
    print("=" * 60)
    print("Portfolio Allocation Engine Demo")
    print("=" * 60)
    
    # Initialize the engine
    engine = PortfolioAllocationEngine()
    
    print(f"\nInitialized {engine.name}")
    print(f"Supported risk profiles: {engine.get_supported_risk_profiles()}")
    
    # Demonstrate each risk profile
    risk_profiles = ['low', 'moderate', 'high']
    
    for risk_profile in risk_profiles:
        print(f"\n{'-' * 40}")
        print(f"{risk_profile.upper()} RISK ALLOCATION")
        print(f"{'-' * 40}")
        
        # Get allocation
        allocation = engine.get_allocation_by_risk_profile(risk_profile)
        
        # Display detailed allocation
        print("\nDetailed Asset Allocation:")
        for asset, percentage in allocation.to_dict().items():
            print(f"  {asset.replace('_', ' ').title():<18}: {percentage:>6.1f}%")
        
        # Display summary by category
        summary = engine.get_allocation_summary(risk_profile)
        print(f"\nAllocation Summary:")
        print(f"  Equity (Stocks)    : {summary['equity']:>6.1f}%")
        print(f"  Bonds              : {summary['bonds']:>6.1f}%")
        print(f"  Alternatives       : {summary['alternatives']:>6.1f}%")
        
        # Validate allocation
        is_valid = engine.validate_allocation(allocation.to_dict())
        print(f"\nAllocation Valid     : {is_valid}")
        
        # Show total
        total = sum(allocation.to_dict().values())
        print(f"Total Allocation     : {total:.1f}%")
    
    # Demonstrate validation with invalid allocations
    print(f"\n{'-' * 40}")
    print("VALIDATION EXAMPLES")
    print(f"{'-' * 40}")
    
    # Valid allocation
    valid_allocation = {
        'sp500': 30.0,
        'small_cap': 20.0,
        't_bills': 10.0,
        't_bonds': 20.0,
        'corporate_bonds': 20.0,
        'real_estate': 0.0,
        'gold': 0.0
    }
    
    print(f"\nValid allocation (sums to 100%): {engine.validate_allocation(valid_allocation)}")
    
    # Invalid allocation - doesn't sum to 100%
    invalid_allocation = {
        'sp500': 30.0,
        'small_cap': 20.0,
        't_bills': 10.0,
        't_bonds': 20.0,
        'corporate_bonds': 10.0,  # Total = 90%
        'real_estate': 0.0,
        'gold': 0.0
    }
    
    print(f"Invalid allocation (sums to 90%): {engine.validate_allocation(invalid_allocation)}")
    
    # Demonstrate error handling
    print(f"\n{'-' * 40}")
    print("ERROR HANDLING")
    print(f"{'-' * 40}")
    
    try:
        engine.get_allocation_by_risk_profile('invalid')
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    # Show risk progression
    print(f"\n{'-' * 40}")
    print("RISK PROGRESSION ANALYSIS")
    print(f"{'-' * 40}")
    
    print("\nEquity Allocation by Risk Level:")
    for risk in risk_profiles:
        allocation = engine.get_allocation_by_risk_profile(risk)
        equity_pct = allocation.get_equity_percentage()
        print(f"  {risk.capitalize():<10}: {equity_pct:>6.1f}%")
    
    print("\nBonds Allocation by Risk Level:")
    for risk in risk_profiles:
        allocation = engine.get_allocation_by_risk_profile(risk)
        bonds_pct = allocation.get_bonds_percentage()
        print(f"  {risk.capitalize():<10}: {bonds_pct:>6.1f}%")
    
    print(f"\n{'-' * 40}")
    print("Demo completed successfully!")
    print(f"{'-' * 40}")


if __name__ == "__main__":
    main()