#!/usr/bin/env python3
"""
Rebalancing Agent Demo

This demo showcases the enhanced rebalancing agent functionality including:
- Time-based rebalancing rules
- Equity reduction logic
- Rebalancing schedule calculation
- Impact on portfolio projections
- Visualization data preparation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.rebalancing_agent import RebalancingAgent
from utils.visualization_data import VisualizationDataPreparator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_basic_rebalancing():
    """Demonstrate basic rebalancing functionality"""
    print("\n" + "="*60)
    print("BASIC REBALANCING DEMO")
    print("="*60)
    
    # Initialize agent
    agent = RebalancingAgent()
    
    # Sample portfolio state
    state = {
        'portfolio_allocation': {
            'sp500': 50.0,
            'small_cap': 20.0,
            't_bills': 10.0,
            't_bonds': 15.0,
            'corporate_bonds': 5.0,
            'real_estate': 0.0,
            'gold': 0.0
        },
        'investment_horizon': 10,
        'rebalancing_frequency': 2,
        'equity_reduction_rate': 5.0
    }
    
    print(f"Initial Portfolio Allocation:")
    for asset, allocation in state['portfolio_allocation'].items():
        if allocation > 0:
            print(f"  {asset}: {allocation}%")
    
    print(f"\nRebalancing Parameters:")
    print(f"  Investment Horizon: {state['investment_horizon']} years")
    print(f"  Rebalancing Frequency: Every {state['rebalancing_frequency']} years")
    print(f"  Equity Reduction Rate: {state['equity_reduction_rate']}% per rebalancing")
    
    # Create rebalancing strategy
    result = agent.create_rebalancing_strategy(state)
    
    if result.get('agent_status') == 'rebalancing_complete':
        print(f"\n✅ Rebalancing strategy created successfully!")
        
        schedule = result['rebalancing_schedule']
        print(f"\nRebalancing Schedule ({len(schedule)} events):")
        
        for event in schedule:
            year = event['year']
            allocation = event['allocation']
            equity_total = allocation.get('sp500', 0) + allocation.get('small_cap', 0)
            bond_total = (allocation.get('t_bills', 0) + 
                         allocation.get('t_bonds', 0) + 
                         allocation.get('corporate_bonds', 0))
            
            print(f"\n  Year {year}:")
            print(f"    Equity: {equity_total:.1f}% | Bonds: {bond_total:.1f}%")
            if 'changes' in event and event['changes']:
                print(f"    Changes: {event['changes']}")
            print(f"    Rationale: {event['rationale']}")
        
        print(f"\n📊 Final Allocation:")
        final_allocation = result['final_allocation']
        for asset, allocation in final_allocation.items():
            if allocation > 0:
                print(f"  {asset}: {allocation:.1f}%")
        
        print(f"\n📝 Strategy Rationale:")
        print(result['rebalancing_rationale'])
    
    else:
        print(f"❌ Rebalancing strategy failed: {result.get('error', 'Unknown error')}")
    
    return result


def demo_rebalancing_impact():
    """Demonstrate rebalancing impact calculation"""
    print("\n" + "="*60)
    print("REBALANCING IMPACT DEMO")
    print("="*60)
    
    agent = RebalancingAgent()
    
    # Create initial state with rebalancing strategy
    state = {
        'portfolio_allocation': {
            'sp500': 60.0,
            'small_cap': 20.0,
            't_bills': 5.0,
            't_bonds': 10.0,
            'corporate_bonds': 5.0,
            'real_estate': 0.0,
            'gold': 0.0
        },
        'investment_horizon': 8,
        'rebalancing_frequency': 2,
        'equity_reduction_rate': 7.5,
        'predicted_returns': {
            'sp500': 0.10,
            'small_cap': 0.12,
            't_bills': 0.03,
            't_bonds': 0.05,
            'corporate_bonds': 0.06,
            'real_estate': 0.08,
            'gold': 0.07
        },
        'investment_amount': 500000.0,
        'investment_type': 'lump_sum',
        'monthly_amount': 0
    }
    
    print(f"Investment Details:")
    print(f"  Initial Amount: ${state['investment_amount']:,.2f}")
    print(f"  Investment Type: {state['investment_type']}")
    print(f"  Investment Horizon: {state['investment_horizon']} years")
    
    # Create rebalancing strategy
    result = agent.create_rebalancing_strategy(state)
    
    if result.get('agent_status') == 'rebalancing_complete':
        # Calculate impact
        impact_result = agent.calculate_rebalancing_impact(result)
        
        if 'rebalancing_benefits' in impact_result:
            benefits = impact_result['rebalancing_benefits']
            
            print(f"\n💰 Rebalancing Impact Analysis:")
            print(f"  Final Value WITH Rebalancing: ${benefits['final_value_with_rebalancing']:,.2f}")
            print(f"  Final Value WITHOUT Rebalancing: ${benefits['final_value_without_rebalancing']:,.2f}")
            print(f"  Benefit Amount: ${benefits['benefit_amount']:,.2f}")
            print(f"  Benefit Percentage: {benefits['benefit_percentage']:+.2f}%")
            print(f"  Risk Reduction Score: {benefits['risk_reduction_score']:.1f}")
            print(f"\n🎯 Recommendation: {benefits['recommendation']}")
        
        # Prepare visualization data
        viz_result = agent.prepare_allocation_visualization_data(impact_result)
        
        if 'allocation_changes_summary' in viz_result:
            summary = viz_result['allocation_changes_summary']
            
            print(f"\n📈 Allocation Changes Summary:")
            print(f"  Total Equity Change: {summary['total_equity_change']:+.1f}%")
            print(f"  Total Bond Change: {summary['total_bond_change']:+.1f}%")
            print(f"  Number of Rebalancing Events: {summary['rebalancing_events']}")
            print(f"  Years Covered: {summary['years_covered']}")
            
            if summary['largest_decreases']:
                print(f"\n  Largest Decreases:")
                for decrease in summary['largest_decreases']:
                    print(f"    {decrease['asset']}: -{decrease['change']:.1f}%")
            
            if summary['largest_increases']:
                print(f"\n  Largest Increases:")
                for increase in summary['largest_increases']:
                    print(f"    {increase['asset']}: +{increase['change']:.1f}%")
    
    return impact_result


def demo_sip_with_rebalancing():
    """Demonstrate SIP (Systematic Investment Plan) with rebalancing"""
    print("\n" + "="*60)
    print("SIP WITH REBALANCING DEMO")
    print("="*60)
    
    agent = RebalancingAgent()
    
    state = {
        'portfolio_allocation': {
            'sp500': 40.0,
            'small_cap': 20.0,
            't_bills': 15.0,
            't_bonds': 20.0,
            'corporate_bonds': 5.0,
            'real_estate': 0.0,
            'gold': 0.0
        },
        'investment_horizon': 6,
        'rebalancing_frequency': 3,
        'equity_reduction_rate': 10.0,
        'predicted_returns': {
            'sp500': 0.09,
            'small_cap': 0.11,
            't_bills': 0.03,
            't_bonds': 0.05,
            'corporate_bonds': 0.06,
            'real_estate': 0.08,
            'gold': 0.07
        },
        'investment_amount': 100000.0,  # Initial lump sum
        'investment_type': 'sip',
        'monthly_amount': 5000.0  # Monthly SIP
    }
    
    print(f"SIP Investment Details:")
    print(f"  Initial Amount: ${state['investment_amount']:,.2f}")
    print(f"  Monthly SIP: ${state['monthly_amount']:,.2f}")
    print(f"  Total Monthly Contributions: ${state['monthly_amount'] * 12 * state['investment_horizon']:,.2f}")
    print(f"  Investment Horizon: {state['investment_horizon']} years")
    
    # Create rebalancing strategy and calculate impact
    result = agent.create_rebalancing_strategy(state)
    
    if result.get('agent_status') == 'rebalancing_complete':
        impact_result = agent.calculate_rebalancing_impact(result)
        
        if 'portfolio_projections_with_rebalancing' in impact_result:
            projections = impact_result['portfolio_projections_with_rebalancing']
            
            print(f"\n📊 Portfolio Growth with SIP + Rebalancing:")
            for projection in projections:
                year = projection['year']
                value = projection['end_period_value']
                allocation = projection['allocation']
                equity_pct = allocation.get('sp500', 0) + allocation.get('small_cap', 0)
                
                print(f"  Year {year}: ${value:,.2f} (Equity: {equity_pct:.1f}%)")
        
        if 'rebalancing_benefits' in impact_result:
            benefits = impact_result['rebalancing_benefits']
            print(f"\n💡 SIP + Rebalancing Benefits:")
            print(f"  Final Portfolio Value: ${benefits['final_value_with_rebalancing']:,.2f}")
            print(f"  Benefit vs Static Allocation: {benefits['benefit_percentage']:+.2f}%")
    
    return impact_result


def demo_visualization_data():
    """Demonstrate visualization data preparation"""
    print("\n" + "="*60)
    print("VISUALIZATION DATA DEMO")
    print("="*60)
    
    preparator = VisualizationDataPreparator()
    
    # Sample rebalancing data
    with_rebalancing = [
        {'year': 0, 'portfolio_value': 100000, 'end_period_value': 110000},
        {'year': 2, 'portfolio_value': 110000, 'end_period_value': 125000},
        {'year': 4, 'portfolio_value': 125000, 'end_period_value': 142000},
        {'year': 6, 'portfolio_value': 142000, 'end_period_value': 160000}
    ]
    
    without_rebalancing = [
        {'year': 0, 'portfolio_value': 100000},
        {'year': 2, 'portfolio_value': 120000},
        {'year': 4, 'portfolio_value': 138000},
        {'year': 6, 'portfolio_value': 155000}
    ]
    
    # Prepare comparison data
    comparison_data = preparator.prepare_rebalancing_comparison_data(
        with_rebalancing, without_rebalancing
    )
    
    print(f"📊 Rebalancing vs Static Allocation Comparison:")
    print(f"{'Year':<6} {'With Rebal.':<15} {'Without Rebal.':<15} {'Outperformance':<15}")
    print("-" * 60)
    
    for point in comparison_data:
        year = point['year']
        with_val = point['with_rebalancing_formatted']
        without_val = point['without_rebalancing_formatted']
        outperf = point['outperformance_formatted']
        print(f"{year:<6} {with_val:<15} {without_val:<15} {outperf:<15}")
    
    # Sample rebalancing events
    rebalancing_schedule = [
        {
            'year': 0,
            'allocation': {'sp500': 60, 'small_cap': 20, 't_bills': 20},
            'changes': {},
            'rationale': 'Initial allocation'
        },
        {
            'year': 2,
            'allocation': {'sp500': 55, 'small_cap': 15, 't_bills': 30},
            'changes': {'sp500': -5, 'small_cap': -5, 't_bills': 10},
            'rationale': 'Reduce equity exposure by 5%'
        },
        {
            'year': 4,
            'allocation': {'sp500': 50, 'small_cap': 10, 't_bills': 40},
            'changes': {'sp500': -5, 'small_cap': -5, 't_bills': 10},
            'rationale': 'Further risk reduction'
        }
    ]
    
    # Prepare events data
    events_data = preparator.prepare_rebalancing_events_data(rebalancing_schedule)
    
    print(f"\n🔄 Rebalancing Events Timeline:")
    for event in events_data:
        year = event['year']
        equity_change = event['equity_change']
        bond_change = event['bond_change']
        rationale = event['rationale']
        
        print(f"\n  Year {year}:")
        print(f"    Equity Change: {equity_change:+.1f}%")
        print(f"    Bond Change: {bond_change:+.1f}%")
        print(f"    Rationale: {rationale}")
    
    print(f"\n✅ Visualization data prepared successfully!")
    print(f"   - Comparison data points: {len(comparison_data)}")
    print(f"   - Rebalancing events: {len(events_data)}")


def main():
    """Run all rebalancing demos"""
    print("🚀 REBALANCING AGENT COMPREHENSIVE DEMO")
    print("This demo showcases the enhanced rebalancing system functionality")
    
    try:
        # Run demos
        demo_basic_rebalancing()
        demo_rebalancing_impact()
        demo_sip_with_rebalancing()
        demo_visualization_data()
        
        print("\n" + "="*60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("• Time-based rebalancing rules with equity reduction")
        print("• Rebalancing schedule calculation and validation")
        print("• Impact analysis comparing rebalanced vs static portfolios")
        print("• Support for different investment types (Lump Sum, SIP)")
        print("• Comprehensive visualization data preparation")
        print("• Risk reduction scoring and recommendations")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())