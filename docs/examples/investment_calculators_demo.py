#!/usr/bin/env python3
"""
Investment Calculators Demo

This script demonstrates the usage of the InvestmentCalculators class
for different investment strategies: Lump Sum, SIP, and SWP.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.investment_calculators import InvestmentCalculators
import json


def demo_lump_sum():
    """Demonstrate lump sum investment calculation."""
    print("=" * 60)
    print("LUMP SUM INVESTMENT DEMO")
    print("=" * 60)
    
    calculator = InvestmentCalculators()
    
    # Example: $100,000 lump sum investment for 10 years
    amount = 100000
    returns = {
        'sp500': 10.5,      # S&P 500 expected return
        'bonds': 4.2,       # Bonds expected return
        'real_estate': 7.8  # Real estate expected return
    }
    years = 10
    
    print(f"Investment Amount: ${amount:,}")
    print(f"Investment Period: {years} years")
    print(f"Expected Returns: {returns}")
    print()
    
    projections = calculator.calculate_lump_sum(amount, returns, years)
    summary = calculator.generate_investment_summary(projections)
    
    print("Year-by-Year Projections:")
    print("-" * 50)
    for projection in projections:
        print(f"Year {projection.year:2d}: ${projection.portfolio_value:12,.2f} "
              f"(Return: {projection.annual_return:.1f}%)")
    
    print("\nSummary:")
    print("-" * 30)
    print(f"Initial Investment: ${summary['initial_investment']:,.2f}")
    print(f"Final Value:        ${summary['final_value']:,.2f}")
    print(f"Total Return:       ${summary['total_return']:,.2f}")
    print(f"Total Return %:     {summary['total_return_percentage']:.2f}%")
    print(f"CAGR:              {summary['cagr']:.2f}%")
    print()


def demo_sip():
    """Demonstrate SIP (Systematic Investment Plan) calculation."""
    print("=" * 60)
    print("SIP (SYSTEMATIC INVESTMENT PLAN) DEMO")
    print("=" * 60)
    
    calculator = InvestmentCalculators()
    
    # Example: $2,000 monthly SIP for 15 years
    monthly_amount = 2000
    returns = {
        'diversified_equity': 12.0,  # Diversified equity fund
        'debt_fund': 6.5             # Debt fund
    }
    years = 15
    
    print(f"Monthly Investment: ${monthly_amount:,}")
    print(f"Investment Period: {years} years")
    print(f"Expected Returns: {returns}")
    print()
    
    projections = calculator.calculate_sip(monthly_amount, returns, years)
    summary = calculator.generate_investment_summary(projections)
    
    print("Year-by-Year Projections:")
    print("-" * 70)
    for projection in projections:
        print(f"Year {projection.year:2d}: ${projection.portfolio_value:12,.2f} "
              f"(Contributed: ${projection.cumulative_contributions:10,.2f})")
    
    print("\nSummary:")
    print("-" * 30)
    print(f"Total Contributions: ${summary['total_contributions']:,.2f}")
    print(f"Final Value:         ${summary['final_value']:,.2f}")
    print(f"Total Return:        ${summary['total_return']:,.2f}")
    print(f"Total Return %:      {summary['total_return_percentage']:.2f}%")
    print(f"CAGR:               {summary['cagr']:.2f}%")
    print()


def demo_swp():
    """Demonstrate SWP (Systematic Withdrawal Plan) calculation."""
    print("=" * 60)
    print("SWP (SYSTEMATIC WITHDRAWAL PLAN) DEMO")
    print("=" * 60)
    
    calculator = InvestmentCalculators()
    
    # Example: $800,000 retirement corpus with $4,000 monthly withdrawal
    initial_amount = 800000
    monthly_withdrawal = 4000
    returns = {
        'conservative_portfolio': 6.0  # Conservative portfolio return
    }
    years = 20
    
    print(f"Initial Corpus:      ${initial_amount:,}")
    print(f"Monthly Withdrawal:  ${monthly_withdrawal:,}")
    print(f"Withdrawal Period:   {years} years")
    print(f"Expected Returns:    {returns}")
    print()
    
    projections = calculator.calculate_swp(initial_amount, monthly_withdrawal, returns, years)
    summary = calculator.generate_investment_summary(projections)
    
    print("Year-by-Year Projections:")
    print("-" * 70)
    for projection in projections:
        print(f"Year {projection.year:2d}: ${projection.portfolio_value:12,.2f} "
              f"(Withdrawn: ${projection.cumulative_withdrawals:10,.2f})")
    
    print("\nSummary:")
    print("-" * 30)
    print(f"Initial Investment:  ${summary['initial_investment']:,.2f}")
    print(f"Final Value:         ${summary['final_value']:,.2f}")
    print(f"Total Withdrawals:   ${summary['total_withdrawals']:,.2f}")
    print(f"Portfolio Lasted:    {summary['investment_years']} years")
    print()


def demo_comparison():
    """Compare different investment strategies."""
    print("=" * 60)
    print("INVESTMENT STRATEGY COMPARISON")
    print("=" * 60)
    
    calculator = InvestmentCalculators()
    
    # Common parameters
    returns = {'balanced_fund': 8.0}
    years = 20
    
    # Scenario 1: Lump sum of $240,000 (equivalent to $1,000/month for 20 years)
    lump_sum_amount = 240000
    lump_sum_projections = calculator.calculate_lump_sum(lump_sum_amount, returns, years)
    lump_sum_summary = calculator.generate_investment_summary(lump_sum_projections)
    
    # Scenario 2: SIP of $1,000/month for 20 years
    monthly_sip = 1000
    sip_projections = calculator.calculate_sip(monthly_sip, returns, years)
    sip_summary = calculator.generate_investment_summary(sip_projections)
    
    print(f"Comparison over {years} years with {returns['balanced_fund']}% annual return:")
    print("-" * 60)
    print(f"{'Strategy':<15} {'Investment':<15} {'Final Value':<15} {'CAGR':<10}")
    print("-" * 60)
    print(f"{'Lump Sum':<15} ${lump_sum_amount:>12,} ${lump_sum_summary['final_value']:>12,.0f} {lump_sum_summary['cagr']:>7.2f}%")
    print(f"{'SIP':<15} ${monthly_sip*12*years:>12,} ${sip_summary['final_value']:>12,.0f} {sip_summary['cagr']:>7.2f}%")
    print()
    
    # Analysis
    if lump_sum_summary['final_value'] > sip_summary['final_value']:
        advantage = lump_sum_summary['final_value'] - sip_summary['final_value']
        print(f"Lump Sum advantage: ${advantage:,.0f}")
    else:
        advantage = sip_summary['final_value'] - lump_sum_summary['final_value']
        print(f"SIP advantage: ${advantage:,.0f}")
    
    print("\nNote: Lump sum typically performs better if you have the full amount")
    print("available upfront, but SIP helps with rupee cost averaging and")
    print("disciplined investing when you don't have a large sum initially.")
    print()


def main():
    """Run all investment calculator demos."""
    print("Investment Calculators Demo")
    print("This demo shows different investment calculation strategies\n")
    
    try:
        demo_lump_sum()
        demo_sip()
        demo_swp()
        demo_comparison()
        
        print("=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during demo: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())