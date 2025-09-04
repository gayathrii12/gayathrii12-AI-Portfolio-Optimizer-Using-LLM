"""
Terminal output generator for human-readable financial results.

This module provides formatted terminal output for portfolio allocations,
projections, risk metrics, and explanatory text.
"""

from typing import List, Dict, Any
from models.data_models import (
    PortfolioAllocation, 
    ProjectionResult, 
    RiskMetrics, 
    UserInputModel
)


class TerminalOutputGenerator:
    """Generates human-readable terminal output for financial analysis results."""
    
    def __init__(self):
        """Initialize the terminal output generator."""
        self.width = 80  # Terminal width for formatting
        
    def format_portfolio_allocation(self, allocation: PortfolioAllocation, 
                                  user_input: UserInputModel) -> str:
        """
        Format portfolio allocation as a readable table.
        
        Args:
            allocation: Portfolio allocation percentages
            user_input: User investment parameters
            
        Returns:
            Formatted allocation table string
        """
        lines = []
        lines.append("=" * self.width)
        lines.append("PORTFOLIO ALLOCATION RECOMMENDATION".center(self.width))
        lines.append("=" * self.width)
        lines.append("")
        
        # Risk profile context
        lines.append(f"Risk Profile: {user_input.risk_profile}")
        lines.append(f"Investment Amount: ${user_input.investment_amount:,.2f}")
        lines.append(f"Investment Type: {user_input.investment_type.upper()}")
        lines.append(f"Investment Tenure: {user_input.tenure_years} years")
        lines.append("")
        
        # Allocation table
        lines.append("Asset Class Allocation:")
        lines.append("-" * 50)
        lines.append(f"{'Asset Class':<25} {'Allocation':<12} {'Amount':<12}")
        lines.append("-" * 50)
        
        # Calculate amounts for each asset
        total_amount = user_input.investment_amount
        allocations = [
            ("S&P 500", allocation.sp500, total_amount * allocation.sp500 / 100),
            ("US Small Cap", allocation.small_cap, total_amount * allocation.small_cap / 100),
            ("Bonds (Combined)", allocation.bonds, total_amount * allocation.bonds / 100),
            ("Real Estate", allocation.real_estate, total_amount * allocation.real_estate / 100),
            ("Gold", allocation.gold, total_amount * allocation.gold / 100),
        ]
        
        for asset_name, percentage, amount in allocations:
            if percentage > 0:  # Only show non-zero allocations
                lines.append(f"{asset_name:<25} {percentage:>6.1f}%     ${amount:>11,.2f}")
        
        lines.append("-" * 50)
        total_percentage = sum(alloc[1] for alloc in allocations)
        lines.append(f"{'TOTAL':<25} {total_percentage:>6.1f}%     ${total_amount:>11,.2f}")
        lines.append("")
        
        return "\n".join(lines)
    
    def format_year_by_year_projections(self, projections: List[ProjectionResult],
                                      user_input: UserInputModel) -> str:
        """
        Format year-by-year portfolio growth projections.
        
        Args:
            projections: List of yearly projection results
            user_input: User investment parameters
            
        Returns:
            Formatted projections table string
        """
        lines = []
        lines.append("=" * self.width)
        lines.append("PORTFOLIO GROWTH PROJECTIONS".center(self.width))
        lines.append("=" * self.width)
        lines.append("")
        
        # Calculate overall CAGR
        if len(projections) > 1:
            initial_value = user_input.investment_amount
            final_value = projections[-1].portfolio_value
            years = len(projections)
            cagr = ((final_value / initial_value) ** (1/years) - 1) * 100
            lines.append(f"Overall CAGR: {cagr:.2f}%")
            lines.append("")
        
        # Projections table
        lines.append("Year-by-Year Growth:")
        lines.append("-" * 70)
        lines.append(f"{'Year':<6} {'Portfolio Value':<16} {'Annual Return':<14} {'Cumulative Return':<16}")
        lines.append("-" * 70)
        
        # Starting value
        lines.append(f"{'0':<6} ${user_input.investment_amount:<15,.2f} {'--':<14} {'0.0%':<16}")
        
        for projection in projections:
            lines.append(
                f"{projection.year:<6} "
                f"${projection.portfolio_value:<15,.2f} "
                f"{projection.annual_return:>6.2f}%      "
                f"{projection.cumulative_return:>6.2f}%"
            )
        
        lines.append("-" * 70)
        lines.append("")
        
        return "\n".join(lines)
    
    def format_risk_metrics(self, risk_metrics: RiskMetrics, 
                           benchmark_name: str = "S&P 500") -> str:
        """
        Format risk metrics with benchmark comparisons.
        
        Args:
            risk_metrics: Portfolio risk analysis metrics
            benchmark_name: Name of benchmark for comparison
            
        Returns:
            Formatted risk metrics string
        """
        lines = []
        lines.append("=" * self.width)
        lines.append("RISK ANALYSIS & BENCHMARK COMPARISON".center(self.width))
        lines.append("=" * self.width)
        lines.append("")
        
        lines.append(f"Benchmark: {benchmark_name}")
        lines.append("")
        
        # Risk metrics table
        lines.append("Risk Metrics:")
        lines.append("-" * 40)
        lines.append(f"{'Metric':<20} {'Value':<15} {'Interpretation'}")
        lines.append("-" * 40)
        
        # Alpha interpretation
        alpha_interp = "Outperforming" if risk_metrics.alpha > 0 else "Underperforming"
        if abs(risk_metrics.alpha) < 0.5:
            alpha_interp = "Similar to benchmark"
        
        # Beta interpretation
        if risk_metrics.beta < 0.85:
            beta_interp = "Less volatile"
        elif risk_metrics.beta > 1.2:
            beta_interp = "More volatile"
        else:
            beta_interp = "Similar volatility"
        
        # Sharpe ratio interpretation
        if risk_metrics.sharpe_ratio > 1.0:
            sharpe_interp = "Excellent risk-adj return"
        elif risk_metrics.sharpe_ratio > 0.5:
            sharpe_interp = "Good risk-adj return"
        elif risk_metrics.sharpe_ratio > 0:
            sharpe_interp = "Positive risk-adj return"
        else:
            sharpe_interp = "Poor risk-adj return"
        
        metrics_data = [
            ("Alpha", f"{risk_metrics.alpha:+.2f}%", alpha_interp),
            ("Beta", f"{risk_metrics.beta:.2f}", beta_interp),
            ("Volatility", f"{risk_metrics.volatility:.2f}%", "Annual std deviation"),
            ("Sharpe Ratio", f"{risk_metrics.sharpe_ratio:.2f}", sharpe_interp),
            ("Max Drawdown", f"{risk_metrics.max_drawdown:.2f}%", "Worst decline period"),
        ]
        
        for metric, value, interpretation in metrics_data:
            lines.append(f"{metric:<20} {value:<15} {interpretation}")
        
        lines.append("-" * 40)
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_explanation(self, allocation: PortfolioAllocation,
                           user_input: UserInputModel,
                           risk_metrics: RiskMetrics) -> str:
        """
        Generate human-friendly explanation of the portfolio strategy.
        
        Args:
            allocation: Portfolio allocation percentages
            user_input: User investment parameters
            risk_metrics: Portfolio risk metrics
            
        Returns:
            Human-friendly explanation text
        """
        lines = []
        lines.append("=" * self.width)
        lines.append("PORTFOLIO STRATEGY EXPLANATION".center(self.width))
        lines.append("=" * self.width)
        lines.append("")
        
        # Risk profile explanation
        risk_explanations = {
            "Low": "This conservative allocation prioritizes capital preservation with "
                   "lower volatility. The portfolio emphasizes bonds and stable assets "
                   "to minimize risk while providing steady returns.",
            "Moderate": "This balanced allocation seeks moderate growth while managing "
                       "risk through diversification. The portfolio combines growth "
                       "assets with stable investments for steady long-term returns.",
            "High": "This aggressive allocation targets higher returns through growth "
                    "assets. The portfolio accepts higher volatility in exchange for "
                    "greater long-term wealth building potential."
        }
        
        lines.append("Investment Strategy:")
        lines.append(risk_explanations[user_input.risk_profile])
        lines.append("")
        
        # Allocation strategy explanation
        lines.append("Asset Allocation Rationale:")
        
        # Identify dominant asset classes
        allocations = [
            ("Stocks (S&P 500 + Small Cap)", allocation.sp500 + allocation.small_cap),
            ("Bonds", allocation.bonds),
            ("Real Estate", allocation.real_estate),
            ("Gold", allocation.gold),
        ]
        
        allocations.sort(key=lambda x: x[1], reverse=True)
        
        for asset_class, percentage in allocations:
            if percentage > 10:  # Only explain significant allocations
                if "Stocks" in asset_class:
                    lines.append(f"• {asset_class} ({percentage:.1f}%): Provides growth "
                               f"potential and inflation protection through equity ownership.")
                elif "Bonds" in asset_class:
                    lines.append(f"• {asset_class} ({percentage:.1f}%): Offers stability "
                               f"and regular income with lower volatility than stocks.")
                elif "Real Estate" in asset_class:
                    lines.append(f"• {asset_class} ({percentage:.1f}%): Provides "
                               f"diversification and inflation hedge through property exposure.")
                elif "Gold" in asset_class:
                    lines.append(f"• {asset_class} ({percentage:.1f}%): Acts as a hedge "
                               f"against inflation and market uncertainty.")
        
        lines.append("")
        
        # Risk considerations
        lines.append("Risk Considerations:")
        if risk_metrics.volatility < 10:
            lines.append("• Low volatility portfolio suitable for conservative investors")
        elif risk_metrics.volatility > 20:
            lines.append("• Higher volatility expected - suitable for long-term investors")
        else:
            lines.append("• Moderate volatility balanced with growth potential")
        
        if risk_metrics.max_drawdown < -20:
            lines.append("• Potential for significant short-term declines during market stress")
        elif risk_metrics.max_drawdown > -10:
            lines.append("• Limited downside risk with stable asset allocation")
        
        lines.append("")
        
        # Investment type specific advice
        if user_input.investment_type == "sip":
            lines.append("SIP Investment Benefits:")
            lines.append("• Dollar-cost averaging reduces timing risk")
            lines.append("• Regular investments build discipline and compound growth")
            lines.append("• Market volatility becomes an advantage over time")
        else:
            lines.append("Lumpsum Investment Considerations:")
            lines.append("• Immediate market exposure captures full investment period")
            lines.append("• Consider market timing and current valuations")
            lines.append("• May benefit from rebalancing over the investment period")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_complete_report(self, allocation: PortfolioAllocation,
                               projections: List[ProjectionResult],
                               risk_metrics: RiskMetrics,
                               user_input: UserInputModel) -> str:
        """
        Generate a complete terminal report combining all sections.
        
        Args:
            allocation: Portfolio allocation percentages
            projections: List of yearly projection results
            risk_metrics: Portfolio risk metrics
            user_input: User investment parameters
            
        Returns:
            Complete formatted report string
        """
        report_sections = [
            self.format_portfolio_allocation(allocation, user_input),
            self.format_year_by_year_projections(projections, user_input),
            self.format_risk_metrics(risk_metrics),
            self.generate_explanation(allocation, user_input, risk_metrics)
        ]
        
        # Add disclaimer
        disclaimer = [
            "=" * self.width,
            "IMPORTANT DISCLAIMER".center(self.width),
            "=" * self.width,
            "",
            "This analysis is for educational purposes only and does not constitute",
            "financial advice. Past performance does not guarantee future results.",
            "Please consult with a qualified financial advisor before making",
            "investment decisions. All projections are based on historical data",
            "and mathematical models, not predictions of future market performance.",
            "",
            "=" * self.width
        ]
        
        report_sections.append("\n".join(disclaimer))
        
        return "\n\n".join(report_sections)