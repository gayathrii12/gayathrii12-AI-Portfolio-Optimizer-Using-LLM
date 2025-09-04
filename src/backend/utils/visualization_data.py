"""
Visualization data preparation module for React frontend integration.

This module provides specialized data formatting for various chart types
including pie charts, line charts, and comparison charts. It ensures
data validation and React component compatibility.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator
from models.data_models import PortfolioAllocation, ProjectionResult, RiskMetrics


class PieChartDataPoint(BaseModel):
    """Data point for pie chart visualization."""
    
    name: str = Field(description="Asset class name")
    value: float = Field(ge=0, le=100, description="Allocation percentage")
    color: str = Field(description="Hex color code for chart segment")
    percentage: str = Field(description="Formatted percentage string")
    
    @field_validator('color')
    @classmethod
    def validate_color(cls, v):
        """Validate hex color format."""
        if not v.startswith('#') or len(v) != 7:
            raise ValueError("Color must be in hex format (#RRGGBB)")
        return v
    
    @field_validator('value', mode='before')
    @classmethod
    def round_value(cls, v):
        """Round value to 2 decimal places."""
        return round(float(v), 2)


class LineChartDataPoint(BaseModel):
    """Data point for line chart visualization."""
    
    year: int = Field(ge=0, description="Year in projection timeline")
    portfolio_value: float = Field(ge=0, description="Portfolio value")
    formatted_value: str = Field(description="Formatted currency string")
    annual_return: Optional[float] = Field(default=None, description="Annual return percentage")
    cumulative_return: Optional[float] = Field(default=None, description="Cumulative return percentage")
    
    @field_validator('portfolio_value', 'annual_return', 'cumulative_return', mode='before')
    @classmethod
    def round_values(cls, v):
        """Round financial values to 2 decimal places."""
        if v is None:
            return v
        return round(float(v), 2)


class ComparisonChartDataPoint(BaseModel):
    """Data point for portfolio vs benchmark comparison chart."""
    
    year: int = Field(ge=0, description="Year in projection timeline")
    portfolio_value: float = Field(ge=0, description="Portfolio value")
    benchmark_value: float = Field(ge=0, description="Benchmark value")
    portfolio_return: float = Field(description="Portfolio cumulative return percentage")
    benchmark_return: float = Field(description="Benchmark cumulative return percentage")
    outperformance: float = Field(description="Portfolio outperformance vs benchmark")
    
    @field_validator('portfolio_value', 'benchmark_value', 'portfolio_return', 
                    'benchmark_return', 'outperformance', mode='before')
    @classmethod
    def round_values(cls, v):
        """Round values to 2 decimal places."""
        return round(float(v), 2)


class VisualizationDataPreparator:
    """Prepares data structures for React visualization components."""
    
    # Color palette for consistent chart styling
    ASSET_COLORS = {
        "S&P 500": "#1f77b4",
        "US Small Cap": "#ff7f0e", 
        "Bonds": "#2ca02c",
        "Real Estate": "#d62728",
        "Gold": "#9467bd",
        "Cash": "#8c564b",
        "International": "#e377c2",
        "Commodities": "#7f7f7f"
    }
    
    def __init__(self):
        """Initialize the visualization data preparator."""
        self.default_benchmark_return = 10.5  # S&P 500 historical average
        
    def prepare_pie_chart_data(self, allocation: PortfolioAllocation,
                             include_zero_allocations: bool = False) -> List[PieChartDataPoint]:
        """
        Prepare data for pie chart allocation visualization.
        
        Args:
            allocation: Portfolio allocation percentages
            include_zero_allocations: Whether to include assets with 0% allocation
            
        Returns:
            List[PieChartDataPoint]: Validated pie chart data points
        """
        allocation_items = [
            ("S&P 500", allocation.sp500),
            ("US Small Cap", allocation.small_cap),
            ("Bonds", allocation.bonds),
            ("Real Estate", allocation.real_estate),
            ("Gold", allocation.gold)
        ]
        
        pie_data = []
        for name, percentage in allocation_items:
            if percentage > 0 or include_zero_allocations:
                pie_data.append(PieChartDataPoint(
                    name=name,
                    value=percentage,
                    color=self.ASSET_COLORS.get(name, "#cccccc"),
                    percentage=f"{percentage:.1f}%"
                ))
        
        # Sort by value descending for better visual presentation
        pie_data.sort(key=lambda x: x.value, reverse=True)
        
        return pie_data
    
    def prepare_line_chart_data(self, projections: List[ProjectionResult],
                              initial_investment: float,
                              currency_symbol: str = "$") -> List[LineChartDataPoint]:
        """
        Prepare data for line chart portfolio value visualization.
        
        Args:
            projections: List of yearly projection results
            initial_investment: Initial investment amount
            currency_symbol: Currency symbol for formatting
            
        Returns:
            List[LineChartDataPoint]: Validated line chart data points
        """
        line_data = []
        
        # Add initial year (year 0)
        line_data.append(LineChartDataPoint(
            year=0,
            portfolio_value=initial_investment,
            formatted_value=f"{currency_symbol}{initial_investment:,.2f}",
            annual_return=0.0,
            cumulative_return=0.0
        ))
        
        # Add projection years
        for projection in projections:
            line_data.append(LineChartDataPoint(
                year=projection.year,
                portfolio_value=projection.portfolio_value,
                formatted_value=f"{currency_symbol}{projection.portfolio_value:,.2f}",
                annual_return=projection.annual_return,
                cumulative_return=projection.cumulative_return
            ))
        
        return line_data
    
    def prepare_comparison_chart_data(self, projections: List[ProjectionResult],
                                    initial_investment: float,
                                    benchmark_annual_return: Optional[float] = None) -> List[ComparisonChartDataPoint]:
        """
        Prepare data for portfolio vs benchmark comparison chart.
        
        Args:
            projections: List of yearly projection results
            initial_investment: Initial investment amount
            benchmark_annual_return: Annual return of benchmark (defaults to S&P 500)
            
        Returns:
            List[ComparisonChartDataPoint]: Validated comparison chart data points
        """
        if benchmark_annual_return is None:
            benchmark_annual_return = self.default_benchmark_return
        
        comparison_data = []
        
        # Add initial year (year 0)
        comparison_data.append(ComparisonChartDataPoint(
            year=0,
            portfolio_value=initial_investment,
            benchmark_value=initial_investment,
            portfolio_return=0.0,
            benchmark_return=0.0,
            outperformance=0.0
        ))
        
        # Add projection years
        for projection in projections:
            # Calculate benchmark value
            benchmark_value = initial_investment * ((1 + benchmark_annual_return / 100) ** projection.year)
            benchmark_return = ((benchmark_value / initial_investment - 1) * 100) if initial_investment > 0 else 0
            outperformance = projection.cumulative_return - benchmark_return
            
            comparison_data.append(ComparisonChartDataPoint(
                year=projection.year,
                portfolio_value=projection.portfolio_value,
                benchmark_value=benchmark_value,
                portfolio_return=projection.cumulative_return,
                benchmark_return=benchmark_return,
                outperformance=outperformance
            ))
        
        return comparison_data
    
    def prepare_risk_visualization_data(self, risk_metrics: RiskMetrics,
                                      benchmark_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Prepare data for risk metrics visualization (radar chart or bar chart).
        
        Args:
            risk_metrics: Portfolio risk metrics
            benchmark_metrics: Benchmark risk metrics for comparison
            
        Returns:
            Dict[str, Any]: Risk visualization data structure
        """
        if benchmark_metrics is None:
            benchmark_metrics = {
                "volatility": 16.0,  # S&P 500 historical volatility
                "sharpe_ratio": 0.65,  # S&P 500 historical Sharpe ratio
                "max_drawdown": -37.0,  # S&P 500 worst drawdown
                "beta": 1.0,  # By definition
                "alpha": 0.0   # By definition
            }
        
        risk_data = {
            "portfolio_metrics": [
                {"metric": "Volatility (%)", "value": risk_metrics.volatility, "benchmark": benchmark_metrics.get("volatility", 0)},
                {"metric": "Sharpe Ratio", "value": risk_metrics.sharpe_ratio, "benchmark": benchmark_metrics.get("sharpe_ratio", 0)},
                {"metric": "Max Drawdown (%)", "value": abs(risk_metrics.max_drawdown), "benchmark": abs(benchmark_metrics.get("max_drawdown", 0))},
                {"metric": "Beta", "value": risk_metrics.beta, "benchmark": benchmark_metrics.get("beta", 1)},
                {"metric": "Alpha (%)", "value": risk_metrics.alpha, "benchmark": benchmark_metrics.get("alpha", 0)}
            ],
            "risk_score": self._calculate_risk_score(risk_metrics),
            "risk_level": self._determine_risk_level(risk_metrics)
        }
        
        return risk_data
    
    def prepare_allocation_trend_data(self, allocations_over_time: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare data for allocation changes over time (stacked area chart).
        
        Args:
            allocations_over_time: List of allocation dictionaries with year and percentages
            
        Returns:
            List[Dict[str, Any]]: Allocation trend data for stacked chart
        """
        trend_data = []
        
        for allocation_data in allocations_over_time:
            year = allocation_data.get("year", 0)
            
            # Ensure all asset classes are present
            trend_point = {
                "year": year,
                "sp500": allocation_data.get("sp500", 0),
                "small_cap": allocation_data.get("small_cap", 0),
                "t_bills": allocation_data.get("t_bills", 0),
                "t_bonds": allocation_data.get("t_bonds", 0),
                "corporate_bonds": allocation_data.get("corporate_bonds", 0),
                "real_estate": allocation_data.get("real_estate", 0),
                "gold": allocation_data.get("gold", 0)
            }
            
            # Validate total allocation
            total = sum([v for k, v in trend_point.items() if k != "year"])
            if abs(total - 100) > 0.01:
                # Normalize to 100% if needed
                factor = 100 / total if total > 0 else 1
                for key in trend_point:
                    if key != "year":
                        trend_point[key] *= factor
            
            trend_data.append(trend_point)
        
        return trend_data
    
    def prepare_rebalancing_comparison_data(self, with_rebalancing: List[Dict[str, Any]],
                                          without_rebalancing: List[Dict[str, Any]],
                                          currency_symbol: str = "$") -> List[Dict[str, Any]]:
        """
        Prepare data for comparing portfolio performance with and without rebalancing.
        
        Args:
            with_rebalancing: Portfolio projections with rebalancing
            without_rebalancing: Portfolio projections without rebalancing
            currency_symbol: Currency symbol for formatting
            
        Returns:
            List[Dict[str, Any]]: Comparison data for line chart
        """
        comparison_data = []
        
        # Ensure both datasets have the same length
        min_length = min(len(with_rebalancing), len(without_rebalancing))
        
        for i in range(min_length):
            with_data = with_rebalancing[i]
            without_data = without_rebalancing[i]
            
            year = with_data.get('year', i)
            with_value = with_data.get('end_period_value', with_data.get('portfolio_value', 0))
            without_value = without_data.get('portfolio_value', 0)
            
            # Calculate outperformance
            outperformance = ((with_value - without_value) / without_value * 100) if without_value > 0 else 0
            
            comparison_data.append({
                'year': year,
                'with_rebalancing': with_value,
                'without_rebalancing': without_value,
                'with_rebalancing_formatted': f"{currency_symbol}{with_value:,.2f}",
                'without_rebalancing_formatted': f"{currency_symbol}{without_value:,.2f}",
                'outperformance': round(outperformance, 2),
                'outperformance_formatted': f"{outperformance:+.2f}%"
            })
        
        return comparison_data
    
    def prepare_rebalancing_events_data(self, rebalancing_schedule: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare data for visualizing rebalancing events and their impact.
        
        Args:
            rebalancing_schedule: List of rebalancing events
            
        Returns:
            List[Dict[str, Any]]: Rebalancing events data for timeline visualization
        """
        events_data = []
        
        for i, event in enumerate(rebalancing_schedule):
            if i == 0:  # Skip initial allocation
                continue
                
            year = event.get('year', 0)
            changes = event.get('changes', {})
            rationale = event.get('rationale', '')
            
            # Calculate major changes
            equity_change = sum(changes.get(asset, 0) for asset in ['sp500', 'small_cap'])
            bond_change = sum(changes.get(asset, 0) for asset in ['t_bills', 't_bonds', 'corporate_bonds'])
            
            # Find the most significant change
            max_change_asset = max(changes.items(), key=lambda x: abs(x[1])) if changes else ('', 0)
            
            events_data.append({
                'year': year,
                'equity_change': round(equity_change, 2),
                'bond_change': round(bond_change, 2),
                'max_change_asset': max_change_asset[0],
                'max_change_value': round(max_change_asset[1], 2),
                'rationale': rationale,
                'total_changes': len([c for c in changes.values() if abs(c) > 0.1])
            })
        
        return events_data
    
    def validate_chart_data(self, chart_data: Union[List[Dict[str, Any]], Dict[str, Any]],
                          chart_type: str) -> bool:
        """
        Validate chart data structure for React component consumption.
        
        Args:
            chart_data: Chart data to validate
            chart_type: Type of chart ('pie', 'line', 'comparison', 'risk')
            
        Returns:
            bool: True if data is valid for React consumption
        """
        try:
            if chart_type == "pie":
                if not isinstance(chart_data, list):
                    return False
                for point in chart_data:
                    PieChartDataPoint(**point)
                    
            elif chart_type == "line":
                if not isinstance(chart_data, list):
                    return False
                for point in chart_data:
                    LineChartDataPoint(**point)
                    
            elif chart_type == "comparison":
                if not isinstance(chart_data, list):
                    return False
                for point in chart_data:
                    ComparisonChartDataPoint(**point)
                    
            elif chart_type == "risk":
                if not isinstance(chart_data, dict):
                    return False
                # Basic structure validation for risk data
                required_keys = ["portfolio_metrics", "risk_score", "risk_level"]
                if not all(key in chart_data for key in required_keys):
                    return False
                    
            return True
            
        except Exception:
            return False
    
    def _calculate_risk_score(self, risk_metrics: RiskMetrics) -> float:
        """
        Calculate a composite risk score (0-100 scale).
        
        Args:
            risk_metrics: Portfolio risk metrics
            
        Returns:
            float: Risk score between 0 (low risk) and 100 (high risk)
        """
        # Normalize metrics to 0-100 scale and weight them
        volatility_score = min(risk_metrics.volatility * 2.5, 100)  # 40% vol = 100 points
        drawdown_score = min(abs(risk_metrics.max_drawdown) * 2, 100)  # 50% drawdown = 100 points
        beta_score = min(risk_metrics.beta * 50, 100)  # Beta of 2 = 100 points
        
        # Sharpe ratio contributes negatively (higher is better)
        sharpe_score = max(0, 50 - risk_metrics.sharpe_ratio * 25)  # Sharpe of 2 = 0 points
        
        # Weighted average
        risk_score = (volatility_score * 0.3 + drawdown_score * 0.3 + 
                     beta_score * 0.2 + sharpe_score * 0.2)
        
        return round(risk_score, 1)
    
    def _determine_risk_level(self, risk_metrics: RiskMetrics) -> str:
        """
        Determine risk level category based on metrics.
        
        Args:
            risk_metrics: Portfolio risk metrics
            
        Returns:
            str: Risk level ('Low', 'Moderate', 'High')
        """
        risk_score = self._calculate_risk_score(risk_metrics)
        
        if risk_score < 30:
            return "Low"
        elif risk_score < 60:
            return "Moderate"
        else:
            return "High"