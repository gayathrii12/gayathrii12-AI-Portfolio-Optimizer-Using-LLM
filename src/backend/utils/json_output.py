"""
JSON output generator for React frontend integration.

This module provides structured JSON output that conforms to React component
requirements for portfolio allocations, projections, risk metrics, and
visualization data.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from models.data_models import (
    PortfolioAllocation, 
    ProjectionResult, 
    RiskMetrics, 
    UserInputModel
)
from utils.visualization_data import VisualizationDataPreparator


class AllocationJSON(BaseModel):
    """JSON schema for portfolio allocation data."""
    
    sp500: float = Field(ge=0, le=100, description="S&P 500 allocation percentage")
    small_cap: float = Field(ge=0, le=100, description="US Small Cap allocation percentage")
    bonds: float = Field(ge=0, le=100, description="Combined bonds allocation percentage")
    real_estate: float = Field(ge=0, le=100, description="Real Estate allocation percentage")
    gold: float = Field(ge=0, le=100, description="Gold allocation percentage")
    
    @field_validator('sp500', 'small_cap', 'bonds', 'real_estate', 'gold', mode='before')
    @classmethod
    def round_percentages(cls, v):
        """Round percentages to 2 decimal places for JSON output."""
        return round(float(v), 2)


class ProjectionJSON(BaseModel):
    """JSON schema for yearly projection data."""
    
    year: int = Field(ge=0, description="Year number in projection timeline")
    portfolio_value: float = Field(ge=0, description="Portfolio value at end of year")
    annual_return: float = Field(description="Annual return percentage")
    cumulative_return: float = Field(description="Cumulative return percentage")
    
    @field_validator('portfolio_value', 'annual_return', 'cumulative_return', mode='before')
    @classmethod
    def round_values(cls, v):
        """Round financial values to 2 decimal places."""
        return round(float(v), 2)


class BenchmarkJSON(BaseModel):
    """JSON schema for benchmark comparison data."""
    
    name: str = Field(description="Benchmark name")
    annual_return: float = Field(description="Benchmark annual return percentage")
    cumulative_return: float = Field(description="Benchmark cumulative return percentage")
    volatility: float = Field(ge=0, description="Benchmark volatility percentage")
    
    @field_validator('annual_return', 'cumulative_return', 'volatility', mode='before')
    @classmethod
    def round_values(cls, v):
        """Round benchmark values to 2 decimal places."""
        return round(float(v), 2)


class RiskMetricsJSON(BaseModel):
    """JSON schema for risk metrics data."""
    
    alpha: float = Field(description="Alpha relative to benchmark")
    beta: float = Field(ge=0, description="Beta relative to benchmark")
    volatility: float = Field(ge=0, description="Portfolio volatility percentage")
    sharpe_ratio: float = Field(description="Sharpe ratio")
    max_drawdown: float = Field(le=0, description="Maximum drawdown percentage")
    
    @field_validator('alpha', 'beta', 'volatility', 'sharpe_ratio', 'max_drawdown', mode='before')
    @classmethod
    def round_values(cls, v):
        """Round risk metrics to 3 decimal places for precision."""
        return round(float(v), 3)


class VisualizationDataJSON(BaseModel):
    """JSON schema for React visualization components."""
    
    pie_chart_data: List[Dict[str, Any]] = Field(
        description="Data structure for pie chart allocation visualization"
    )
    line_chart_data: List[Dict[str, Any]] = Field(
        description="Data structure for portfolio value line chart"
    )
    comparison_chart_data: List[Dict[str, Any]] = Field(
        description="Data structure for portfolio vs benchmark comparison"
    )


class CompleteOutputJSON(BaseModel):
    """Complete JSON output schema for React frontend."""
    
    allocation: AllocationJSON = Field(description="Portfolio allocation percentages")
    projections: List[ProjectionJSON] = Field(description="Year-by-year projections")
    benchmark: BenchmarkJSON = Field(description="Benchmark comparison data")
    risk_metrics: RiskMetricsJSON = Field(description="Portfolio risk analysis")
    visualization_data: VisualizationDataJSON = Field(description="Chart visualization data")
    metadata: Dict[str, Any] = Field(description="Additional metadata")
    generated_at: datetime = Field(default_factory=datetime.now, description="Generation timestamp")


class JSONOutputGenerator:
    """Generates React-compatible JSON output for financial analysis results."""
    
    def __init__(self):
        """Initialize the JSON output generator."""
        self.schema_version = "1.0.0"
        self.viz_preparator = VisualizationDataPreparator()
        
    def generate_allocation_json(self, allocation: PortfolioAllocation) -> AllocationJSON:
        """
        Generate JSON structure for portfolio allocation.
        
        Args:
            allocation: Portfolio allocation percentages
            
        Returns:
            AllocationJSON: Validated allocation JSON structure
        """
        return AllocationJSON(
            sp500=allocation.sp500,
            small_cap=allocation.small_cap,
            bonds=allocation.bonds,
            real_estate=allocation.real_estate,
            gold=allocation.gold
        )
    
    def generate_projections_json(self, projections: List[ProjectionResult],
                                user_input: UserInputModel) -> List[ProjectionJSON]:
        """
        Generate JSON array for year-by-year projections.
        
        Args:
            projections: List of yearly projection results
            user_input: User investment parameters for initial value
            
        Returns:
            List[ProjectionJSON]: Validated projections array
        """
        projection_list = []
        
        # Add initial year (year 0) with starting investment
        projection_list.append(ProjectionJSON(
            year=0,
            portfolio_value=user_input.investment_amount,
            annual_return=0.0,
            cumulative_return=0.0
        ))
        
        # Add all projection years
        for projection in projections:
            projection_list.append(ProjectionJSON(
                year=projection.year,
                portfolio_value=projection.portfolio_value,
                annual_return=projection.annual_return,
                cumulative_return=projection.cumulative_return
            ))
        
        return projection_list
    
    def generate_benchmark_json(self, projections: List[ProjectionResult],
                              benchmark_name: str = "S&P 500") -> BenchmarkJSON:
        """
        Generate JSON structure for benchmark comparison.
        
        Args:
            projections: Portfolio projections for comparison
            benchmark_name: Name of the benchmark
            
        Returns:
            BenchmarkJSON: Validated benchmark JSON structure
        """
        # Calculate benchmark metrics (simplified - using S&P 500 historical average)
        # In a real implementation, this would use actual benchmark data
        sp500_annual_return = 10.5  # Historical S&P 500 average
        sp500_volatility = 16.0     # Historical S&P 500 volatility
        
        # Calculate cumulative return for the projection period
        years = len(projections)
        cumulative_return = ((1 + sp500_annual_return / 100) ** years - 1) * 100
        
        return BenchmarkJSON(
            name=benchmark_name,
            annual_return=sp500_annual_return,
            cumulative_return=cumulative_return,
            volatility=sp500_volatility
        )
    
    def generate_risk_metrics_json(self, risk_metrics: RiskMetrics) -> RiskMetricsJSON:
        """
        Generate JSON structure for risk metrics.
        
        Args:
            risk_metrics: Portfolio risk analysis metrics
            
        Returns:
            RiskMetricsJSON: Validated risk metrics JSON structure
        """
        return RiskMetricsJSON(
            alpha=risk_metrics.alpha,
            beta=risk_metrics.beta,
            volatility=risk_metrics.volatility,
            sharpe_ratio=risk_metrics.sharpe_ratio,
            max_drawdown=risk_metrics.max_drawdown
        )
    
    def generate_visualization_data(self, allocation: PortfolioAllocation,
                                  projections: List[ProjectionJSON],
                                  benchmark: BenchmarkJSON,
                                  risk_metrics: Optional[RiskMetrics] = None) -> VisualizationDataJSON:
        """
        Generate data structures for React visualization components.
        
        Args:
            allocation: Portfolio allocation percentages
            projections: Year-by-year projections
            benchmark: Benchmark comparison data
            risk_metrics: Optional risk metrics for enhanced visualization
            
        Returns:
            VisualizationDataJSON: Validated visualization data structure
        """
        # Use the visualization data preparator for consistent formatting
        pie_chart_points = self.viz_preparator.prepare_pie_chart_data(allocation)
        pie_chart_data = [point.model_dump() for point in pie_chart_points]
        
        # Convert projections to ProjectionResult format for the preparator
        initial_value = projections[0].portfolio_value if projections else 0
        projection_results = []
        for proj in projections[1:]:  # Skip year 0
            projection_results.append(ProjectionResult(
                year=proj.year,
                portfolio_value=proj.portfolio_value,
                annual_return=proj.annual_return or 0,
                cumulative_return=proj.cumulative_return or 0
            ))
        
        # Line chart data
        line_chart_points = self.viz_preparator.prepare_line_chart_data(
            projection_results, initial_value
        )
        line_chart_data = [point.model_dump() for point in line_chart_points]
        
        # Comparison chart data
        comparison_chart_points = self.viz_preparator.prepare_comparison_chart_data(
            projection_results, initial_value, benchmark.annual_return
        )
        comparison_chart_data = [point.model_dump() for point in comparison_chart_points]
        
        return VisualizationDataJSON(
            pie_chart_data=pie_chart_data,
            line_chart_data=line_chart_data,
            comparison_chart_data=comparison_chart_data
        )
    
    def generate_complete_json(self, allocation: PortfolioAllocation,
                             projections: List[ProjectionResult],
                             risk_metrics: RiskMetrics,
                             user_input: UserInputModel,
                             benchmark_name: str = "S&P 500") -> CompleteOutputJSON:
        """
        Generate complete JSON output for React frontend.
        
        Args:
            allocation: Portfolio allocation percentages
            projections: List of yearly projection results
            risk_metrics: Portfolio risk metrics
            user_input: User investment parameters
            benchmark_name: Name of benchmark for comparison
            
        Returns:
            CompleteOutputJSON: Complete validated JSON structure
        """
        # Generate individual components
        allocation_json = self.generate_allocation_json(allocation)
        projections_json = self.generate_projections_json(projections, user_input)
        benchmark_json = self.generate_benchmark_json(projections, benchmark_name)
        risk_metrics_json = self.generate_risk_metrics_json(risk_metrics)
        visualization_data = self.generate_visualization_data(
            allocation, projections_json, benchmark_json, risk_metrics
        )
        
        # Generate metadata
        metadata = {
            "schema_version": self.schema_version,
            "user_input": {
                "investment_amount": user_input.investment_amount,
                "investment_type": user_input.investment_type,
                "tenure_years": user_input.tenure_years,
                "risk_profile": user_input.risk_profile,
                "return_expectation": user_input.return_expectation
            },
            "calculation_method": "historical_data_analysis",
            "data_source": "histretSP.xls",
            "disclaimer": "This analysis is for educational purposes only and does not constitute financial advice."
        }
        
        return CompleteOutputJSON(
            allocation=allocation_json,
            projections=projections_json,
            benchmark=benchmark_json,
            risk_metrics=risk_metrics_json,
            visualization_data=visualization_data,
            metadata=metadata
        )
    
    def export_to_json_string(self, complete_output: CompleteOutputJSON,
                            indent: Optional[int] = 2) -> str:
        """
        Export complete output to JSON string.
        
        Args:
            complete_output: Complete JSON output structure
            indent: JSON indentation level (None for compact)
            
        Returns:
            str: JSON string representation
        """
        return complete_output.model_dump_json(indent=indent, exclude_none=True)
    
    def export_to_json_dict(self, complete_output: CompleteOutputJSON) -> Dict[str, Any]:
        """
        Export complete output to Python dictionary.
        
        Args:
            complete_output: Complete JSON output structure
            
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return complete_output.model_dump(exclude_none=True)
    
    def validate_json_schema(self, json_data: Dict[str, Any]) -> bool:
        """
        Validate JSON data against the expected schema.
        
        Args:
            json_data: JSON data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            CompleteOutputJSON(**json_data)
            return True
        except Exception:
            return False
    
    def get_json_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema definition for React frontend integration.
        
        Returns:
            Dict[str, Any]: JSON schema definition
        """
        return CompleteOutputJSON.model_json_schema()