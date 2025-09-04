"""
Unit tests for JSON output generator.

Tests JSON schema validation, data formatting, and React frontend compatibility.
"""

import pytest
import json
from datetime import datetime
from typing import Dict, Any

from utils.json_output import (
    JSONOutputGenerator,
    AllocationJSON,
    ProjectionJSON,
    BenchmarkJSON,
    RiskMetricsJSON,
    VisualizationDataJSON,
    CompleteOutputJSON
)
from models.data_models import (
    PortfolioAllocation,
    ProjectionResult,
    RiskMetrics,
    UserInputModel
)


# Global fixtures for all test classes
@pytest.fixture
def json_generator():
    """Create JSONOutputGenerator instance for testing."""
    return JSONOutputGenerator()

@pytest.fixture
def sample_user_input():
    """Create sample user input for testing."""
    return UserInputModel(
        investment_amount=100000.0,
        investment_type="lumpsum",
        tenure_years=10,
        risk_profile="Moderate",
        return_expectation=12.0
    )

@pytest.fixture
def sample_allocation():
    """Create sample portfolio allocation for testing."""
    return PortfolioAllocation(
        sp500=45.0,
        small_cap=15.0,
        bonds=25.0,
        real_estate=10.0,
        gold=5.0
    )

@pytest.fixture
def sample_projections():
    """Create sample projection results for testing."""
    return [
        ProjectionResult(year=1, portfolio_value=110000.0, annual_return=10.0, cumulative_return=10.0),
        ProjectionResult(year=2, portfolio_value=121000.0, annual_return=10.0, cumulative_return=21.0),
        ProjectionResult(year=3, portfolio_value=133100.0, annual_return=10.0, cumulative_return=33.1)
    ]

@pytest.fixture
def sample_risk_metrics():
    """Create sample risk metrics for testing."""
    return RiskMetrics(
        alpha=2.5,
        beta=0.95,
        volatility=15.2,
        sharpe_ratio=0.85,
        max_drawdown=-12.5
    )


class TestJSONOutputGenerator:
    """Test cases for JSONOutputGenerator class."""


class TestAllocationJSON:
    """Test cases for AllocationJSON schema."""
    
    def test_allocation_json_creation(self):
        """Test creation of AllocationJSON with valid data."""
        allocation_json = AllocationJSON(
            sp500=45.0,
            small_cap=15.0,
            bonds=25.0,
            real_estate=10.0,
            gold=5.0
        )
        
        assert allocation_json.sp500 == 45.0
        assert allocation_json.small_cap == 15.0
        assert allocation_json.bonds == 25.0
        assert allocation_json.real_estate == 10.0
        assert allocation_json.gold == 5.0
    
    def test_allocation_json_rounding(self):
        """Test that allocation percentages are rounded to 2 decimal places."""
        allocation_json = AllocationJSON(
            sp500=45.123456,
            small_cap=15.987654,
            bonds=25.555555,
            real_estate=10.111111,
            gold=5.222222
        )
        
        assert allocation_json.sp500 == 45.12
        assert allocation_json.small_cap == 15.99
        assert allocation_json.bonds == 25.56
        assert allocation_json.real_estate == 10.11
        assert allocation_json.gold == 5.22
    
    def test_allocation_json_validation_errors(self):
        """Test validation errors for invalid allocation values."""
        with pytest.raises(ValueError):
            AllocationJSON(
                sp500=-5.0,  # Negative value should fail
                small_cap=15.0,
                bonds=25.0,
                real_estate=10.0,
                gold=5.0
            )
        
        with pytest.raises(ValueError):
            AllocationJSON(
                sp500=105.0,  # Value > 100 should fail
                small_cap=15.0,
                bonds=25.0,
                real_estate=10.0,
                gold=5.0
            )


class TestProjectionJSON:
    """Test cases for ProjectionJSON schema."""
    
    def test_projection_json_creation(self):
        """Test creation of ProjectionJSON with valid data."""
        projection_json = ProjectionJSON(
            year=1,
            portfolio_value=110000.0,
            annual_return=10.0,
            cumulative_return=10.0
        )
        
        assert projection_json.year == 1
        assert projection_json.portfolio_value == 110000.0
        assert projection_json.annual_return == 10.0
        assert projection_json.cumulative_return == 10.0
    
    def test_projection_json_rounding(self):
        """Test that projection values are rounded to 2 decimal places."""
        projection_json = ProjectionJSON(
            year=1,
            portfolio_value=110000.123456,
            annual_return=10.987654,
            cumulative_return=10.555555
        )
        
        assert projection_json.portfolio_value == 110000.12
        assert projection_json.annual_return == 10.99
        assert projection_json.cumulative_return == 10.56
    
    def test_projection_json_validation_errors(self):
        """Test validation errors for invalid projection values."""
        with pytest.raises(ValueError):
            ProjectionJSON(
                year=-1,  # Negative year should fail
                portfolio_value=110000.0,
                annual_return=10.0,
                cumulative_return=10.0
            )
        
        with pytest.raises(ValueError):
            ProjectionJSON(
                year=1,
                portfolio_value=-1000.0,  # Negative portfolio value should fail
                annual_return=10.0,
                cumulative_return=10.0
            )


class TestRiskMetricsJSON:
    """Test cases for RiskMetricsJSON schema."""
    
    def test_risk_metrics_json_creation(self):
        """Test creation of RiskMetricsJSON with valid data."""
        risk_json = RiskMetricsJSON(
            alpha=2.5,
            beta=0.95,
            volatility=15.2,
            sharpe_ratio=0.85,
            max_drawdown=-12.5
        )
        
        assert risk_json.alpha == 2.5
        assert risk_json.beta == 0.95
        assert risk_json.volatility == 15.2
        assert risk_json.sharpe_ratio == 0.85
        assert risk_json.max_drawdown == -12.5
    
    def test_risk_metrics_json_rounding(self):
        """Test that risk metrics are rounded to 3 decimal places."""
        risk_json = RiskMetricsJSON(
            alpha=2.123456789,
            beta=0.987654321,
            volatility=15.555555555,
            sharpe_ratio=0.888888888,
            max_drawdown=-12.777777777
        )
        
        assert risk_json.alpha == 2.123
        assert risk_json.beta == 0.988
        assert risk_json.volatility == 15.556
        assert risk_json.sharpe_ratio == 0.889
        assert risk_json.max_drawdown == -12.778
    
    def test_risk_metrics_json_validation_errors(self):
        """Test validation errors for invalid risk metrics values."""
        with pytest.raises(ValueError):
            RiskMetricsJSON(
                alpha=2.5,
                beta=-0.5,  # Negative beta should fail
                volatility=15.2,
                sharpe_ratio=0.85,
                max_drawdown=-12.5
            )
        
        with pytest.raises(ValueError):
            RiskMetricsJSON(
                alpha=2.5,
                beta=0.95,
                volatility=-5.0,  # Negative volatility should fail
                sharpe_ratio=0.85,
                max_drawdown=-12.5
            )
        
        with pytest.raises(ValueError):
            RiskMetricsJSON(
                alpha=2.5,
                beta=0.95,
                volatility=15.2,
                sharpe_ratio=0.85,
                max_drawdown=5.0  # Positive max drawdown should fail
            )


class TestJSONOutputGeneratorMethods:
    """Test cases for JSONOutputGenerator methods."""
    
    def test_generate_allocation_json(self, json_generator, sample_allocation):
        """Test generation of allocation JSON structure."""
        allocation_json = json_generator.generate_allocation_json(sample_allocation)
        
        assert isinstance(allocation_json, AllocationJSON)
        assert allocation_json.sp500 == 45.0
        assert allocation_json.small_cap == 15.0
        assert allocation_json.bonds == 25.0
        assert allocation_json.real_estate == 10.0
        assert allocation_json.gold == 5.0
    
    def test_generate_projections_json(self, json_generator, sample_projections, sample_user_input):
        """Test generation of projections JSON array."""
        projections_json = json_generator.generate_projections_json(sample_projections, sample_user_input)
        
        assert isinstance(projections_json, list)
        assert len(projections_json) == 4  # Initial year + 3 projection years
        
        # Check initial year (year 0)
        assert projections_json[0].year == 0
        assert projections_json[0].portfolio_value == 100000.0
        assert projections_json[0].annual_return == 0.0
        assert projections_json[0].cumulative_return == 0.0
        
        # Check projection years
        for i, projection in enumerate(projections_json[1:], 1):
            assert projection.year == i
            assert projection.portfolio_value > 0
            assert isinstance(projection.annual_return, float)
            assert isinstance(projection.cumulative_return, float)
    
    def test_generate_benchmark_json(self, json_generator, sample_projections):
        """Test generation of benchmark JSON structure."""
        benchmark_json = json_generator.generate_benchmark_json(sample_projections)
        
        assert isinstance(benchmark_json, BenchmarkJSON)
        assert benchmark_json.name == "S&P 500"
        assert benchmark_json.annual_return == 10.5
        assert benchmark_json.volatility == 16.0
        assert benchmark_json.cumulative_return > 0
    
    def test_generate_risk_metrics_json(self, json_generator, sample_risk_metrics):
        """Test generation of risk metrics JSON structure."""
        risk_json = json_generator.generate_risk_metrics_json(sample_risk_metrics)
        
        assert isinstance(risk_json, RiskMetricsJSON)
        assert risk_json.alpha == 2.5
        assert risk_json.beta == 0.95
        assert risk_json.volatility == 15.2
        assert risk_json.sharpe_ratio == 0.85
        assert risk_json.max_drawdown == -12.5
    
    def test_generate_visualization_data(self, json_generator, sample_allocation):
        """Test generation of visualization data structure."""
        # Create sample projections for visualization
        projections_json = [
            ProjectionJSON(year=0, portfolio_value=100000.0, annual_return=0.0, cumulative_return=0.0),
            ProjectionJSON(year=1, portfolio_value=110000.0, annual_return=10.0, cumulative_return=10.0)
        ]
        
        benchmark_json = BenchmarkJSON(
            name="S&P 500",
            annual_return=10.5,
            cumulative_return=10.5,
            volatility=16.0
        )
        
        viz_data = json_generator.generate_visualization_data(
            sample_allocation, projections_json, benchmark_json
        )
        
        assert isinstance(viz_data, VisualizationDataJSON)
        
        # Test pie chart data
        assert len(viz_data.pie_chart_data) == 5  # All asset classes have non-zero allocation
        for item in viz_data.pie_chart_data:
            assert "name" in item
            assert "value" in item
            assert "color" in item
            assert "percentage" in item
        
        # Test line chart data
        assert len(viz_data.line_chart_data) == 2  # Year 0 and Year 1
        for item in viz_data.line_chart_data:
            assert "year" in item
            assert "portfolio_value" in item
            assert "formatted_value" in item
        
        # Test comparison chart data
        assert len(viz_data.comparison_chart_data) == 2  # Year 0 and Year 1
        for item in viz_data.comparison_chart_data:
            assert "year" in item
            assert "portfolio_value" in item
            assert "benchmark_value" in item
            assert "portfolio_return" in item
            assert "benchmark_return" in item


class TestCompleteJSONOutput:
    """Test cases for complete JSON output generation."""
    
    def test_generate_complete_json(self, json_generator, sample_allocation, 
                                  sample_projections, sample_risk_metrics, sample_user_input):
        """Test generation of complete JSON output."""
        complete_json = json_generator.generate_complete_json(
            sample_allocation, sample_projections, sample_risk_metrics, sample_user_input
        )
        
        assert isinstance(complete_json, CompleteOutputJSON)
        assert isinstance(complete_json.allocation, AllocationJSON)
        assert isinstance(complete_json.projections, list)
        assert isinstance(complete_json.benchmark, BenchmarkJSON)
        assert isinstance(complete_json.risk_metrics, RiskMetricsJSON)
        assert isinstance(complete_json.visualization_data, VisualizationDataJSON)
        assert isinstance(complete_json.metadata, dict)
        assert isinstance(complete_json.generated_at, datetime)
        
        # Test metadata content
        metadata = complete_json.metadata
        assert metadata["schema_version"] == "1.0.0"
        assert "user_input" in metadata
        assert metadata["user_input"]["investment_amount"] == 100000.0
        assert metadata["user_input"]["risk_profile"] == "Moderate"
        assert "disclaimer" in metadata
    
    def test_export_to_json_string(self, json_generator, sample_allocation, 
                                 sample_projections, sample_risk_metrics, sample_user_input):
        """Test export to JSON string format."""
        complete_json = json_generator.generate_complete_json(
            sample_allocation, sample_projections, sample_risk_metrics, sample_user_input
        )
        
        json_string = json_generator.export_to_json_string(complete_json)
        
        assert isinstance(json_string, str)
        assert len(json_string) > 0
        
        # Verify it's valid JSON
        parsed_json = json.loads(json_string)
        assert isinstance(parsed_json, dict)
        assert "allocation" in parsed_json
        assert "projections" in parsed_json
        assert "risk_metrics" in parsed_json
    
    def test_export_to_json_dict(self, json_generator, sample_allocation, 
                                sample_projections, sample_risk_metrics, sample_user_input):
        """Test export to dictionary format."""
        complete_json = json_generator.generate_complete_json(
            sample_allocation, sample_projections, sample_risk_metrics, sample_user_input
        )
        
        json_dict = json_generator.export_to_json_dict(complete_json)
        
        assert isinstance(json_dict, dict)
        assert "allocation" in json_dict
        assert "projections" in json_dict
        assert "benchmark" in json_dict
        assert "risk_metrics" in json_dict
        assert "visualization_data" in json_dict
        assert "metadata" in json_dict
    
    def test_validate_json_schema(self, json_generator, sample_allocation, 
                                sample_projections, sample_risk_metrics, sample_user_input):
        """Test JSON schema validation."""
        complete_json = json_generator.generate_complete_json(
            sample_allocation, sample_projections, sample_risk_metrics, sample_user_input
        )
        
        json_dict = json_generator.export_to_json_dict(complete_json)
        
        # Valid JSON should pass validation
        assert json_generator.validate_json_schema(json_dict) is True
        
        # Invalid JSON should fail validation
        invalid_json = {"invalid": "structure"}
        assert json_generator.validate_json_schema(invalid_json) is False
    
    def test_get_json_schema(self, json_generator):
        """Test retrieval of JSON schema definition."""
        schema = json_generator.get_json_schema()
        
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "allocation" in schema["properties"]
        assert "projections" in schema["properties"]
        assert "risk_metrics" in schema["properties"]


class TestJSONOutputIntegration:
    """Integration tests for JSON output with React requirements."""
    
    def test_react_compatibility(self, json_generator, sample_allocation, 
                                sample_projections, sample_risk_metrics, sample_user_input):
        """Test that JSON output is compatible with React frontend requirements."""
        complete_json = json_generator.generate_complete_json(
            sample_allocation, sample_projections, sample_risk_metrics, sample_user_input
        )
        
        json_dict = json_generator.export_to_json_dict(complete_json)
        
        # Test allocation structure for React components
        allocation = json_dict["allocation"]
        assert all(isinstance(v, (int, float)) for v in allocation.values())
        assert all(0 <= v <= 100 for v in allocation.values())
        
        # Test projections array structure
        projections = json_dict["projections"]
        assert isinstance(projections, list)
        assert len(projections) > 0
        for projection in projections:
            assert "year" in projection
            assert "portfolio_value" in projection
            assert "annual_return" in projection
            assert "cumulative_return" in projection
        
        # Test visualization data for React charts
        viz_data = json_dict["visualization_data"]
        assert "pie_chart_data" in viz_data
        assert "line_chart_data" in viz_data
        assert "comparison_chart_data" in viz_data
        
        # Pie chart data should be ready for React pie chart component
        pie_data = viz_data["pie_chart_data"]
        for item in pie_data:
            assert "name" in item
            assert "value" in item
            assert "color" in item
            assert isinstance(item["value"], (int, float))
    
    def test_json_serialization_compatibility(self, json_generator, sample_allocation, 
                                            sample_projections, sample_risk_metrics, sample_user_input):
        """Test that JSON output can be serialized and deserialized properly."""
        complete_json = json_generator.generate_complete_json(
            sample_allocation, sample_projections, sample_risk_metrics, sample_user_input
        )
        
        # Test serialization to string and back
        json_string = json_generator.export_to_json_string(complete_json)
        parsed_back = json.loads(json_string)
        
        # Verify structure is preserved
        assert "allocation" in parsed_back
        assert "projections" in parsed_back
        assert "risk_metrics" in parsed_back
        
        # Verify data types are preserved
        assert isinstance(parsed_back["allocation"]["sp500"], (int, float))
        assert isinstance(parsed_back["projections"], list)
        assert isinstance(parsed_back["risk_metrics"]["alpha"], (int, float))


if __name__ == "__main__":
    pytest.main([__file__])