"""
Unit tests for visualization data preparation module.

Tests data formatting for pie charts, line charts, comparison charts,
and data validation for React component consumption.
"""

import pytest
from typing import List, Dict, Any
from models.data_models import PortfolioAllocation, ProjectionResult, RiskMetrics
from utils.visualization_data import (
    VisualizationDataPreparator,
    PieChartDataPoint,
    LineChartDataPoint,
    ComparisonChartDataPoint
)


class TestPieChartDataPoint:
    """Test PieChartDataPoint model validation."""
    
    def test_valid_pie_chart_data_point(self):
        """Test creation of valid pie chart data point."""
        data_point = PieChartDataPoint(
            name="S&P 500",
            value=60.0,
            color="#1f77b4",
            percentage="60.0%"
        )
        
        assert data_point.name == "S&P 500"
        assert data_point.value == 60.0
        assert data_point.color == "#1f77b4"
        assert data_point.percentage == "60.0%"
    
    def test_invalid_color_format(self):
        """Test validation of invalid color format."""
        with pytest.raises(ValueError, match="Color must be in hex format"):
            PieChartDataPoint(
                name="S&P 500",
                value=60.0,
                color="blue",  # Invalid format
                percentage="60.0%"
            )
    
    def test_value_rounding(self):
        """Test automatic rounding of values."""
        data_point = PieChartDataPoint(
            name="Bonds",
            value=33.333333,
            color="#2ca02c",
            percentage="33.33%"
        )
        
        assert data_point.value == 33.33
    
    def test_negative_value_validation(self):
        """Test validation of negative values."""
        with pytest.raises(ValueError):
            PieChartDataPoint(
                name="Gold",
                value=-5.0,  # Invalid negative value
                color="#9467bd",
                percentage="-5.0%"
            )


class TestLineChartDataPoint:
    """Test LineChartDataPoint model validation."""
    
    def test_valid_line_chart_data_point(self):
        """Test creation of valid line chart data point."""
        data_point = LineChartDataPoint(
            year=5,
            portfolio_value=150000.0,
            formatted_value="$150,000.00",
            annual_return=8.5,
            cumulative_return=50.0
        )
        
        assert data_point.year == 5
        assert data_point.portfolio_value == 150000.0
        assert data_point.formatted_value == "$150,000.00"
        assert data_point.annual_return == 8.5
        assert data_point.cumulative_return == 50.0
    
    def test_optional_fields(self):
        """Test line chart data point with optional fields."""
        data_point = LineChartDataPoint(
            year=0,
            portfolio_value=100000.0,
            formatted_value="$100,000.00"
        )
        
        assert data_point.annual_return is None
        assert data_point.cumulative_return is None
    
    def test_value_rounding(self):
        """Test automatic rounding of financial values."""
        data_point = LineChartDataPoint(
            year=3,
            portfolio_value=123456.789,
            formatted_value="$123,456.79",
            annual_return=7.123456,
            cumulative_return=23.456789
        )
        
        assert data_point.portfolio_value == 123456.79
        assert data_point.annual_return == 7.12
        assert data_point.cumulative_return == 23.46


class TestComparisonChartDataPoint:
    """Test ComparisonChartDataPoint model validation."""
    
    def test_valid_comparison_data_point(self):
        """Test creation of valid comparison chart data point."""
        data_point = ComparisonChartDataPoint(
            year=3,
            portfolio_value=130000.0,
            benchmark_value=125000.0,
            portfolio_return=30.0,
            benchmark_return=25.0,
            outperformance=5.0
        )
        
        assert data_point.year == 3
        assert data_point.portfolio_value == 130000.0
        assert data_point.benchmark_value == 125000.0
        assert data_point.portfolio_return == 30.0
        assert data_point.benchmark_return == 25.0
        assert data_point.outperformance == 5.0
    
    def test_value_rounding(self):
        """Test automatic rounding of comparison values."""
        data_point = ComparisonChartDataPoint(
            year=2,
            portfolio_value=115555.555,
            benchmark_value=112222.222,
            portfolio_return=15.555555,
            benchmark_return=12.222222,
            outperformance=3.333333
        )
        
        assert data_point.portfolio_value == 115555.55  # Python's round() behavior
        assert data_point.benchmark_value == 112222.22
        assert data_point.portfolio_return == 15.56
        assert data_point.benchmark_return == 12.22
        assert data_point.outperformance == 3.33


class TestVisualizationDataPreparator:
    """Test VisualizationDataPreparator functionality."""
    
    @pytest.fixture
    def preparator(self):
        """Create visualization data preparator instance."""
        return VisualizationDataPreparator()
    
    @pytest.fixture
    def sample_allocation(self):
        """Create sample portfolio allocation."""
        return PortfolioAllocation(
            sp500=50.0,
            small_cap=20.0,
            bonds=20.0,
            real_estate=10.0,
            gold=0.0
        )
    
    @pytest.fixture
    def sample_projections(self):
        """Create sample projection results."""
        return [
            ProjectionResult(year=1, portfolio_value=110000.0, annual_return=10.0, cumulative_return=10.0),
            ProjectionResult(year=2, portfolio_value=121000.0, annual_return=10.0, cumulative_return=21.0),
            ProjectionResult(year=3, portfolio_value=133100.0, annual_return=10.0, cumulative_return=33.1)
        ]
    
    @pytest.fixture
    def sample_risk_metrics(self):
        """Create sample risk metrics."""
        return RiskMetrics(
            alpha=2.5,
            beta=1.2,
            volatility=18.0,
            sharpe_ratio=0.8,
            max_drawdown=-25.0
        )
    
    def test_prepare_pie_chart_data(self, preparator, sample_allocation):
        """Test pie chart data preparation."""
        pie_data = preparator.prepare_pie_chart_data(sample_allocation)
        
        # Should exclude zero allocations by default
        assert len(pie_data) == 4  # Gold excluded (0%)
        
        # Check data structure
        for data_point in pie_data:
            assert isinstance(data_point, PieChartDataPoint)
            assert data_point.value > 0
            assert data_point.color.startswith('#')
            assert data_point.percentage.endswith('%')
        
        # Check sorting (descending by value)
        values = [point.value for point in pie_data]
        assert values == sorted(values, reverse=True)
    
    def test_prepare_pie_chart_data_include_zeros(self, preparator, sample_allocation):
        """Test pie chart data preparation including zero allocations."""
        pie_data = preparator.prepare_pie_chart_data(sample_allocation, include_zero_allocations=True)
        
        # Should include all allocations
        assert len(pie_data) == 5  # All assets included
        
        # Find gold allocation (should be 0)
        gold_data = next((point for point in pie_data if point.name == "Gold"), None)
        assert gold_data is not None
        assert gold_data.value == 0.0
    
    def test_prepare_line_chart_data(self, preparator, sample_projections):
        """Test line chart data preparation."""
        initial_investment = 100000.0
        line_data = preparator.prepare_line_chart_data(sample_projections, initial_investment)
        
        # Should include initial year + projection years
        assert len(line_data) == 4  # Year 0 + 3 projection years
        
        # Check initial year
        initial_point = line_data[0]
        assert initial_point.year == 0
        assert initial_point.portfolio_value == initial_investment
        assert initial_point.annual_return == 0.0
        assert initial_point.cumulative_return == 0.0
        
        # Check data structure
        for data_point in line_data:
            assert isinstance(data_point, LineChartDataPoint)
            assert data_point.formatted_value.startswith('$')
            assert ',' in data_point.formatted_value  # Should have thousand separators
    
    def test_prepare_comparison_chart_data(self, preparator, sample_projections):
        """Test comparison chart data preparation."""
        initial_investment = 100000.0
        comparison_data = preparator.prepare_comparison_chart_data(sample_projections, initial_investment)
        
        # Should include initial year + projection years
        assert len(comparison_data) == 4
        
        # Check initial year
        initial_point = comparison_data[0]
        assert initial_point.year == 0
        assert initial_point.portfolio_value == initial_investment
        assert initial_point.benchmark_value == initial_investment
        assert initial_point.outperformance == 0.0
        
        # Check benchmark calculations
        for i, data_point in enumerate(comparison_data[1:], 1):
            assert isinstance(data_point, ComparisonChartDataPoint)
            assert data_point.benchmark_value > initial_investment  # Should grow
            # Outperformance should be portfolio return - benchmark return
            expected_outperformance = data_point.portfolio_return - data_point.benchmark_return
            assert abs(data_point.outperformance - expected_outperformance) < 0.01
    
    def test_prepare_comparison_chart_data_custom_benchmark(self, preparator, sample_projections):
        """Test comparison chart data with custom benchmark return."""
        initial_investment = 100000.0
        custom_benchmark_return = 8.0
        comparison_data = preparator.prepare_comparison_chart_data(
            sample_projections, initial_investment, custom_benchmark_return
        )
        
        # Check that custom benchmark return is used
        year_3_point = comparison_data[3]  # Year 3
        expected_benchmark_value = initial_investment * ((1 + custom_benchmark_return / 100) ** 3)
        assert abs(year_3_point.benchmark_value - expected_benchmark_value) < 0.01
    
    def test_prepare_risk_visualization_data(self, preparator, sample_risk_metrics):
        """Test risk visualization data preparation."""
        risk_data = preparator.prepare_risk_visualization_data(sample_risk_metrics)
        
        # Check structure
        assert "portfolio_metrics" in risk_data
        assert "risk_score" in risk_data
        assert "risk_level" in risk_data
        
        # Check portfolio metrics
        metrics = risk_data["portfolio_metrics"]
        assert len(metrics) == 5  # 5 risk metrics
        
        metric_names = [m["metric"] for m in metrics]
        expected_metrics = ["Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)", "Beta", "Alpha (%)"]
        assert all(name in metric_names for name in expected_metrics)
        
        # Check risk score and level
        assert isinstance(risk_data["risk_score"], float)
        assert 0 <= risk_data["risk_score"] <= 100
        assert risk_data["risk_level"] in ["Low", "Moderate", "High"]
    
    def test_prepare_allocation_trend_data(self, preparator):
        """Test allocation trend data preparation."""
        allocations_over_time = [
            {"year": 0, "sp500": 60, "small_cap": 20, "bonds": 15, "real_estate": 5, "gold": 0},
            {"year": 5, "sp500": 55, "small_cap": 20, "bonds": 20, "real_estate": 5, "gold": 0},
            {"year": 10, "sp500": 50, "small_cap": 20, "bonds": 25, "real_estate": 5, "gold": 0}
        ]
        
        trend_data = preparator.prepare_allocation_trend_data(allocations_over_time)
        
        assert len(trend_data) == 3
        
        for trend_point in trend_data:
            # Check all required fields are present
            required_fields = ["year", "sp500", "small_cap", "bonds", "real_estate", "gold"]
            assert all(field in trend_point for field in required_fields)
            
            # Check allocation sums to 100%
            total_allocation = sum(v for k, v in trend_point.items() if k != "year")
            assert abs(total_allocation - 100) < 0.01
    
    def test_prepare_allocation_trend_data_normalization(self, preparator):
        """Test allocation trend data with normalization needed."""
        # Allocation that doesn't sum to 100%
        allocations_over_time = [
            {"year": 0, "sp500": 60, "small_cap": 20, "bonds": 15, "real_estate": 4, "gold": 0}  # Sums to 99%
        ]
        
        trend_data = preparator.prepare_allocation_trend_data(allocations_over_time)
        
        # Should be normalized to 100%
        trend_point = trend_data[0]
        total_allocation = sum(v for k, v in trend_point.items() if k != "year")
        assert abs(total_allocation - 100) < 0.01
    
    def test_validate_chart_data_pie(self, preparator, sample_allocation):
        """Test chart data validation for pie charts."""
        pie_data = preparator.prepare_pie_chart_data(sample_allocation)
        pie_data_dict = [point.model_dump() for point in pie_data]
        
        assert preparator.validate_chart_data(pie_data_dict, "pie") is True
        
        # Test invalid data
        invalid_data = [{"name": "Test", "value": -5}]  # Missing required fields
        assert preparator.validate_chart_data(invalid_data, "pie") is False
    
    def test_validate_chart_data_line(self, preparator, sample_projections):
        """Test chart data validation for line charts."""
        line_data = preparator.prepare_line_chart_data(sample_projections, 100000.0)
        line_data_dict = [point.model_dump() for point in line_data]
        
        assert preparator.validate_chart_data(line_data_dict, "line") is True
        
        # Test invalid data
        invalid_data = [{"year": 1}]  # Missing required fields
        assert preparator.validate_chart_data(invalid_data, "line") is False
    
    def test_validate_chart_data_comparison(self, preparator, sample_projections):
        """Test chart data validation for comparison charts."""
        comparison_data = preparator.prepare_comparison_chart_data(sample_projections, 100000.0)
        comparison_data_dict = [point.model_dump() for point in comparison_data]
        
        assert preparator.validate_chart_data(comparison_data_dict, "comparison") is True
        
        # Test invalid data
        invalid_data = [{"year": 1, "portfolio_value": 100000}]  # Missing required fields
        assert preparator.validate_chart_data(invalid_data, "comparison") is False
    
    def test_validate_chart_data_risk(self, preparator, sample_risk_metrics):
        """Test chart data validation for risk charts."""
        risk_data = preparator.prepare_risk_visualization_data(sample_risk_metrics)
        
        assert preparator.validate_chart_data(risk_data, "risk") is True
        
        # Test invalid data
        invalid_data = {"portfolio_metrics": []}  # Missing required fields
        assert preparator.validate_chart_data(invalid_data, "risk") is False
    
    def test_calculate_risk_score(self, preparator, sample_risk_metrics):
        """Test risk score calculation."""
        risk_score = preparator._calculate_risk_score(sample_risk_metrics)
        
        assert isinstance(risk_score, float)
        assert 0 <= risk_score <= 100
        
        # Test with low risk metrics
        low_risk_metrics = RiskMetrics(
            alpha=1.0,
            beta=0.8,
            volatility=10.0,
            sharpe_ratio=1.5,
            max_drawdown=-10.0
        )
        low_risk_score = preparator._calculate_risk_score(low_risk_metrics)
        assert low_risk_score < risk_score  # Should be lower than sample
    
    def test_determine_risk_level(self, preparator):
        """Test risk level determination."""
        # Low risk metrics
        low_risk_metrics = RiskMetrics(
            alpha=0.5,
            beta=0.7,
            volatility=8.0,
            sharpe_ratio=1.2,
            max_drawdown=-8.0
        )
        assert preparator._determine_risk_level(low_risk_metrics) == "Low"
        
        # High risk metrics
        high_risk_metrics = RiskMetrics(
            alpha=1.0,
            beta=1.8,
            volatility=25.0,
            sharpe_ratio=0.3,
            max_drawdown=-40.0
        )
        assert preparator._determine_risk_level(high_risk_metrics) == "High"
    
    def test_asset_colors_consistency(self, preparator):
        """Test that asset colors are consistent."""
        assert "S&P 500" in preparator.ASSET_COLORS
        assert "Bonds" in preparator.ASSET_COLORS
        assert "Gold" in preparator.ASSET_COLORS
        
        # All colors should be valid hex codes
        for color in preparator.ASSET_COLORS.values():
            assert color.startswith('#')
            assert len(color) == 7
    
    def test_currency_formatting(self, preparator, sample_projections):
        """Test currency formatting in line chart data."""
        initial_investment = 1234567.89
        line_data = preparator.prepare_line_chart_data(sample_projections, initial_investment, "€")
        
        # Check custom currency symbol
        initial_point = line_data[0]
        assert initial_point.formatted_value.startswith('€')
        assert ',' in initial_point.formatted_value  # Thousand separators
        
        # Test default currency symbol
        line_data_default = preparator.prepare_line_chart_data(sample_projections, initial_investment)
        initial_point_default = line_data_default[0]
        assert initial_point_default.formatted_value.startswith('$')
    
    def test_prepare_rebalancing_comparison_data(self, preparator):
        """Test rebalancing comparison data preparation."""
        with_rebalancing = [
            {'year': 0, 'portfolio_value': 100000, 'end_period_value': 110000},
            {'year': 2, 'portfolio_value': 110000, 'end_period_value': 125000},
            {'year': 4, 'portfolio_value': 125000, 'end_period_value': 140000}
        ]
        
        without_rebalancing = [
            {'year': 0, 'portfolio_value': 100000},
            {'year': 2, 'portfolio_value': 120000},
            {'year': 4, 'portfolio_value': 135000}
        ]
        
        comparison_data = preparator.prepare_rebalancing_comparison_data(
            with_rebalancing, without_rebalancing
        )
        
        # Verify structure
        assert len(comparison_data) == 3
        
        for point in comparison_data:
            assert 'year' in point
            assert 'with_rebalancing' in point
            assert 'without_rebalancing' in point
            assert 'with_rebalancing_formatted' in point
            assert 'without_rebalancing_formatted' in point
            assert 'outperformance' in point
            assert 'outperformance_formatted' in point
        
        # Check calculations
        final_point = comparison_data[-1]
        assert final_point['with_rebalancing'] == 140000
        assert final_point['without_rebalancing'] == 135000
        expected_outperformance = (140000 - 135000) / 135000 * 100
        assert abs(final_point['outperformance'] - expected_outperformance) < 0.01
        
        # Check formatting
        assert final_point['with_rebalancing_formatted'].startswith('$')
        assert final_point['outperformance_formatted'].endswith('%')
    
    def test_prepare_rebalancing_events_data(self, preparator):
        """Test rebalancing events data preparation."""
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
                'rationale': 'Reduce equity exposure'
            },
            {
                'year': 4,
                'allocation': {'sp500': 50, 'small_cap': 10, 't_bills': 40},
                'changes': {'sp500': -5, 'small_cap': -5, 't_bills': 10},
                'rationale': 'Further risk reduction'
            }
        ]
        
        events_data = preparator.prepare_rebalancing_events_data(rebalancing_schedule)
        
        # Should skip initial allocation (year 0)
        assert len(events_data) == 2
        
        # Check first rebalancing event
        first_event = events_data[0]
        assert first_event['year'] == 2
        assert first_event['equity_change'] == -10  # -5 sp500 + -5 small_cap
        assert first_event['bond_change'] == 10     # +10 t_bills
        assert first_event['max_change_asset'] == 't_bills'
        assert first_event['max_change_value'] == 10
        assert first_event['rationale'] == 'Reduce equity exposure'
        assert first_event['total_changes'] == 3  # 3 assets changed
        
        # Check structure
        for event in events_data:
            assert 'year' in event
            assert 'equity_change' in event
            assert 'bond_change' in event
            assert 'max_change_asset' in event
            assert 'max_change_value' in event
            assert 'rationale' in event
            assert 'total_changes' in event
    
    def test_prepare_rebalancing_events_data_empty_schedule(self, preparator):
        """Test rebalancing events data with empty schedule."""
        events_data = preparator.prepare_rebalancing_events_data([])
        assert events_data == []
    
    def test_prepare_rebalancing_events_data_only_initial(self, preparator):
        """Test rebalancing events data with only initial allocation."""
        rebalancing_schedule = [
            {
                'year': 0,
                'allocation': {'sp500': 60, 'small_cap': 20, 't_bills': 20},
                'changes': {},
                'rationale': 'Initial allocation'
            }
        ]
        
        events_data = preparator.prepare_rebalancing_events_data(rebalancing_schedule)
        assert events_data == []  # Should skip initial allocation
    
    def test_allocation_trend_data_with_detailed_bonds(self, preparator):
        """Test allocation trend data with detailed bond breakdown."""
        allocations_over_time = [
            {
                "year": 0, 
                "sp500": 50, 
                "small_cap": 20, 
                "t_bills": 10, 
                "t_bonds": 15, 
                "corporate_bonds": 5,
                "real_estate": 0, 
                "gold": 0
            },
            {
                "year": 5, 
                "sp500": 45, 
                "small_cap": 15, 
                "t_bills": 15, 
                "t_bonds": 20, 
                "corporate_bonds": 5,
                "real_estate": 0, 
                "gold": 0
            }
        ]
        
        trend_data = preparator.prepare_allocation_trend_data(allocations_over_time)
        
        assert len(trend_data) == 2
        
        for trend_point in trend_data:
            # Check all bond types are present
            assert 't_bills' in trend_point
            assert 't_bonds' in trend_point
            assert 'corporate_bonds' in trend_point
            
            # Check allocation sums to 100%
            total_allocation = sum(v for k, v in trend_point.items() if k != "year")
            assert abs(total_allocation - 100) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])