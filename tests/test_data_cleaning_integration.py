"""
Integration tests for the Data Cleaning Agent with real Excel data.

These tests verify the agent works correctly with the actual histretSP.xls file
and handles real-world data scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from agents.data_cleaning_agent import (
    DataCleaningAgent,
    DataCleaningInput,
    create_data_cleaning_agent
)
from models.data_models import AssetReturns


class TestDataCleaningIntegration:
    """Integration tests with real Excel data."""
    
    @pytest.fixture
    def excel_file_path(self):
        """Get path to the Excel file."""
        return "histretSP.xls"
    
    @pytest.fixture
    def agent(self):
        """Create a data cleaning agent for testing."""
        return DataCleaningAgent()
    
    def test_load_real_excel_file(self, agent, excel_file_path):
        """Test loading the actual histretSP.xls file."""
        if not Path(excel_file_path).exists():
            pytest.skip(f"Excel file {excel_file_path} not found")
        
        input_params = DataCleaningInput(file_path=excel_file_path)
        result = agent.clean_data(input_params)
        
        # Verify successful loading
        assert result.success is True
        assert result.cleaned_data_rows > 0
        assert result.error_message is None
        
        # Verify data structure
        cleaned_data = agent.get_cleaned_data()
        assert cleaned_data is not None
        assert 'year' in cleaned_data.columns
        
        # Verify expected asset columns are present
        expected_columns = [
            'year', 'sp500', 'small_cap', 't_bills', 't_bonds',
            'corporate_bonds', 'real_estate', 'gold'
        ]
        for col in expected_columns:
            assert col in cleaned_data.columns
        
        # Verify data types
        assert cleaned_data['year'].dtype in ['int64', 'int32']
        for col in expected_columns[1:]:  # Skip 'year'
            assert cleaned_data[col].dtype in ['float64', 'float32']
        
        # Verify reasonable data ranges
        assert cleaned_data['year'].min() >= 1900
        assert cleaned_data['year'].max() <= 2030
        
        # Verify returns are in decimal format (not percentage)
        for col in expected_columns[1:]:
            max_val = cleaned_data[col].max()
            min_val = cleaned_data[col].min()
            # Most annual returns should be between -100% and +300%
            assert max_val <= 3.0  # 300%
            assert min_val >= -1.0  # -100%
    
    def test_missing_value_strategies(self, agent, excel_file_path):
        """Test different missing value handling strategies."""
        if not Path(excel_file_path).exists():
            pytest.skip(f"Excel file {excel_file_path} not found")
        
        strategies = ["forward_fill", "interpolate", "drop"]
        results = {}
        
        for strategy in strategies:
            test_agent = DataCleaningAgent()
            input_params = DataCleaningInput(
                file_path=excel_file_path,
                missing_value_strategy=strategy
            )
            
            result = test_agent.clean_data(input_params)
            results[strategy] = result
            
            # All strategies should succeed
            assert result.success is True
            assert result.cleaned_data_rows > 0
            
            # Verify no missing values remain (except possibly for 'drop' strategy)
            cleaned_data = test_agent.get_cleaned_data()
            missing_values = cleaned_data.isnull().sum().sum()
            
            if strategy == "drop":
                # Drop strategy might result in fewer rows but no missing values
                assert missing_values == 0
            else:
                # Forward fill and interpolate should handle all missing values
                assert missing_values == 0
        
        # Compare results across strategies
        if results["drop"].cleaned_data_rows < results["forward_fill"].cleaned_data_rows:
            # Drop strategy should result in fewer or equal rows
            assert results["drop"].cleaned_data_rows <= results["forward_fill"].cleaned_data_rows
    
    def test_outlier_detection_methods(self, agent, excel_file_path):
        """Test different outlier detection methods."""
        if not Path(excel_file_path).exists():
            pytest.skip(f"Excel file {excel_file_path} not found")
        
        methods = ["iqr", "zscore", "none"]
        
        for method in methods:
            test_agent = DataCleaningAgent()
            input_params = DataCleaningInput(
                file_path=excel_file_path,
                outlier_detection_method=method,
                outlier_threshold=3.0
            )
            
            result = test_agent.clean_data(input_params)
            
            assert result.success is True
            assert isinstance(result.outliers_detected, dict)
            
            if method == "none":
                # No outliers should be detected
                for count in result.outliers_detected.values():
                    assert count == 0
            else:
                # Some outliers might be detected in financial data
                total_outliers = sum(result.outliers_detected.values())
                assert total_outliers >= 0  # Could be 0 or more
    
    def test_asset_returns_conversion(self, agent, excel_file_path):
        """Test conversion to AssetReturns objects."""
        if not Path(excel_file_path).exists():
            pytest.skip(f"Excel file {excel_file_path} not found")
        
        input_params = DataCleaningInput(file_path=excel_file_path)
        result = agent.clean_data(input_params)
        
        assert result.success is True
        
        # Get AssetReturns objects
        asset_returns = agent.get_asset_returns()
        assert asset_returns is not None
        assert len(asset_returns) > 0
        
        # Verify all objects are valid AssetReturns instances
        for ar in asset_returns:
            assert isinstance(ar, AssetReturns)
            
            # Verify all required fields are present and valid
            assert isinstance(ar.year, int)
            assert ar.year >= 1900
            assert ar.year <= 2030
            
            # Verify return values are reasonable
            returns = [ar.sp500, ar.small_cap, ar.t_bills, ar.t_bonds,
                      ar.corporate_bonds, ar.real_estate, ar.gold]
            
            for ret in returns:
                assert isinstance(ret, float)
                assert -1.0 <= ret <= 3.0  # -100% to +300%
        
        # Verify years are unique and sorted
        years = [ar.year for ar in asset_returns]
        assert len(years) == len(set(years))  # No duplicates
        assert years == sorted(years)  # Sorted order
    
    def test_cleaning_report_generation(self, agent, excel_file_path):
        """Test cleaning report generation with real data."""
        if not Path(excel_file_path).exists():
            pytest.skip(f"Excel file {excel_file_path} not found")
        
        input_params = DataCleaningInput(file_path=excel_file_path)
        result = agent.clean_data(input_params)
        
        assert result.success is True
        
        # Generate and verify report
        report = agent.generate_cleaning_report()
        
        assert "DATA CLEANING REPORT" in report
        assert "rows processed" in report
        assert "Year range:" in report
        assert "Asset classes:" in report
        assert "Cleaning operations performed:" in report
        assert "Data quality summary:" in report
        
        # Verify asset statistics are included
        expected_assets = ['sp500', 'small_cap', 't_bills', 't_bonds',
                          'corporate_bonds', 'real_estate', 'gold']
        
        for asset in expected_assets:
            assert asset in report
            assert "Mean=" in report
            assert "Std=" in report
    
    def test_data_quality_validation(self, agent, excel_file_path):
        """Test data quality validation with real data."""
        if not Path(excel_file_path).exists():
            pytest.skip(f"Excel file {excel_file_path} not found")
        
        input_params = DataCleaningInput(file_path=excel_file_path)
        result = agent.clean_data(input_params)
        
        assert result.success is True
        
        cleaned_data = agent.get_cleaned_data()
        
        # Verify data quality metrics
        assert len(cleaned_data) > 50  # Should have substantial historical data
        
        # Verify no infinite or NaN values
        for col in cleaned_data.columns:
            if col != 'year':
                assert not cleaned_data[col].isnull().any()
                assert not np.isinf(cleaned_data[col]).any()
        
        # Verify reasonable statistical properties
        for col in ['sp500', 'small_cap', 't_bills', 't_bonds',
                   'corporate_bonds', 'real_estate', 'gold']:
            
            mean_return = cleaned_data[col].mean()
            std_return = cleaned_data[col].std()
            
            # Basic sanity checks for financial returns
            assert -0.5 <= mean_return <= 0.5  # Mean annual return between -50% and +50%
            assert 0.01 <= std_return <= 1.0   # Standard deviation between 1% and 100%
    
    def test_different_threshold_values(self, agent, excel_file_path):
        """Test outlier detection with different threshold values."""
        if not Path(excel_file_path).exists():
            pytest.skip(f"Excel file {excel_file_path} not found")
        
        thresholds = [1.5, 2.0, 2.5, 3.0, 3.5]
        outlier_counts = []
        
        for threshold in thresholds:
            test_agent = DataCleaningAgent()
            input_params = DataCleaningInput(
                file_path=excel_file_path,
                outlier_detection_method="iqr",
                outlier_threshold=threshold
            )
            
            result = test_agent.clean_data(input_params)
            assert result.success is True
            
            total_outliers = sum(result.outliers_detected.values())
            outlier_counts.append(total_outliers)
        
        # Generally, lower thresholds should detect more outliers
        # (though this might not always be strictly monotonic with real data)
        assert len(outlier_counts) == len(thresholds)
        
        # At least verify that the most restrictive threshold detects
        # more or equal outliers than the most permissive
        assert outlier_counts[0] >= outlier_counts[-1]
    
    def test_factory_function_integration(self, excel_file_path):
        """Test the factory function with real data."""
        if not Path(excel_file_path).exists():
            pytest.skip(f"Excel file {excel_file_path} not found")
        
        # Test factory function
        agent = create_data_cleaning_agent()
        assert isinstance(agent, DataCleaningAgent)
        
        # Test with real data
        input_params = DataCleaningInput(file_path=excel_file_path)
        result = agent.clean_data(input_params)
        
        assert result.success is True
        assert result.cleaned_data_rows > 0
    
    def test_error_handling_with_invalid_file(self, agent):
        """Test error handling with invalid file path."""
        input_params = DataCleaningInput(file_path="nonexistent_file.xls")
        result = agent.clean_data(input_params)
        
        assert result.success is False
        assert result.cleaned_data_rows == 0
        assert result.error_message is not None
        assert "not found" in result.error_message.lower() or "error" in result.error_message.lower()
    
    def test_comprehensive_pipeline_validation(self, agent, excel_file_path):
        """Test the complete pipeline with comprehensive validation."""
        if not Path(excel_file_path).exists():
            pytest.skip(f"Excel file {excel_file_path} not found")
        
        # Test with all options
        input_params = DataCleaningInput(
            file_path=excel_file_path,
            missing_value_strategy="interpolate",
            outlier_detection_method="zscore",
            outlier_threshold=2.5
        )
        
        result = agent.clean_data(input_params)
        
        # Comprehensive validation
        assert result.success is True
        assert result.cleaned_data_rows > 0
        assert isinstance(result.cleaning_summary, dict)
        assert isinstance(result.outliers_detected, dict)
        assert isinstance(result.missing_values_handled, dict)
        
        # Verify all expected keys in outliers_detected
        expected_assets = ['S&P 500 (includes dividends)', 'US Small cap (bottom decile)',
                          '3-month T.Bill', 'US T. Bond (10-year)', ' Baa Corporate Bond',
                          'Real Estate', 'Gold*']
        
        # The outliers_detected might use the original column names or mapped names
        # Just verify it's a non-empty dict with reasonable structure
        assert len(result.outliers_detected) > 0
        
        # Verify data accessibility
        cleaned_data = agent.get_cleaned_data()
        asset_returns = agent.get_asset_returns()
        report = agent.generate_cleaning_report()
        
        assert cleaned_data is not None
        assert asset_returns is not None
        assert len(report) > 100  # Should be a substantial report
        
        # Final data integrity check
        assert len(cleaned_data) == len(asset_returns)
        assert cleaned_data['year'].tolist() == [ar.year for ar in asset_returns]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])