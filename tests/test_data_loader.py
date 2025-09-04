"""
Unit tests for historical data loading and preprocessing utilities.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

from utils.data_loader import (
    HistoricalDataLoader,
    DataValidationError,
    load_and_clean_historical_data,
    validate_data_integrity
)
from models.data_models import AssetReturns


class TestHistoricalDataLoader:
    """Test cases for HistoricalDataLoader class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample historical returns data for testing."""
        return pd.DataFrame({
            'Year': [2020, 2021, 2022, 2023],
            'S&P 500 (includes dividends)': [0.1843, 0.2889, -0.1811, 0.2643],
            'US Small cap (bottom decile)': [0.1998, 0.1468, -0.2044, 0.1647],
            '3-month T.Bill': [0.0037, 0.0005, 0.0178, 0.0524],
            'US T. Bond (10-year)': [0.0001, -0.0429, -0.1311, -0.0216],
            ' Baa Corporate Bond': [0.0934, -0.0109, -0.1513, 0.0856],
            'Real Estate': [0.0211, 0.4346, -0.2595, 0.1134],
            'Gold*': [0.2495, -0.0374, -0.0011, 0.1341]
        })
    
    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values."""
        data = pd.DataFrame({
            'Year': [2020, 2021, 2022, 2023],
            'S&P 500 (includes dividends)': [0.1843, np.nan, -0.1811, 0.2643],
            'US Small cap (bottom decile)': [0.1998, 0.1468, np.nan, 0.1647],
            '3-month T.Bill': [0.0037, 0.0005, 0.0178, np.nan],
            'US T. Bond (10-year)': [0.0001, -0.0429, -0.1311, -0.0216],
            ' Baa Corporate Bond': [0.0934, -0.0109, -0.1513, 0.0856],  # Remove NaN from first row
            'Real Estate': [0.0211, 0.4346, -0.2595, 0.1134],
            'Gold*': [0.2495, -0.0374, -0.0011, 0.1341]
        })
        return data
    
    @pytest.fixture
    def sample_data_percentage_format(self):
        """Create sample data in percentage format (needs conversion)."""
        return pd.DataFrame({
            'Year': [2020, 2021, 2022, 2023],
            'S&P 500 (includes dividends)': [18.43, 28.89, -18.11, 26.43],
            'US Small cap (bottom decile)': [19.98, 14.68, -20.44, 16.47],
            '3-month T.Bill': [0.37, 0.05, 1.78, 5.24],
            'US T. Bond (10-year)': [0.01, -4.29, -13.11, -2.16],
            ' Baa Corporate Bond': [9.34, -1.09, -15.13, 8.56],
            'Real Estate': [2.11, 43.46, -25.95, 11.34],
            'Gold*': [24.95, -3.74, -0.11, 13.41]
        })
    
    def test_init(self):
        """Test HistoricalDataLoader initialization."""
        loader = HistoricalDataLoader("test_file.xls")
        assert loader.file_path == Path("test_file.xls")
        assert loader.raw_data is None
        assert loader.cleaned_data is None
        assert loader.validation_summary == {}
    
    def test_load_raw_data_file_not_found(self):
        """Test loading data when file doesn't exist."""
        loader = HistoricalDataLoader("nonexistent_file.xls")
        
        with pytest.raises(FileNotFoundError):
            loader.load_raw_data()
    
    @patch('pandas.read_excel')
    def test_load_raw_data_success(self, mock_read_excel, sample_data):
        """Test successful data loading."""
        # Mock the Excel file reading
        mock_read_excel.return_value = sample_data
        
        # Create a temporary file to satisfy file existence check
        with tempfile.NamedTemporaryFile(suffix='.xls', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            loader = HistoricalDataLoader(tmp_path)
            result = loader.load_raw_data()
            
            assert len(result) == 4
            assert 'Year' in result.columns
            assert result['Year'].dtype == int
            assert loader.raw_data is not None
            
        finally:
            os.unlink(tmp_path)
    
    def test_validate_numeric_ranges(self, sample_data):
        """Test numeric range validation."""
        loader = HistoricalDataLoader()
        
        # Rename columns to match expected format
        sample_data_renamed = sample_data.rename(columns=loader.column_mapping)
        
        validation_results = loader.validate_numeric_ranges(sample_data_renamed)
        
        assert validation_results['total_rows'] == 4
        assert len(validation_results['columns_validated']) > 0
        assert 'summary_stats' in validation_results
        assert 'outliers_found' in validation_results
        assert 'invalid_values' in validation_results
    
    def test_validate_numeric_ranges_with_outliers(self):
        """Test validation with outlier values."""
        loader = HistoricalDataLoader()
        
        # Create data with outliers
        outlier_data = pd.DataFrame({
            'year': [2020, 2021, 2022, 2023],
            'sp500': [0.18, 5.0, -0.18, 0.26],  # 5.0 is an outlier (500% return)
            'small_cap': [0.20, 0.15, -0.20, 0.16],
            't_bills': [0.004, 0.001, 0.018, 0.052],
            't_bonds': [0.0001, -0.043, -0.131, -0.022],
            'corporate_bonds': [0.093, -0.011, -0.151, 0.086],
            'real_estate': [0.021, 0.435, -0.260, 0.113],
            'gold': [0.250, -0.037, -0.001, 0.134]
        })
        
        validation_results = loader.validate_numeric_ranges(outlier_data)
        
        # Should detect outlier in sp500
        assert validation_results['outliers_found']['sp500'] > 0
    
    def test_handle_missing_values_forward_fill(self, sample_data_with_missing):
        """Test forward fill strategy for missing values."""
        loader = HistoricalDataLoader()
        
        result = loader.handle_missing_values(sample_data_with_missing, "forward_fill")
        
        # Check that missing values are filled
        assert not result.isnull().any().any()
        
        # Check specific forward fill behavior
        assert result.loc[1, 'S&P 500 (includes dividends)'] == 0.1843  # Forward filled from row 0
    
    def test_handle_missing_values_interpolate(self, sample_data_with_missing):
        """Test interpolation strategy for missing values."""
        loader = HistoricalDataLoader()
        
        result = loader.handle_missing_values(sample_data_with_missing, "interpolate")
        
        # Check that missing values are filled
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert not result[numeric_cols].isnull().any().any()
    
    def test_handle_missing_values_drop(self, sample_data_with_missing):
        """Test drop strategy for missing values."""
        loader = HistoricalDataLoader()
        
        result = loader.handle_missing_values(sample_data_with_missing, "drop")
        
        # Should have fewer rows after dropping
        assert len(result) < len(sample_data_with_missing)
        assert not result.isnull().any().any()
    
    def test_handle_missing_values_invalid_strategy(self, sample_data):
        """Test invalid missing value strategy."""
        loader = HistoricalDataLoader()
        
        with pytest.raises(ValueError, match="Unknown missing value strategy"):
            loader.handle_missing_values(sample_data, "invalid_strategy")
    
    def test_normalize_returns_format_percentage_to_decimal(self, sample_data_percentage_format):
        """Test conversion from percentage to decimal format."""
        loader = HistoricalDataLoader()
        
        result = loader.normalize_returns_format(sample_data_percentage_format)
        
        # Check that large values were converted to decimals
        assert result['S&P 500 (includes dividends)'].max() < 1.0  # Should be 0.2889, not 28.89
        assert abs(result.loc[0, 'S&P 500 (includes dividends)'] - 0.1843) < 0.0001
    
    def test_normalize_returns_format_already_decimal(self, sample_data):
        """Test normalization when data is already in decimal format."""
        loader = HistoricalDataLoader()
        
        result = loader.normalize_returns_format(sample_data)
        
        # Values should remain similar (just rounded)
        assert abs(result.loc[0, 'S&P 500 (includes dividends)'] - 0.1843) < 0.0001
    
    @patch.object(HistoricalDataLoader, 'load_raw_data')
    def test_clean_and_preprocess_pipeline(self, mock_load_raw_data, sample_data):
        """Test the complete cleaning and preprocessing pipeline."""
        loader = HistoricalDataLoader()
        loader.raw_data = sample_data  # Set raw_data directly
        mock_load_raw_data.return_value = sample_data
        
        result = loader.clean_and_preprocess()
        
        # Check that data was processed
        assert loader.cleaned_data is not None
        assert len(result) > 0
        assert 'year' in result.columns  # Should be renamed
        assert result['year'].dtype == int
        
        # Check that validation summary was created
        assert loader.validation_summary is not None
        assert 'total_rows' in loader.validation_summary
    
    def test_get_cleaning_summary_no_operations(self):
        """Test getting summary when no operations performed."""
        loader = HistoricalDataLoader()
        
        summary = loader.get_cleaning_summary()
        
        assert "error" in summary
    
    @patch.object(HistoricalDataLoader, 'clean_and_preprocess')
    def test_get_cleaning_summary_with_operations(self, mock_clean, sample_data):
        """Test getting summary after operations."""
        loader = HistoricalDataLoader()
        
        # Set up mock data
        loader.validation_summary = {
            'total_rows': 4,
            'columns_validated': ['sp500', 'small_cap'],
            'outliers_found': {'sp500': 0},
            'invalid_values': {'sp500': 0},
            'summary_stats': {'sp500': {'mean': 0.15}}
        }
        
        # Rename sample data columns
        cleaned_data = sample_data.rename(columns=loader.column_mapping)
        loader.cleaned_data = cleaned_data
        
        summary = loader.get_cleaning_summary()
        
        assert summary['total_rows_processed'] == 4
        assert 'final_dataset' in summary
        assert summary['final_dataset']['rows'] == 4
    
    def test_to_asset_returns_list_no_data(self):
        """Test converting to AssetReturns list when no cleaned data."""
        loader = HistoricalDataLoader()
        
        with pytest.raises(ValueError, match="No cleaned data available"):
            loader.to_asset_returns_list()
    
    def test_to_asset_returns_list_success(self, sample_data):
        """Test successful conversion to AssetReturns list."""
        loader = HistoricalDataLoader()
        
        # Set up cleaned data
        cleaned_data = sample_data.rename(columns=loader.column_mapping)
        loader.cleaned_data = cleaned_data
        
        result = loader.to_asset_returns_list()
        
        assert len(result) == 4
        assert all(isinstance(item, AssetReturns) for item in result)
        assert result[0].year == 2020
        assert abs(result[0].sp500 - 0.1843) < 0.0001


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    @patch.object(HistoricalDataLoader, 'clean_and_preprocess')
    @patch.object(HistoricalDataLoader, 'get_cleaning_summary')
    def test_load_and_clean_historical_data(self, mock_summary, mock_clean):
        """Test convenience function for loading and cleaning data."""
        sample_data = pd.DataFrame({
            'Year': [2020, 2021, 2022, 2023],
            'sp500': [0.18, 0.29, -0.18, 0.26]
        })
        mock_clean.return_value = sample_data
        mock_summary.return_value = {'test': 'summary'}
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.xls', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            cleaned_data, summary = load_and_clean_historical_data(tmp_path)
            
            assert len(cleaned_data) == 4
            assert summary == {'test': 'summary'}
            
        finally:
            os.unlink(tmp_path)
    
    def test_validate_data_integrity_success(self):
        """Test successful data integrity validation."""
        # Create data with enough rows (10+) to pass validation
        years = list(range(2010, 2024))  # 14 years
        valid_data = pd.DataFrame({
            'year': years,
            'sp500': [0.18] * len(years),
            'small_cap': [0.20] * len(years),
            't_bills': [0.004] * len(years),
            't_bonds': [0.0001] * len(years),
            'corporate_bonds': [0.093] * len(years),
            'real_estate': [0.021] * len(years),
            'gold': [0.250] * len(years)
        })
        
        result = validate_data_integrity(valid_data)
        assert result is True
    
    def test_validate_data_integrity_missing_columns(self):
        """Test validation failure due to missing columns."""
        invalid_data = pd.DataFrame({
            'year': [2020, 2021, 2022, 2023],
            'sp500': [0.18, 0.29, -0.18, 0.26]
            # Missing other required columns
        })
        
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_data_integrity(invalid_data)
    
    def test_validate_data_integrity_duplicate_years(self):
        """Test validation failure due to duplicate years."""
        invalid_data = pd.DataFrame({
            'year': [2020, 2020, 2022, 2023],  # Duplicate 2020
            'sp500': [0.18, 0.29, -0.18, 0.26],
            'small_cap': [0.20, 0.15, -0.20, 0.16],
            't_bills': [0.004, 0.001, 0.018, 0.052],
            't_bonds': [0.0001, -0.043, -0.131, -0.022],
            'corporate_bonds': [0.093, -0.011, -0.151, 0.086],
            'real_estate': [0.021, 0.435, -0.260, 0.113],
            'gold': [0.250, -0.037, -0.001, 0.134]
        })
        
        with pytest.raises(DataValidationError, match="Duplicate years found"):
            validate_data_integrity(invalid_data)
    
    def test_validate_data_integrity_extreme_values(self):
        """Test validation failure due to extreme return values."""
        invalid_data = pd.DataFrame({
            'year': [2020, 2021, 2022, 2023],
            'sp500': [0.18, 15.0, -0.18, 0.26],  # 1500% return is extreme
            'small_cap': [0.20, 0.15, -0.20, 0.16],
            't_bills': [0.004, 0.001, 0.018, 0.052],
            't_bonds': [0.0001, -0.043, -0.131, -0.022],
            'corporate_bonds': [0.093, -0.011, -0.151, 0.086],
            'real_estate': [0.021, 0.435, -0.260, 0.113],
            'gold': [0.250, -0.037, -0.001, 0.134]
        })
        
        with pytest.raises(DataValidationError, match="returns above 1000%"):
            validate_data_integrity(invalid_data)
    
    def test_validate_data_integrity_insufficient_data(self):
        """Test validation failure due to insufficient data points."""
        insufficient_data = pd.DataFrame({
            'year': [2020, 2021],  # Only 2 rows
            'sp500': [0.18, 0.29],
            'small_cap': [0.20, 0.15],
            't_bills': [0.004, 0.001],
            't_bonds': [0.0001, -0.043],
            'corporate_bonds': [0.093, -0.011],
            'real_estate': [0.021, 0.435],
            'gold': [0.250, -0.037]
        })
        
        with pytest.raises(DataValidationError, match="Insufficient data points"):
            validate_data_integrity(insufficient_data)


class TestDataValidationError:
    """Test cases for custom DataValidationError exception."""
    
    def test_data_validation_error_creation(self):
        """Test creating DataValidationError."""
        error = DataValidationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)


if __name__ == "__main__":
    pytest.main([__file__])