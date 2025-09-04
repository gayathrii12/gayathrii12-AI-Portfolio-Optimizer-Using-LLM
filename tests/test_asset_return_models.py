"""
Unit tests for AssetReturnModels class.

Tests ML model training, prediction, and validation functionality
for various asset classes.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

# Add the project root to the path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.asset_return_models import AssetReturnModels


class TestAssetReturnModels:
    """Test cases for AssetReturnModels class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample historical data for testing."""
        np.random.seed(42)
        years = list(range(1970, 2024))
        n_years = len(years)
        
        data = {
            'Year': years,
            'S&P 500 (includes dividends)': np.random.normal(0.08, 0.15, n_years),
            'US Small cap (bottom decile)': np.random.normal(0.10, 0.20, n_years),
            '3-month T.Bill': np.random.normal(0.03, 0.02, n_years),
            'US T. Bond (10-year)': np.random.normal(0.05, 0.08, n_years),
            ' Baa Corporate Bond': np.random.normal(0.06, 0.06, n_years),
            'Real Estate': np.random.normal(0.07, 0.12, n_years),
            'Gold*': np.random.normal(0.04, 0.18, n_years)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mock_excel_data(self, sample_data):
        """Mock Excel data loading."""
        # Create a mock DataFrame that simulates the Excel structure
        header_rows = pd.DataFrame({
            'col1': ['Some header', 'More header', 'Year'],
            'col2': ['data', 'data', 'S&P 500 (includes dividends)'],
            'col3': ['data', 'data', 'US Small cap (bottom decile)']
        })
        
        # Combine header with actual data
        full_data = pd.concat([header_rows, sample_data], ignore_index=True)
        return full_data
    
    @pytest.fixture
    def asset_models(self, sample_data):
        """Create AssetReturnModels instance with mocked data."""
        models = AssetReturnModels("test_file.xls")
        models.historical_data = sample_data
        return models
    
    def test_initialization(self):
        """Test AssetReturnModels initialization."""
        models = AssetReturnModels("test_file.xls")
        
        assert models.data_file == "test_file.xls"
        assert len(models.asset_columns) == 7
        assert 'sp500' in models.asset_columns
        assert 'small_cap' in models.asset_columns
        assert models.historical_data is None
        assert len(models.models) == 0
    
    @patch('pandas.read_excel')
    def test_load_historical_data(self, mock_read_excel, mock_excel_data):
        """Test loading historical data from Excel file."""
        mock_read_excel.return_value = mock_excel_data
        
        models = AssetReturnModels("test_file.xls")
        data = models.load_historical_data()
        
        assert isinstance(data, pd.DataFrame)
        assert 'Year' in data.columns
        assert len(data) > 0
        assert models.historical_data is not None
    
    def test_create_features(self, asset_models, sample_data):
        """Test feature creation for ML models."""
        target_col = 'S&P 500 (includes dividends)'
        
        X, y = asset_models.create_features(sample_data, target_col, lookback_years=3)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X) == len(y)
        assert len(X) > 0
        assert X.shape[1] > 3  # Should have multiple features
    
    def test_train_model_random_forest(self, asset_models):
        """Test training a random forest model."""
        metrics = asset_models.train_model('sp500', 'random_forest')
        
        assert isinstance(metrics, dict)
        assert 'train_r2' in metrics
        assert 'test_r2' in metrics
        assert 'train_rmse' in metrics
        assert 'test_rmse' in metrics
        assert 'cv_mean' in metrics
        assert 'sp500' in asset_models.models
        assert 'sp500' in asset_models.scalers
    
    def test_train_model_linear(self, asset_models):
        """Test training a linear regression model."""
        metrics = asset_models.train_model('sp500', 'linear')
        
        assert isinstance(metrics, dict)
        assert 'train_r2' in metrics
        assert 'test_r2' in metrics
        assert 'sp500' in asset_models.models
        assert 'sp500' in asset_models.scalers
    
    def test_train_model_invalid_asset(self, asset_models):
        """Test training model with invalid asset class."""
        with pytest.raises(ValueError, match="Unknown asset class"):
            asset_models.train_model('invalid_asset')
    
    def test_train_all_models(self, asset_models):
        """Test training models for all asset classes."""
        all_metrics = asset_models.train_all_models('random_forest')
        
        assert isinstance(all_metrics, dict)
        assert len(all_metrics) == len(asset_models.asset_columns)
        
        # Check that most models trained successfully
        successful_models = [k for k, v in all_metrics.items() if 'error' not in v]
        assert len(successful_models) >= 5  # At least 5 out of 7 should work
    
    def test_predict_returns(self, asset_models):
        """Test return prediction."""
        # First train a model
        asset_models.train_model('sp500')
        
        # Then predict
        prediction = asset_models.predict_returns('sp500', horizon=1)
        
        assert isinstance(prediction, (float, np.float64))
        assert -1.0 <= prediction <= 1.0  # Reasonable return range
    
    def test_predict_returns_untrained_model(self, asset_models):
        """Test prediction with untrained model."""
        with pytest.raises(ValueError, match="Model for sp500 not trained"):
            asset_models.predict_returns('sp500')
    
    def test_get_all_predictions(self, asset_models):
        """Test getting predictions for all trained models."""
        # Train a few models
        asset_models.train_model('sp500')
        asset_models.train_model('t_bills')
        
        predictions = asset_models.get_all_predictions()
        
        assert isinstance(predictions, dict)
        assert 'sp500' in predictions
        assert 't_bills' in predictions
        assert predictions['sp500'] is not None
        assert predictions['t_bills'] is not None
    
    def test_validate_model_accuracy(self, asset_models):
        """Test model validation."""
        # Train a model first
        asset_models.train_model('sp500')
        
        validation_results = asset_models.validate_model_accuracy('sp500', test_years=3)
        
        assert isinstance(validation_results, dict)
        assert 'rmse' in validation_results
        assert 'mae' in validation_results
        assert 'direction_accuracy' in validation_results
        assert 'predictions' in validation_results
        assert 'actuals' in validation_results
    
    def test_validate_untrained_model(self, asset_models):
        """Test validation with untrained model."""
        with pytest.raises(ValueError, match="Model for sp500 not trained"):
            asset_models.validate_model_accuracy('sp500')
    
    def test_save_and_load_models(self, asset_models):
        """Test saving and loading models."""
        # Train a model
        asset_models.train_model('sp500')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save models
            asset_models.save_models(temp_dir)
            
            # Check files were created
            model_file = Path(temp_dir) / "sp500_model.joblib"
            scaler_file = Path(temp_dir) / "sp500_scaler.joblib"
            assert model_file.exists()
            assert scaler_file.exists()
            
            # Create new instance and load models
            new_models = AssetReturnModels("test_file.xls")
            new_models.historical_data = asset_models.historical_data
            new_models.load_models(temp_dir)
            
            assert 'sp500' in new_models.models
            assert 'sp500' in new_models.scalers
    
    def test_load_models_nonexistent_directory(self, asset_models):
        """Test loading models from nonexistent directory."""
        with pytest.raises(ValueError, match="Model directory .* does not exist"):
            asset_models.load_models("nonexistent_directory")
    
    def test_get_model_summary(self, asset_models):
        """Test getting model summary."""
        # Train a few models
        asset_models.train_model('sp500', 'random_forest')
        asset_models.train_model('t_bills', 'linear')
        
        summary = asset_models.get_model_summary()
        
        assert isinstance(summary, dict)
        assert 'sp500' in summary
        assert 't_bills' in summary
        
        sp500_info = summary['sp500']
        assert 'model_type' in sp500_info
        assert 'trained' in sp500_info
        assert 'asset_name' in sp500_info
        assert sp500_info['trained'] is True
    
    def test_feature_creation_edge_cases(self, asset_models):
        """Test feature creation with edge cases."""
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'Year': [2020, 2021, 2022],
            'S&P 500 (includes dividends)': [0.1, 0.2, 0.15],
            'US Small cap (bottom decile)': [0.12, 0.18, 0.10],
            '3-month T.Bill': [0.01, 0.02, 0.03]
        })
        
        target_col = 'S&P 500 (includes dividends)'
        X, y = asset_models.create_features(minimal_data, target_col, lookback_years=2)
        
        assert len(X) >= 1  # Should have at least one sample
        assert len(X) == len(y)
    
    def test_prediction_with_missing_data(self, asset_models):
        """Test prediction handling when some data is missing."""
        # Create data with some missing values
        data_with_missing = asset_models.historical_data.copy()
        data_with_missing.loc[data_with_missing.index[-5:], 'Gold*'] = np.nan
        
        asset_models.historical_data = data_with_missing
        
        # Should still be able to train and predict
        metrics = asset_models.train_model('sp500')
        assert 'train_r2' in metrics
        
        prediction = asset_models.predict_returns('sp500')
        assert isinstance(prediction, (float, np.float64))


class TestAssetReturnModelsIntegration:
    """Integration tests using real data structure."""
    
    def test_real_data_structure_simulation(self):
        """Test with data structure similar to real Excel file."""
        # Create realistic test data
        years = list(range(1970, 2024))
        n_years = len(years)
        
        # Simulate realistic returns with some correlation
        np.random.seed(42)
        market_factor = np.random.normal(0, 0.1, n_years)
        
        realistic_data = pd.DataFrame({
            'Year': years,
            'S&P 500 (includes dividends)': 0.08 + market_factor + np.random.normal(0, 0.05, n_years),
            'US Small cap (bottom decile)': 0.10 + 1.2 * market_factor + np.random.normal(0, 0.08, n_years),
            '3-month T.Bill': np.maximum(0, 0.03 + 0.3 * market_factor + np.random.normal(0, 0.01, n_years)),
            'US T. Bond (10-year)': 0.05 + 0.5 * market_factor + np.random.normal(0, 0.03, n_years),
            ' Baa Corporate Bond': 0.06 + 0.7 * market_factor + np.random.normal(0, 0.04, n_years),
            'Real Estate': 0.07 + 0.8 * market_factor + np.random.normal(0, 0.06, n_years),
            'Gold*': 0.04 + np.random.normal(0, 0.15, n_years)  # Less correlated with market
        })
        
        models = AssetReturnModels("test_file.xls")
        models.historical_data = realistic_data
        
        # Test complete workflow
        all_metrics = models.train_all_models('random_forest')
        
        # Should successfully train most models
        successful_models = [k for k, v in all_metrics.items() if 'error' not in v]
        assert len(successful_models) >= 6
        
        # Test predictions
        predictions = models.get_all_predictions()
        assert len([p for p in predictions.values() if p is not None]) >= 6
        
        # Test validation
        validation = models.validate_model_accuracy('sp500', test_years=5)
        assert 'rmse' in validation
        assert validation['direction_accuracy'] >= 0.0  # Should be between 0 and 1


if __name__ == "__main__":
    pytest.main([__file__])