"""
Integration tests for AssetReturnModels with real Excel data.

These tests use the actual histretSP.xls file to verify the models
work correctly with real historical data.
"""

import pytest
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.asset_return_models import AssetReturnModels


class TestAssetReturnModelsIntegration:
    """Integration tests with real Excel data."""
    
    @pytest.fixture
    def models_with_real_data(self):
        """Create AssetReturnModels instance with real data."""
        # Check if the Excel file exists
        excel_file = "histretSP.xls"
        if not os.path.exists(excel_file):
            pytest.skip(f"Excel file {excel_file} not found")
        
        models = AssetReturnModels(excel_file)
        models.load_historical_data()
        return models
    
    def test_load_real_historical_data(self, models_with_real_data):
        """Test loading real historical data."""
        models = models_with_real_data
        
        assert models.historical_data is not None
        assert len(models.historical_data) > 50  # Should have many years of data
        assert 'Year' in models.historical_data.columns
        
        # Check that we have data for major asset classes
        for asset_class, col_name in models.asset_columns.items():
            if col_name in models.historical_data.columns:
                # Should have some non-null values
                non_null_count = models.historical_data[col_name].notna().sum()
                assert non_null_count > 10, f"Too few data points for {asset_class}"
    
    def test_train_models_with_real_data(self, models_with_real_data):
        """Test training models with real data."""
        models = models_with_real_data
        
        # Train a few key models
        key_assets = ['sp500', 't_bills', 't_bonds']
        
        for asset_class in key_assets:
            metrics = models.train_model(asset_class, 'random_forest')
            
            # Check that training completed
            assert isinstance(metrics, dict)
            assert 'train_r2' in metrics
            assert 'test_r2' in metrics
            assert 'train_rmse' in metrics
            assert 'test_rmse' in metrics
            
            # Check that model was stored
            assert asset_class in models.models
            assert asset_class in models.scalers
            
            # RMSE should be reasonable (not too high)
            assert metrics['test_rmse'] < 1.0, f"RMSE too high for {asset_class}: {metrics['test_rmse']}"
    
    def test_predictions_with_real_data(self, models_with_real_data):
        """Test making predictions with real data."""
        models = models_with_real_data
        
        # Train S&P 500 model
        models.train_model('sp500')
        
        # Make prediction
        prediction = models.predict_returns('sp500', horizon=1)
        
        # Prediction should be reasonable
        assert isinstance(prediction, (float, int))
        assert -0.5 <= prediction <= 0.5, f"Prediction seems unreasonable: {prediction}"
    
    def test_all_asset_classes_training(self, models_with_real_data):
        """Test training all asset classes with real data."""
        models = models_with_real_data
        
        all_metrics = models.train_all_models('random_forest')
        
        # Should have results for all asset classes
        assert len(all_metrics) == len(models.asset_columns)
        
        # Count successful trainings
        successful = sum(1 for metrics in all_metrics.values() if 'error' not in metrics)
        
        # At least 5 out of 7 should train successfully
        assert successful >= 5, f"Only {successful} models trained successfully"
        
        # Test predictions for successful models
        predictions = models.get_all_predictions()
        successful_predictions = sum(1 for pred in predictions.values() if pred is not None)
        
        assert successful_predictions >= 5, f"Only {successful_predictions} predictions successful"
    
    def test_model_validation_with_real_data(self, models_with_real_data):
        """Test model validation with real data."""
        models = models_with_real_data
        
        # Train S&P 500 model
        models.train_model('sp500')
        
        # Validate with last 3 years
        validation = models.validate_model_accuracy('sp500', test_years=3)
        
        assert 'error' not in validation
        assert 'rmse' in validation
        assert 'mae' in validation
        assert 'direction_accuracy' in validation
        
        # Direction accuracy should be reasonable
        assert 0.0 <= validation['direction_accuracy'] <= 1.0
        
        # Should have predictions and actuals
        assert len(validation['predictions']) == len(validation['actuals'])
        assert len(validation['predictions']) > 0
    
    def test_model_persistence_with_real_data(self, models_with_real_data):
        """Test saving and loading models with real data."""
        models = models_with_real_data
        
        # Train a model
        models.train_model('sp500')
        original_prediction = models.predict_returns('sp500')
        
        # Save models
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            models.save_models(temp_dir)
            
            # Create new instance and load
            new_models = AssetReturnModels("histretSP.xls")
            new_models.load_historical_data()
            new_models.load_models(temp_dir)
            
            # Should be able to make the same prediction
            loaded_prediction = new_models.predict_returns('sp500')
            
            # Predictions should be very close (allowing for small numerical differences)
            assert abs(original_prediction - loaded_prediction) < 1e-10
    
    def test_data_quality_checks(self, models_with_real_data):
        """Test data quality with real data."""
        models = models_with_real_data
        data = models.historical_data
        
        # Check year range
        assert data['Year'].min() >= 1900, "Data should start from reasonable year"
        assert data['Year'].max() >= 2020, "Data should be recent"
        
        # Check for reasonable return values
        for asset_class, col_name in models.asset_columns.items():
            if col_name in data.columns:
                returns = data[col_name].dropna()
                if len(returns) > 0:
                    # Returns should be within reasonable bounds
                    assert returns.min() > -1.0, f"{asset_class} has unreasonable negative returns"
                    assert returns.max() < 5.0, f"{asset_class} has unreasonable positive returns"
                    
                    # Should have some variation
                    assert returns.std() > 0.01, f"{asset_class} has too little variation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])