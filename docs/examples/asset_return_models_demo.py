"""
Demo script for AssetReturnModels class.

This script demonstrates how to use the ML models for asset return prediction
with the historical data from the Excel file.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.asset_return_models import AssetReturnModels

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    print("=== Asset Return Models Demo ===\n")
    
    # Initialize the models
    print("1. Initializing AssetReturnModels...")
    models = AssetReturnModels("histretSP.xls")
    
    # Load historical data
    print("2. Loading historical data from Excel file...")
    try:
        data = models.load_historical_data()
        print(f"   ✓ Loaded {len(data)} years of data from {data['Year'].min()} to {data['Year'].max()}")
        print(f"   ✓ Available asset classes: {list(models.asset_columns.keys())}")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return
    
    # Train models for all asset classes
    print("\n3. Training ML models for all asset classes...")
    try:
        all_metrics = models.train_all_models('random_forest')
        
        print("   Training Results:")
        for asset_class, metrics in all_metrics.items():
            if 'error' in metrics:
                print(f"   ✗ {asset_class}: {metrics['error']}")
            else:
                print(f"   ✓ {asset_class}: Test R² = {metrics['test_r2']:.3f}, RMSE = {metrics['test_rmse']:.3f}")
    except Exception as e:
        print(f"   ✗ Error training models: {e}")
        return
    
    # Make predictions
    print("\n4. Making return predictions for next year...")
    try:
        predictions = models.get_all_predictions()
        
        print("   Predicted Annual Returns:")
        for asset_class, prediction in predictions.items():
            if prediction is not None:
                asset_name = models.asset_columns[asset_class]
                print(f"   • {asset_name}: {prediction:.1%}")
            else:
                print(f"   ✗ {asset_class}: Prediction failed")
    except Exception as e:
        print(f"   ✗ Error making predictions: {e}")
    
    # Validate model accuracy
    print("\n5. Validating model accuracy...")
    try:
        # Test validation on S&P 500 model
        validation = models.validate_model_accuracy('sp500', test_years=5)
        
        if 'error' not in validation:
            print(f"   S&P 500 Model Validation (last 5 years):")
            print(f"   • RMSE: {validation['rmse']:.3f}")
            print(f"   • MAE: {validation['mae']:.3f}")
            print(f"   • Direction Accuracy: {validation['direction_accuracy']:.1%}")
        else:
            print(f"   ✗ Validation error: {validation['error']}")
    except Exception as e:
        print(f"   ✗ Error validating model: {e}")
    
    # Show model summary
    print("\n6. Model Summary:")
    try:
        summary = models.get_model_summary()
        
        for asset_class, info in summary.items():
            print(f"   • {asset_class}: {info['model_type']} trained for {info['asset_name']}")
    except Exception as e:
        print(f"   ✗ Error getting model summary: {e}")
    
    # Save models
    print("\n7. Saving trained models...")
    try:
        models.save_models("saved_models")
        print("   ✓ Models saved to 'saved_models' directory")
    except Exception as e:
        print(f"   ✗ Error saving models: {e}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()