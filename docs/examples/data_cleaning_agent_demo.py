"""
Demo script showing how to use the Data Cleaning Agent for preprocessing
historical financial returns data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.data_cleaning_agent import (
    DataCleaningAgent,
    DataCleaningInput,
    create_data_cleaning_agent
)
import logging

# Configure logging to see the agent's operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    """Demonstrate the Data Cleaning Agent functionality."""
    
    print("=== Data Cleaning Agent Demo ===\n")
    
    # Method 1: Using the DataCleaningAgent class directly
    print("1. Using DataCleaningAgent class:")
    print("-" * 40)
    
    try:
        # Create the agent
        agent = DataCleaningAgent()
        
        # Configure cleaning parameters
        input_params = DataCleaningInput(
            file_path="histretSP.xls",
            missing_value_strategy="forward_fill",
            outlier_detection_method="iqr",
            outlier_threshold=3.0
        )
        
        print(f"✓ Agent created with parameters:")
        print(f"  - File: {input_params.file_path}")
        print(f"  - Missing value strategy: {input_params.missing_value_strategy}")
        print(f"  - Outlier detection: {input_params.outlier_detection_method}")
        print(f"  - Threshold: {input_params.outlier_threshold}")
        
        # Execute the cleaning pipeline
        print(f"\n✓ Starting data cleaning pipeline...")
        result = agent.clean_data(input_params)
        
        if result.success:
            print(f"✓ Data cleaning completed successfully!")
            print(f"  - Rows processed: {result.cleaned_data_rows}")
            print(f"  - Outliers detected: {sum(result.outliers_detected.values())}")
            print(f"  - Missing values handled: {sum(result.missing_values_handled.values())}")
            
            # Show outlier detection results
            print(f"\n✓ Outlier detection results:")
            for asset, count in result.outliers_detected.items():
                if count > 0:
                    print(f"  - {asset}: {count} outliers")
            
            # Show missing value handling results
            print(f"\n✓ Missing value handling results:")
            for asset, count in result.missing_values_handled.items():
                if count > 0:
                    print(f"  - {asset}: {count} values imputed")
            
            # Get cleaned data
            cleaned_data = agent.get_cleaned_data()
            if cleaned_data is not None:
                print(f"\n✓ Cleaned dataset summary:")
                print(f"  - Shape: {cleaned_data.shape}")
                print(f"  - Year range: {cleaned_data['year'].min()} to {cleaned_data['year'].max()}")
                print(f"  - Columns: {', '.join(cleaned_data.columns)}")
                
                # Show sample statistics
                print(f"\n✓ Sample statistics (mean annual returns):")
                for col in cleaned_data.columns:
                    if col != 'year':
                        mean_return = cleaned_data[col].mean()
                        print(f"  - {col:15}: {mean_return:7.4f} ({mean_return*100:6.2f}%)")
            
            # Get AssetReturns objects
            asset_returns = agent.get_asset_returns()
            if asset_returns:
                print(f"\n✓ Created {len(asset_returns)} validated AssetReturns objects")
                print(f"  - Example (year {asset_returns[0].year}):")
                print(f"    S&P500: {asset_returns[0].sp500:.4f}, Gold: {asset_returns[0].gold:.4f}")
            
        else:
            print(f"✗ Data cleaning failed: {result.error_message}")
            return
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    print("\n" + "="*60 + "\n")
    
    # Method 2: Using different cleaning strategies
    print("2. Comparing different cleaning strategies:")
    print("-" * 40)
    
    strategies = ["forward_fill", "interpolate"]
    
    for strategy in strategies:
        try:
            test_agent = DataCleaningAgent()
            test_params = DataCleaningInput(
                file_path="histretSP.xls",
                missing_value_strategy=strategy,
                outlier_detection_method="iqr"
            )
            
            result = test_agent.clean_data(test_params)
            
            if result.success:
                print(f"✓ {strategy:12}: {result.cleaned_data_rows} rows, "
                      f"{sum(result.missing_values_handled.values())} values handled")
            else:
                print(f"✗ {strategy:12}: Failed - {result.error_message}")
                
        except Exception as e:
            print(f"✗ {strategy:12}: Error - {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Method 3: Using different outlier detection methods
    print("3. Comparing outlier detection methods:")
    print("-" * 40)
    
    methods = ["iqr", "zscore", "none"]
    
    for method in methods:
        try:
            test_agent = DataCleaningAgent()
            test_params = DataCleaningInput(
                file_path="histretSP.xls",
                outlier_detection_method=method,
                outlier_threshold=3.0
            )
            
            result = test_agent.clean_data(test_params)
            
            if result.success:
                total_outliers = sum(result.outliers_detected.values())
                print(f"✓ {method:8}: {total_outliers} outliers detected")
            else:
                print(f"✗ {method:8}: Failed - {result.error_message}")
                
        except Exception as e:
            print(f"✗ {method:8}: Error - {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Method 4: Generate comprehensive cleaning report
    print("4. Generating comprehensive cleaning report:")
    print("-" * 40)
    
    try:
        # Use the agent from the first example (it should still have cleaned data)
        report = agent.generate_cleaning_report()
        print(report)
        
    except Exception as e:
        print(f"✗ Error generating report: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Method 5: Using the factory function
    print("5. Using factory function:")
    print("-" * 40)
    
    try:
        # Create agent using factory function
        factory_agent = create_data_cleaning_agent()
        
        input_params = DataCleaningInput(
            file_path="histretSP.xls",
            missing_value_strategy="interpolate",
            outlier_detection_method="zscore",
            outlier_threshold=2.5
        )
        
        result = factory_agent.clean_data(input_params)
        
        if result.success:
            print(f"✓ Factory-created agent processed {result.cleaned_data_rows} rows")
            print(f"✓ Detected {sum(result.outliers_detected.values())} outliers using Z-score method")
        else:
            print(f"✗ Factory agent failed: {result.error_message}")
            
    except Exception as e:
        print(f"✗ Error with factory function: {e}")
    
    print("\n✓ Demo completed successfully!")


if __name__ == "__main__":
    main()