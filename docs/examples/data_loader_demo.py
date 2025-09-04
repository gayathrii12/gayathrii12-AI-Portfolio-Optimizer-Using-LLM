"""
Demo script showing how to use the HistoricalDataLoader for loading and preprocessing
historical financial returns data.
"""

from utils.data_loader import HistoricalDataLoader, load_and_clean_historical_data
import pandas as pd


def main():
    """Demonstrate the data loading and preprocessing functionality."""
    
    print("=== Historical Data Loader Demo ===\n")
    
    # Method 1: Using the HistoricalDataLoader class directly
    print("1. Using HistoricalDataLoader class:")
    print("-" * 40)
    
    loader = HistoricalDataLoader("histretSP.xls")
    
    try:
        # Load and clean the data
        cleaned_data = loader.clean_and_preprocess(missing_value_strategy="forward_fill")
        
        print(f"✓ Successfully loaded {len(cleaned_data)} years of data")
        print(f"✓ Data covers years {cleaned_data['year'].min()} to {cleaned_data['year'].max()}")
        print(f"✓ Columns: {', '.join(cleaned_data.columns)}")
        
        # Get cleaning summary
        summary = loader.get_cleaning_summary()
        print(f"✓ Data quality summary:")
        print(f"  - Total rows processed: {summary['total_rows_processed']}")
        print(f"  - Final dataset rows: {summary['final_dataset']['rows']}")
        
        # Show sample data
        print(f"\n✓ Sample data (first 5 years):")
        print(cleaned_data.head().to_string(index=False))
        
        # Convert to AssetReturns objects
        asset_returns = loader.to_asset_returns_list()
        print(f"\n✓ Created {len(asset_returns)} validated AssetReturns objects")
        print(f"  Example: {asset_returns[0].year} - S&P500: {asset_returns[0].sp500:.4f}, Gold: {asset_returns[0].gold:.4f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    print("\n" + "="*60 + "\n")
    
    # Method 2: Using the convenience function
    print("2. Using convenience function:")
    print("-" * 40)
    
    try:
        cleaned_data, summary = load_and_clean_historical_data(
            file_path="histretSP.xls",
            missing_value_strategy="interpolate"
        )
        
        print(f"✓ Convenience function loaded {len(cleaned_data)} years of data")
        print(f"✓ Summary keys: {list(summary.keys())}")
        
        # Show some statistics
        print(f"\n✓ Asset return statistics (1928-2024):")
        stats = cleaned_data.describe()
        for asset in ['sp500', 'small_cap', 't_bills', 't_bonds', 'gold']:
            mean_return = stats.loc['mean', asset]
            std_return = stats.loc['std', asset]
            print(f"  {asset:15}: Mean={mean_return:7.4f} ({mean_return*100:6.2f}%), Std={std_return:7.4f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    print("\n" + "="*60 + "\n")
    
    # Method 3: Demonstrate different missing value strategies
    print("3. Comparing missing value strategies:")
    print("-" * 40)
    
    strategies = ["forward_fill", "interpolate", "drop"]
    
    for strategy in strategies:
        try:
            loader_test = HistoricalDataLoader("histretSP.xls")
            data = loader_test.clean_and_preprocess(missing_value_strategy=strategy)
            print(f"✓ {strategy:12}: {len(data)} rows, {data.isnull().sum().sum()} missing values")
        except Exception as e:
            print(f"✗ {strategy:12}: Error - {e}")
    
    print("\n✓ Demo completed successfully!")


if __name__ == "__main__":
    main()