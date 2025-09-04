"""
Historical data loading and preprocessing utilities for the Financial Returns Optimizer.

This module provides functionality to load, clean, and preprocess historical financial
returns data from Excel files, with comprehensive validation and error handling.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
from models.data_models import AssetReturns, ErrorResponse

# Import comprehensive logging system
from utils.logging_config import (
    logging_manager, 
    ComponentType, 
    performance_monitor, 
    error_tracker,
    operation_context,
    log_data_quality
)

# Get component-specific logger
logger = logging_manager.get_logger(ComponentType.DATA_LOADER)


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class HistoricalDataLoader:
    """
    Loads and preprocesses historical financial returns data from Excel files.
    
    This class handles loading data from histretSP.xls file, cleaning missing values,
    validating data integrity, and normalizing returns to consistent format.
    """
    
    def __init__(self, file_path: str = "../../assets/histretSP.xls"):
        """
        Initialize the data loader.
        
        Args:
            file_path: Path to the Excel file containing historical returns data
        """
        self.file_path = Path(file_path)
        self.raw_data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.validation_summary: Dict[str, Any] = {}
        
        # Column mapping for the Excel file
        self.column_mapping = {
            'Year': 'year',
            'S&P 500 (includes dividends)': 'sp500',
            'US Small cap (bottom decile)': 'small_cap',
            '3-month T.Bill': 't_bills',
            'US T. Bond (10-year)': 't_bonds',
            ' Baa Corporate Bond': 'corporate_bonds',
            'Real Estate': 'real_estate',
            'Gold*': 'gold'
        }
    
    @performance_monitor(ComponentType.DATA_LOADER, "load_raw_data")
    @error_tracker(ComponentType.DATA_LOADER, "DataLoadError")
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from Excel file.
        
        Returns:
            DataFrame with raw historical returns data
            
        Raises:
            FileNotFoundError: If Excel file doesn't exist
            DataValidationError: If file structure is invalid
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.file_path}")
        
        try:
            logger.info(f"Loading data from {self.file_path}")
            
            # Load the 'Returns by year' sheet with header at row 18 (0-indexed)
            df = pd.read_excel(
                self.file_path, 
                sheet_name='Returns by year',
                header=18
            )
            
            # Extract only the columns we need (first 8 columns contain the returns data)
            expected_columns = list(self.column_mapping.keys())
            
            # Get the actual column names from the first row
            actual_columns = df.columns.tolist()[:8]
            
            # Create a mapping of actual to expected columns
            if len(actual_columns) < len(expected_columns):
                raise DataValidationError(
                    f"Expected {len(expected_columns)} columns, found {len(actual_columns)}"
                )
            
            # Select and rename columns
            df_subset = df.iloc[:, :8].copy()
            df_subset.columns = expected_columns
            
            # Remove any rows where Year is NaN or not numeric
            df_subset = df_subset.dropna(subset=['Year'])
            df_subset = df_subset[pd.to_numeric(df_subset['Year'], errors='coerce').notna()]
            
            # Convert Year to integer
            df_subset['Year'] = df_subset['Year'].astype(int)
            
            # Convert all other columns to numeric, coercing errors to NaN
            for col in df_subset.columns:
                if col != 'Year':
                    df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')
            
            # Filter for reasonable year range (1900-2030)
            df_subset = df_subset[
                (df_subset['Year'] >= 1900) & (df_subset['Year'] <= 2030)
            ]
            
            if df_subset.empty:
                raise DataValidationError("No valid data found after filtering")
            
            self.raw_data = df_subset
            logger.info(f"Loaded {len(df_subset)} years of data from {df_subset['Year'].min()} to {df_subset['Year'].max()}")
            
            return df_subset
            
        except Exception as e:
            if isinstance(e, (FileNotFoundError, DataValidationError)):
                raise
            raise DataValidationError(f"Error loading Excel file: {str(e)}")
    
    def validate_numeric_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that return values are within reasonable numeric ranges.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results and statistics
        """
        validation_results = {
            'total_rows': len(df),
            'columns_validated': [],
            'outliers_found': {},
            'invalid_values': {},
            'summary_stats': {}
        }
        
        # Define reasonable ranges for annual returns (as decimals)
        reasonable_ranges = {
            'sp500': (-0.9, 2.0),        # -90% to +200%
            'small_cap': (-0.9, 3.0),    # -90% to +300%
            't_bills': (-0.1, 0.3),      # -10% to +30%
            't_bonds': (-0.5, 0.8),      # -50% to +80%
            'corporate_bonds': (-0.5, 0.8), # -50% to +80%
            'real_estate': (-0.8, 1.5),  # -80% to +150%
            'gold': (-0.8, 2.0)          # -80% to +200%
        }
        
        for col in df.columns:
            if col == 'Year':
                continue
                
            if col in self.column_mapping.values():
                mapped_col = col
            else:
                # Find the mapped column name
                mapped_col = self.column_mapping.get(col, col)
            
            if mapped_col in reasonable_ranges:
                min_val, max_val = reasonable_ranges[mapped_col]
                
                # Check for invalid values (NaN, inf, etc.)
                # First convert to numeric, coercing errors to NaN
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                invalid_mask = pd.isna(numeric_col) | np.isinf(numeric_col)
                invalid_count = invalid_mask.sum()
                
                # Check for outliers using the numeric column
                outlier_mask = (numeric_col < min_val) | (numeric_col > max_val)
                outlier_count = outlier_mask.sum()
                
                validation_results['columns_validated'].append(col)
                validation_results['invalid_values'][col] = invalid_count
                validation_results['outliers_found'][col] = outlier_count
                validation_results['summary_stats'][col] = {
                    'mean': numeric_col.mean(),
                    'std': numeric_col.std(),
                    'min': numeric_col.min(),
                    'max': numeric_col.max(),
                    'median': numeric_col.median()
                }
                
                if invalid_count > 0:
                    logger.warning(f"Found {invalid_count} invalid values in {col}")
                
                if outlier_count > 0:
                    logger.warning(f"Found {outlier_count} outliers in {col} (outside range {min_val:.1%} to {max_val:.1%})")
        
        return validation_results
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "forward_fill") -> pd.DataFrame:
        """
        Handle missing values in the dataset using specified strategy.
        
        Args:
            df: DataFrame with potential missing values
            strategy: Strategy to use ('forward_fill', 'interpolate', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        df_cleaned = df.copy()
        
        # Log missing values before cleaning
        missing_counts = df_cleaned.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            logger.info(f"Found {total_missing} missing values across all columns")
            for col, count in missing_counts.items():
                if count > 0:
                    logger.info(f"  {col}: {count} missing values")
        
        if strategy == "forward_fill":
            # Forward fill missing values, then backward fill any remaining
            df_cleaned = df_cleaned.ffill().bfill()
            logger.info("Applied forward fill to missing values")
            
        elif strategy == "interpolate":
            # Linear interpolation for numeric columns
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].interpolate(method='linear')
            logger.info("Applied linear interpolation to missing values")
            
        elif strategy == "drop":
            # Drop rows with any missing values
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned.dropna()
            dropped_rows = initial_rows - len(df_cleaned)
            logger.info(f"Dropped {dropped_rows} rows with missing values")
            
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")
        
        # Log remaining missing values
        remaining_missing = df_cleaned.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"{remaining_missing} missing values remain after {strategy}")
        else:
            logger.info("All missing values have been handled")
        
        return df_cleaned
    
    def normalize_returns_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize returns to consistent annualized decimal format.
        
        Args:
            df: DataFrame with returns data
            
        Returns:
            DataFrame with normalized returns
        """
        df_normalized = df.copy()
        
        # Convert percentage returns to decimal format if needed
        # Check if values appear to be in percentage format (> 1 for positive returns)
        for col in df_normalized.columns:
            if col == 'Year':
                continue
                
            # Check if column contains values that look like percentages
            max_val = df_normalized[col].max()
            min_val = df_normalized[col].min()
            
            # If max value > 5, likely in percentage format (e.g., 43.8 instead of 0.438)
            if max_val > 5 or min_val < -50:
                logger.info(f"Converting {col} from percentage to decimal format")
                df_normalized[col] = df_normalized[col] / 100
        
        # Round to reasonable precision (6 decimal places)
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
        df_normalized[numeric_cols] = df_normalized[numeric_cols].round(6)
        
        logger.info("Normalized returns to decimal format")
        return df_normalized
    
    @performance_monitor(ComponentType.DATA_LOADER, "clean_and_preprocess")
    @error_tracker(ComponentType.DATA_LOADER, "DataCleaningError")
    def clean_and_preprocess(self, missing_value_strategy: str = "forward_fill") -> pd.DataFrame:
        """
        Complete data cleaning and preprocessing pipeline.
        
        Args:
            missing_value_strategy: Strategy for handling missing values
            
        Returns:
            Cleaned and preprocessed DataFrame
        """
        logger.info("Starting data cleaning and preprocessing pipeline")
        
        # Step 1: Load raw data
        if self.raw_data is None:
            self.load_raw_data()
        
        df = self.raw_data.copy()
        
        # Step 2: Validate numeric ranges
        self.validation_summary = self.validate_numeric_ranges(df)
        
        # Step 3: Handle missing values
        df = self.handle_missing_values(df, missing_value_strategy)
        
        # Step 4: Normalize returns format
        df = self.normalize_returns_format(df)
        
        # Step 5: Rename columns to standard format
        df = df.rename(columns=self.column_mapping)
        
        # Step 6: Final validation
        final_validation = self.validate_numeric_ranges(df)
        
        # Step 7: Sort by year
        df = df.sort_values('year').reset_index(drop=True)
        
        self.cleaned_data = df
        
        logger.info(f"Data cleaning complete. Final dataset: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Year range: {df['year'].min()} to {df['year'].max()}")
        
        return df
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """
        Get summary of data cleaning operations performed.
        
        Returns:
            Dictionary with cleaning summary statistics
        """
        if not self.validation_summary:
            return {"error": "No cleaning operations performed yet"}
        
        summary = {
            "file_path": str(self.file_path),
            "total_rows_processed": self.validation_summary.get('total_rows', 0),
            "columns_processed": len(self.validation_summary.get('columns_validated', [])),
            "data_quality": {
                "outliers_by_column": self.validation_summary.get('outliers_found', {}),
                "invalid_values_by_column": self.validation_summary.get('invalid_values', {}),
                "summary_statistics": self.validation_summary.get('summary_stats', {})
            }
        }
        
        if self.cleaned_data is not None:
            summary["final_dataset"] = {
                "rows": len(self.cleaned_data),
                "columns": len(self.cleaned_data.columns),
                "year_range": {
                    "start": int(self.cleaned_data['year'].min()),
                    "end": int(self.cleaned_data['year'].max())
                },
                "column_names": self.cleaned_data.columns.tolist()
            }
        
        return summary
    
    def to_asset_returns_list(self) -> List[AssetReturns]:
        """
        Convert cleaned data to list of AssetReturns model instances.
        
        Returns:
            List of validated AssetReturns objects
        """
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Run clean_and_preprocess() first.")
        
        asset_returns_list = []
        
        for _, row in self.cleaned_data.iterrows():
            try:
                asset_return = AssetReturns(
                    year=int(row['year']),
                    sp500=float(row['sp500']),
                    small_cap=float(row['small_cap']),
                    t_bills=float(row['t_bills']),
                    t_bonds=float(row['t_bonds']),
                    corporate_bonds=float(row['corporate_bonds']),
                    real_estate=float(row['real_estate']),
                    gold=float(row['gold'])
                )
                asset_returns_list.append(asset_return)
            except Exception as e:
                logger.error(f"Error creating AssetReturns for year {row['year']}: {str(e)}")
                continue
        
        logger.info(f"Created {len(asset_returns_list)} validated AssetReturns objects")
        return asset_returns_list


def load_and_clean_historical_data(
    file_path: str = "../../assets/histretSP.xls",
    missing_value_strategy: str = "forward_fill"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to load and clean historical data in one step.
    
    Args:
        file_path: Path to Excel file
        missing_value_strategy: Strategy for handling missing values
        
    Returns:
        Tuple of (cleaned_dataframe, cleaning_summary)
    """
    loader = HistoricalDataLoader(file_path)
    cleaned_data = loader.clean_and_preprocess(missing_value_strategy)
    summary = loader.get_cleaning_summary()
    
    return cleaned_data, summary


def validate_data_integrity(df: pd.DataFrame) -> bool:
    """
    Perform comprehensive data integrity validation.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if data passes all integrity checks
        
    Raises:
        DataValidationError: If critical validation failures are found
    """
    errors = []
    
    # Check for required columns
    required_columns = ['year', 'sp500', 'small_cap', 't_bills', 't_bonds', 
                       'corporate_bonds', 'real_estate', 'gold']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check for duplicate years
    if df['year'].duplicated().any():
        duplicate_years = df[df['year'].duplicated()]['year'].tolist()
        errors.append(f"Duplicate years found: {duplicate_years}")
    
    # Check for reasonable data ranges
    for col in required_columns[1:]:  # Skip 'year'
        if col in df.columns:
            if df[col].min() < -1.0:  # Less than -100% return
                errors.append(f"{col} has returns below -100%")
            if df[col].max() > 10.0:  # More than 1000% return
                errors.append(f"{col} has returns above 1000%")
    
    # Check for sufficient data points
    if len(df) < 10:
        errors.append(f"Insufficient data points: {len(df)} (minimum 10 required)")
    
    if errors:
        error_message = "; ".join(errors)
        raise DataValidationError(f"Data integrity validation failed: {error_message}")
    
    return True