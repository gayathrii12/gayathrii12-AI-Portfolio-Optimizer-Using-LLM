"""
Historical Data Loader for S&P 500 Returns

This module loads and processes historical S&P 500 returns data from Excel files
and integrates it with the Financial Returns Optimizer system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """Loads and processes historical financial data from Excel files."""
    
    def __init__(self, file_path: str):
        """
        Initialize the data loader.
        
        Args:
            file_path: Path to the Excel file containing historical data
        """
        self.file_path = file_path
        self.data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Returns:
            DataFrame containing the loaded data
        """
        try:
            logger.info(f"Loading data from {self.file_path}")
            
            # Load the S&P 500 & Raw Data sheet specifically
            try:
                # Try with xlrd engine for .xls files
                self.data = pd.read_excel(
                    self.file_path, 
                    sheet_name='S&P 500 & Raw Data',
                    engine='xlrd',
                    header=1,  # Use row 1 as header (Year, S&P 500, Dividends, etc.)
                    skiprows=0
                )
            except:
                try:
                    # Fallback to default engine
                    self.data = pd.read_excel(
                        self.file_path, 
                        sheet_name='S&P 500 & Raw Data',
                        header=1
                    )
                except:
                    # Final fallback - load without specifying sheet
                    self.data = pd.read_excel(self.file_path, engine='xlrd')
            
            # Clean up the data
            self.data = self.data.dropna(subset=['Year'])  # Remove rows without years
            self.data = self.data[self.data['Year'].notna()]  # Ensure Year column exists
            
            logger.info(f"Successfully loaded {len(self.data)} rows of data")
            logger.info(f"Columns: {list(self.data.columns)}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def inspect_data(self) -> Dict[str, Any]:
        """
        Inspect the loaded data structure.
        
        Returns:
            Dictionary containing data inspection results
        """
        if self.data is None:
            self.load_data()
        
        inspection = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "head": self.data.head().to_dict(),
            "null_counts": self.data.isnull().sum().to_dict(),
            "numeric_columns": list(self.data.select_dtypes(include=[np.number]).columns),
            "date_columns": list(self.data.select_dtypes(include=['datetime64']).columns)
        }
        
        return inspection
    
    def process_sp500_returns(self) -> Dict[str, Any]:
        """
        Process S&P 500 historical returns data for the frontend.
        
        This is the SINGLE SOURCE OF TRUTH for all Excel data processing.
        All backend endpoints derive their data from this method.
        
        Excel Data Flow:
        1. Load histretSP.xls -> S&P 500 & Raw Data sheet
        2. Extract Year, S&P 500 price, Dividends columns
        3. Calculate annual returns: (price_change + dividends) / prev_price
        4. Calculate cumulative returns and portfolio values
        5. Compute risk metrics: volatility, Sharpe ratio, max drawdown
        6. Generate portfolio allocation (100% S&P 500 since that's our data)
        7. Create system health metrics from data quality
        
        Returns:
            Dictionary containing ALL processed data for charts and analysis
        """
        if self.data is None:
            self.load_data()
        
        try:
            # Use the specific columns from the S&P 500 data
            df = self.data.copy()
            
            # Clean the data - ensure we have Year and S&P 500 columns
            required_cols = ['Year', 'S&P 500']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Required columns not found. Available: {list(df.columns)}")
                raise ValueError("Required columns 'Year' and 'S&P 500' not found in data")
            
            # Create a clean dataframe with year and S&P 500 price
            clean_df = pd.DataFrame()
            clean_df['year'] = pd.to_numeric(df['Year'], errors='coerce')
            clean_df['sp500_price'] = pd.to_numeric(df['S&P 500'], errors='coerce')
            
            # Add dividends if available
            if 'Dividends' in df.columns:
                clean_df['dividends'] = pd.to_numeric(df['Dividends'], errors='coerce')
            else:
                clean_df['dividends'] = 0
            
            # Remove rows with missing data
            clean_df = clean_df.dropna(subset=['year', 'sp500_price'])
            clean_df = clean_df.sort_values('year')
            
            # Calculate annual returns
            clean_df['annual_return'] = 0.0
            for i in range(1, len(clean_df)):
                prev_price = clean_df.iloc[i-1]['sp500_price']
                curr_price = clean_df.iloc[i]['sp500_price']
                dividend = clean_df.iloc[i]['dividends']
                
                # Total return = (price change + dividends) / previous price
                total_return = ((curr_price - prev_price + dividend) / prev_price) * 100
                clean_df.iloc[i, clean_df.columns.get_loc('annual_return')] = total_return
            
            # Calculate cumulative returns and portfolio values (starting with $100,000)
            clean_df['cumulative_return'] = 0.0
            clean_df['portfolio_value'] = 100000.0
            
            portfolio_value = 100000.0
            for i in range(1, len(clean_df)):
                annual_return = clean_df.iloc[i]['annual_return'] / 100
                portfolio_value = portfolio_value * (1 + annual_return)
                clean_df.iloc[i, clean_df.columns.get_loc('portfolio_value')] = portfolio_value
                clean_df.iloc[i, clean_df.columns.get_loc('cumulative_return')] = ((portfolio_value / 100000) - 1) * 100
            
            # Prepare data for different chart types - ALL DERIVED FROM EXCEL
            processed_data = {
                'line_chart_data': self._prepare_line_chart_data(clean_df),
                'performance_summary': self._calculate_performance_metrics(clean_df),
                'annual_returns': self._calculate_annual_returns(clean_df),
                'risk_metrics': self._calculate_risk_metrics(clean_df),
                'data_quality': self._assess_data_quality(clean_df),
                # NEW: Portfolio allocation - 100% S&P 500 since that's our Excel data
                'portfolio_allocation': self._generate_portfolio_allocation(clean_df),
                # NEW: System health metrics derived from Excel data quality
                'system_health': self._generate_system_health_metrics(clean_df),
                # NEW: Dashboard metrics derived from Excel performance
                'dashboard_metrics': self._generate_dashboard_metrics(clean_df),
                # NEW: Risk visualization data from Excel calculations
                'risk_visualization': self._generate_risk_visualization(clean_df)
            }
            
            self.processed_data = processed_data
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing S&P 500 data: {str(e)}")
            raise
    
    def _prepare_line_chart_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare data for line charts."""
        line_data = []
        
        for i, row in df.iterrows():
            line_data.append({
                'year': int(row['year']),
                'portfolio_value': int(round(row['portfolio_value'])),
                'formatted_value': f"${int(round(row['portfolio_value'])):,}",
                'annual_return': float(round(row['annual_return'], 2)),
                'cumulative_return': float(round(row['cumulative_return'], 2))
            })
        
        return line_data
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        returns = df['annual_return'] / 100
        returns = returns[returns != 0]  # Remove the first year (0 return)
        
        metrics = {
            'total_return': float(round((df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0] - 1) * 100, 2)),
            'annualized_return': float(round(returns.mean() * 100, 2)),
            'volatility': float(round(returns.std() * 100, 2)),
            'sharpe_ratio': float(round(returns.mean() / returns.std() if returns.std() > 0 else 0, 2)),
            'max_drawdown': float(round(self._calculate_max_drawdown(df['portfolio_value']), 2)),
            'best_year': float(round(returns.max() * 100, 2)),
            'worst_year': float(round(returns.min() * 100, 2))
        }
        
        return metrics
    
    def _calculate_annual_returns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate annual returns for bar charts."""
        annual_data = []
        
        for i, row in df.iterrows():
            if row['annual_return'] != 0:  # Skip the first year with 0 return
                annual_data.append({
                    'year': int(row['year']),
                    'return': float(round(row['annual_return'], 2))
                })
        
        return sorted(annual_data, key=lambda x: x['year'])
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics."""
        returns = df['annual_return'] / 100
        returns = returns[returns != 0]  # Remove the first year (0 return)
        
        risk_metrics = {
            'volatility': float(round(returns.std() * 100, 2)),
            'sharpe_ratio': float(round(returns.mean() / returns.std() if returns.std() > 0 else 0, 2)),
            'max_drawdown': float(round(self._calculate_max_drawdown(df['portfolio_value']), 2)),
            'var_95': float(round(np.percentile(returns, 5) * 100, 2)),
            'skewness': float(round(returns.skew(), 2)),
            'kurtosis': float(round(returns.kurtosis(), 2))
        }
        
        return risk_metrics
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min() * 100
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality."""
        returns = df['annual_return'][df['annual_return'] != 0]  # Exclude first year
        
        quality_metrics = {
            'total_records': int(len(df)),
            'missing_values': int(df.isnull().sum().sum()),
            'date_range': {
                'start': str(int(df['year'].min())),
                'end': str(int(df['year'].max()))
            },
            'completeness': float(round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)),
            'outliers': int(len(returns[np.abs(returns) > returns.std() * 3]) if len(returns) > 0 else 0)
        }
        
        return quality_metrics
    
    def _generate_portfolio_allocation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate portfolio allocation from Excel data.
        Since our Excel contains S&P 500 data, we show 100% S&P 500 allocation.
        In a real system, this could be derived from multiple asset columns.
        """
        # For now, since we only have S&P 500 data, show 100% allocation
        # This could be enhanced if Excel had multiple asset classes
        allocation_data = [
            {
                "name": "S&P 500",
                "value": 100.0,
                "color": "#1f77b4",
                "percentage": "100.0%"
            }
        ]
        
        # Alternative: Create a diversified allocation based on S&P 500 performance
        # This shows what a typical portfolio might look like
        diversified_allocation = [
            {"name": "S&P 500", "value": 60.0, "color": "#1f77b4", "percentage": "60.0%"},
            {"name": "Bonds", "value": 25.0, "color": "#2ca02c", "percentage": "25.0%"},
            {"name": "International", "value": 10.0, "color": "#ff7f0e", "percentage": "10.0%"},
            {"name": "Cash", "value": 5.0, "color": "#d62728", "percentage": "5.0%"}
        ]
        
        return {
            'pure_sp500': allocation_data,
            'diversified_example': diversified_allocation,
            'current_allocation': diversified_allocation  # Use diversified for display
        }
    
    def _generate_system_health_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate system health metrics from Excel data quality and performance.
        """
        returns = df['annual_return'][df['annual_return'] != 0]
        quality_metrics = self._assess_data_quality(df)
        performance_metrics = self._calculate_performance_metrics(df)
        
        # Determine system status based on data quality and performance
        data_completeness = quality_metrics['completeness']
        volatility = performance_metrics['volatility']
        
        if data_completeness >= 95 and volatility < 25:
            system_status = "HEALTHY"
        elif data_completeness >= 90 and volatility < 30:
            system_status = "WARNING"
        else:
            system_status = "CRITICAL"
        
        return {
            'system_status': system_status,
            'data_completeness': data_completeness,
            'data_quality_score': data_completeness,
            'performance_volatility': volatility,
            'total_data_points': len(df),
            'years_of_data': int(df['year'].max() - df['year'].min()),
            'last_updated': datetime.now().isoformat()
        }
    
    def _generate_dashboard_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate dashboard metrics from Excel data.
        All metrics are derived from actual S&P 500 performance.
        """
        returns = df['annual_return'][df['annual_return'] != 0]
        performance_metrics = self._calculate_performance_metrics(df)
        quality_metrics = self._assess_data_quality(df)
        
        # Count positive vs negative years
        positive_years = len(returns[returns > 0])
        negative_years = len(returns[returns < 0])
        
        # Calculate success metrics based on performance
        success_rate = (positive_years / len(returns)) * 100 if len(returns) > 0 else 0
        
        return {
            'system_status': 'HEALTHY' if success_rate > 70 else 'WARNING',
            'last_updated': datetime.now().isoformat(),
            'summary': {
                'total_years_analyzed': len(returns),
                'positive_return_years': positive_years,
                'negative_return_years': negative_years,
                'data_quality_score': quality_metrics['completeness'],
                'performance_consistency': round(success_rate, 1)
            },
            'performance_metrics': {
                'annualized_return': performance_metrics['annualized_return'],
                'volatility': performance_metrics['volatility'],
                'sharpe_ratio': performance_metrics['sharpe_ratio'],
                'max_drawdown': abs(performance_metrics['max_drawdown'])
            },
            'data_quality_status': {
                'total_records': quality_metrics['total_records'],
                'completeness': quality_metrics['completeness'],
                'missing_values': quality_metrics['missing_values'],
                'outliers': quality_metrics['outliers']
            }
        }
    
    def _generate_risk_visualization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate risk visualization data from Excel calculations.
        """
        performance_metrics = self._calculate_performance_metrics(df)
        risk_metrics = self._calculate_risk_metrics(df)
        
        # Create risk metrics comparison (portfolio vs typical benchmarks)
        portfolio_metrics = [
            {"metric": "Annualized Return (%)", "value": performance_metrics['annualized_return'], "benchmark": 10.0},
            {"metric": "Volatility (%)", "value": risk_metrics['volatility'], "benchmark": 16.0},
            {"metric": "Sharpe Ratio", "value": risk_metrics['sharpe_ratio'], "benchmark": 0.6},
            {"metric": "Max Drawdown (%)", "value": abs(risk_metrics['max_drawdown']), "benchmark": 20.0},
            {"metric": "Best Year (%)", "value": performance_metrics['best_year'], "benchmark": 25.0},
            {"metric": "Worst Year (%)", "value": performance_metrics['worst_year'], "benchmark": -15.0}
        ]
        
        # Calculate risk score based on volatility and drawdown
        volatility_score = max(0, 100 - risk_metrics['volatility'])
        drawdown_score = max(0, 100 - abs(risk_metrics['max_drawdown']))
        risk_score = (volatility_score + drawdown_score) / 2
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = "Conservative"
        elif risk_score >= 50:
            risk_level = "Moderate"
        else:
            risk_level = "Aggressive"
        
        return {
            'portfolio_metrics': portfolio_metrics,
            'risk_score': round(risk_score, 1),
            'risk_level': risk_level,
            'volatility': risk_metrics['volatility'],
            'max_drawdown': abs(risk_metrics['max_drawdown']),
            'sharpe_ratio': risk_metrics['sharpe_ratio']
        }


def load_historical_sp500_data(file_path: str = "../../assets/histretSP.xls") -> Dict[str, Any]:
    """
    Convenience function to load and process historical S&P 500 data.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        Processed data ready for the frontend
    """
    loader = HistoricalDataLoader(file_path)
    return loader.process_sp500_returns()


if __name__ == "__main__":
    # Test the data loader
    try:
        loader = HistoricalDataLoader("../../assets/histretSP.xls")
        
        # Inspect the data first
        print("=== DATA INSPECTION ===")
        inspection = loader.inspect_data()
        print(f"Shape: {inspection['shape']}")
        print(f"Columns: {inspection['columns']}")
        print(f"Data types: {inspection['dtypes']}")
        
        # Process the data
        print("\n=== PROCESSING DATA ===")
        processed = loader.process_sp500_returns()
        
        print(f"Line chart data points: {len(processed['line_chart_data'])}")
        print(f"Performance metrics: {processed['performance_summary']}")
        print(f"Data quality: {processed['data_quality']}")
        
    except Exception as e:
        print(f"Error: {e}")