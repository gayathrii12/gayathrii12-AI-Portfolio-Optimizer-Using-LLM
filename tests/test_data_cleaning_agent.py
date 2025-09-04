"""
Unit tests for the Data Cleaning Agent.

Tests cover all data cleaning operations, outlier detection methods,
missing value handling strategies, and error scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from agents.data_cleaning_agent import (
    DataCleaningAgent,
    DataCleaningInput,
    DataCleaningResult,
    LoadDataTool,
    ValidateDataTool,
    DetectOutliersTool,
    HandleMissingValuesTool,
    NormalizeDataTool,
    create_data_cleaning_agent
)
from models.data_models import AssetReturns


class TestDataCleaningInput:
    """Test the DataCleaningInput model."""
    
    def test_default_values(self):
        """Test default values are set correctly."""
        input_data = DataCleaningInput(file_path="test.xls")
        
        assert input_data.file_path == "test.xls"
        assert input_data.missing_value_strategy == "forward_fill"
        assert input_data.outlier_detection_method == "iqr"
        assert input_data.outlier_threshold == 3.0
    
    def test_custom_values(self):
        """Test custom values are accepted."""
        input_data = DataCleaningInput(
            file_path="custom.xls",
            missing_value_strategy="interpolate",
            outlier_detection_method="zscore",
            outlier_threshold=2.5
        )
        
        assert input_data.file_path == "custom.xls"
        assert input_data.missing_value_strategy == "interpolate"
        assert input_data.outlier_detection_method == "zscore"
        assert input_data.outlier_threshold == 2.5


class TestDataCleaningResult:
    """Test the DataCleaningResult model."""
    
    def test_successful_result(self):
        """Test successful cleaning result."""
        result = DataCleaningResult(
            success=True,
            cleaned_data_rows=100,
            cleaning_summary={"test": "data"},
            outliers_detected={"sp500": 2},
            missing_values_handled={"sp500": 1}
        )
        
        assert result.success is True
        assert result.cleaned_data_rows == 100
        assert result.cleaning_summary == {"test": "data"}
        assert result.outliers_detected == {"sp500": 2}
        assert result.missing_values_handled == {"sp500": 1}
        assert result.error_message is None
    
    def test_failed_result(self):
        """Test failed cleaning result."""
        result = DataCleaningResult(
            success=False,
            cleaned_data_rows=0,
            cleaning_summary={},
            outliers_detected={},
            missing_values_handled={},
            error_message="Test error"
        )
        
        assert result.success is False
        assert result.cleaned_data_rows == 0
        assert result.error_message == "Test error"


class TestLoadDataTool:
    """Test the LoadDataTool."""
    
    def test_tool_properties(self):
        """Test tool has correct properties."""
        tool = LoadDataTool()
        
        assert tool.name == "load_data"
        assert "Load raw historical returns data" in tool.description
    
    @patch('agents.data_cleaning_agent.HistoricalDataLoader')
    def test_successful_load(self, mock_loader_class):
        """Test successful data loading."""
        # Mock the loader and its methods
        mock_loader = Mock()
        mock_data = pd.DataFrame({
            'Year': [2020, 2021, 2022],
            'S&P 500': [0.1, 0.2, -0.1],
            'Gold': [0.05, 0.15, 0.25]
        })
        mock_loader.load_raw_data.return_value = mock_data
        mock_loader_class.return_value = mock_loader
        
        tool = LoadDataTool()
        result = tool._run("test.xls")
        
        assert "success" in result
        assert "3" in result  # 3 rows loaded
        mock_loader_class.assert_called_once_with("test.xls")
        mock_loader.load_raw_data.assert_called_once()
    
    @patch('agents.data_cleaning_agent.HistoricalDataLoader')
    def test_failed_load(self, mock_loader_class):
        """Test failed data loading."""
        mock_loader = Mock()
        mock_loader.load_raw_data.side_effect = Exception("File not found")
        mock_loader_class.return_value = mock_loader
        
        tool = LoadDataTool()
        result = tool._run("nonexistent.xls")
        
        assert "Error" in result
        assert "File not found" in result


class TestValidateDataTool:
    """Test the ValidateDataTool."""
    
    def test_tool_properties(self):
        """Test tool has correct properties."""
        tool = ValidateDataTool()
        
        assert tool.name == "validate_data"
        assert "Validate data integrity" in tool.description
    
    def test_validation_run(self):
        """Test validation execution."""
        tool = ValidateDataTool()
        result = tool._run("test_data_info")
        
        assert "validation_completed" in result
        assert "validation_passed" in result


class TestDetectOutliersTool:
    """Test the DetectOutliersTool."""
    
    def test_tool_properties(self):
        """Test tool has correct properties."""
        tool = DetectOutliersTool()
        
        assert tool.name == "detect_outliers"
        assert "Detect outliers" in tool.description
    
    def test_outlier_detection_iqr(self):
        """Test IQR outlier detection."""
        tool = DetectOutliersTool()
        result = tool._run("iqr", 3.0)
        
        assert "iqr" in result
        assert "3.0" in result
        assert "outliers_detected" in result
    
    def test_outlier_detection_zscore(self):
        """Test Z-score outlier detection."""
        tool = DetectOutliersTool()
        result = tool._run("zscore", 2.5)
        
        assert "zscore" in result
        assert "2.5" in result


class TestHandleMissingValuesTool:
    """Test the HandleMissingValuesTool."""
    
    def test_tool_properties(self):
        """Test tool has correct properties."""
        tool = HandleMissingValuesTool()
        
        assert tool.name == "handle_missing_values"
        assert "Handle missing values" in tool.description
    
    def test_forward_fill_strategy(self):
        """Test forward fill strategy."""
        tool = HandleMissingValuesTool()
        result = tool._run("forward_fill")
        
        assert "forward_fill" in result
        assert "missing_values_before" in result
        assert "missing_values_after" in result
    
    def test_interpolate_strategy(self):
        """Test interpolate strategy."""
        tool = HandleMissingValuesTool()
        result = tool._run("interpolate")
        
        assert "interpolate" in result


class TestNormalizeDataTool:
    """Test the NormalizeDataTool."""
    
    def test_tool_properties(self):
        """Test tool has correct properties."""
        tool = NormalizeDataTool()
        
        assert tool.name == "normalize_data"
        assert "Normalize returns data" in tool.description
    
    def test_normalization_run(self):
        """Test normalization execution."""
        tool = NormalizeDataTool()
        result = tool._run("test_data")
        
        assert "normalization_applied" in result
        assert "percentage_to_decimal" in result


class TestDataCleaningAgent:
    """Test the main DataCleaningAgent class."""
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = DataCleaningAgent()
        
        assert agent.llm is None
        assert len(agent.tools) == 5
        assert agent.agent_executor is None
        
        # Check all tools are present
        tool_names = [tool.name for tool in agent.tools]
        expected_tools = [
            "load_data", "validate_data", "detect_outliers", 
            "handle_missing_values", "normalize_data"
        ]
        for expected_tool in expected_tools:
            assert expected_tool in tool_names
    
    def test_agent_initialization_with_llm(self):
        """Test agent initializes with LLM."""
        mock_llm = Mock()
        
        with patch('agents.data_cleaning_agent.create_react_agent') as mock_create_agent, \
             patch('agents.data_cleaning_agent.AgentExecutor') as mock_executor:
            
            agent = DataCleaningAgent(llm=mock_llm)
            
            assert agent.llm == mock_llm
            mock_create_agent.assert_called_once()
            mock_executor.assert_called_once()
    
    def test_detect_outliers_iqr_method(self):
        """Test IQR outlier detection method."""
        agent = DataCleaningAgent()
        
        # Create test data with outliers
        data = pd.DataFrame({
            'Year': [2020, 2021, 2022, 2023, 2024],
            'sp500': [0.1, 0.2, 0.15, 5.0, 0.12],  # 5.0 is an outlier
            'gold': [0.05, 0.06, 0.04, 0.05, 0.07]
        })
        
        outliers = agent._detect_outliers(data, method="iqr", threshold=1.5)
        
        assert 'sp500' in outliers
        assert 'gold' in outliers
        assert outliers['sp500'] >= 1  # Should detect the 5.0 outlier
    
    def test_detect_outliers_zscore_method(self):
        """Test Z-score outlier detection method."""
        agent = DataCleaningAgent()
        
        # Create test data with a more extreme outlier for Z-score detection
        data = pd.DataFrame({
            'Year': [2020, 2021, 2022, 2023, 2024],
            'sp500': [0.1, 0.1, 0.1, 0.1, 5.0],  # 5.0 is a more extreme outlier
            'gold': [0.05, 0.05, 0.05, 0.05, 0.05]
        })
        
        outliers = agent._detect_outliers(data, method="zscore", threshold=1.5)
        
        assert 'sp500' in outliers
        assert 'gold' in outliers
        assert outliers['sp500'] >= 1  # Should detect the 5.0 outlier
    
    def test_detect_outliers_none_method(self):
        """Test no outlier detection."""
        agent = DataCleaningAgent()
        
        data = pd.DataFrame({
            'Year': [2020, 2021, 2022],
            'sp500': [0.1, 0.2, 0.15],
            'gold': [0.05, 0.06, 0.04]
        })
        
        outliers = agent._detect_outliers(data, method="none")
        
        assert outliers['sp500'] == 0
        assert outliers['gold'] == 0
    
    @patch('agents.data_cleaning_agent.HistoricalDataLoader')
    def test_clean_data_success(self, mock_loader_class):
        """Test successful data cleaning pipeline."""
        # Mock the loader and its methods
        mock_loader = Mock()
        
        # Mock raw data
        raw_data = pd.DataFrame({
            'Year': [2020, 2021, 2022],
            'S&P 500': [10.0, 20.0, -10.0],  # Percentage format
            'Gold': [5.0, 15.0, 25.0]
        })
        
        # Mock cleaned data
        cleaned_data = pd.DataFrame({
            'year': [2020, 2021, 2022],
            'sp500': [0.10, 0.20, -0.10],  # Decimal format
            'gold': [0.05, 0.15, 0.25]
        })
        
        # Mock asset returns
        asset_returns = [
            AssetReturns(
                year=2020, sp500=0.10, small_cap=0.08, t_bills=0.02,
                t_bonds=0.05, corporate_bonds=0.06, real_estate=0.07, gold=0.05
            )
        ]
        
        mock_loader.load_raw_data.return_value = raw_data
        mock_loader.validate_numeric_ranges.return_value = {
            'total_rows': 3,
            'invalid_values': {'S&P 500': 0, 'Gold': 0}
        }
        mock_loader.clean_and_preprocess.return_value = cleaned_data
        mock_loader.get_cleaning_summary.return_value = {'test': 'summary'}
        mock_loader.to_asset_returns_list.return_value = asset_returns
        mock_loader_class.return_value = mock_loader
        
        agent = DataCleaningAgent()
        input_params = DataCleaningInput(file_path="test.xls")
        
        result = agent.clean_data(input_params)
        
        assert result.success is True
        assert result.cleaned_data_rows == 3
        assert result.cleaning_summary == {'test': 'summary'}
        assert hasattr(agent, 'cleaned_data')
        assert hasattr(agent, 'asset_returns')
    
    @patch('agents.data_cleaning_agent.HistoricalDataLoader')
    def test_clean_data_failure(self, mock_loader_class):
        """Test data cleaning pipeline failure."""
        mock_loader = Mock()
        mock_loader.load_raw_data.side_effect = Exception("Test error")
        mock_loader_class.return_value = mock_loader
        
        agent = DataCleaningAgent()
        input_params = DataCleaningInput(file_path="test.xls")
        
        result = agent.clean_data(input_params)
        
        assert result.success is False
        assert result.cleaned_data_rows == 0
        assert "Test error" in result.error_message
    
    def test_get_cleaned_data_before_cleaning(self):
        """Test getting cleaned data before cleaning is performed."""
        agent = DataCleaningAgent()
        
        result = agent.get_cleaned_data()
        
        assert result is None
    
    def test_get_asset_returns_before_cleaning(self):
        """Test getting asset returns before cleaning is performed."""
        agent = DataCleaningAgent()
        
        result = agent.get_asset_returns()
        
        assert result is None
    
    def test_generate_cleaning_report_before_cleaning(self):
        """Test generating report before cleaning is performed."""
        agent = DataCleaningAgent()
        
        report = agent.generate_cleaning_report()
        
        assert "No data cleaning has been performed" in report
    
    def test_generate_cleaning_report_after_cleaning(self):
        """Test generating report after cleaning is performed."""
        agent = DataCleaningAgent()
        
        # Mock cleaned data
        agent.cleaned_data = pd.DataFrame({
            'year': [2020, 2021, 2022],
            'sp500': [0.10, 0.20, -0.10],
            'gold': [0.05, 0.15, 0.25]
        })
        
        report = agent.generate_cleaning_report()
        
        assert "DATA CLEANING REPORT" in report
        assert "3 rows processed" in report
        assert "2020 to 2022" in report
        assert "sp500" in report
        assert "gold" in report


class TestCreateDataCleaningAgent:
    """Test the factory function."""
    
    def test_create_agent_without_llm(self):
        """Test creating agent without LLM."""
        agent = create_data_cleaning_agent()
        
        assert isinstance(agent, DataCleaningAgent)
        assert agent.llm is None
    
    def test_create_agent_with_llm(self):
        """Test creating agent with LLM."""
        mock_llm = Mock()
        
        with patch('agents.data_cleaning_agent.create_react_agent'), \
             patch('agents.data_cleaning_agent.AgentExecutor'):
            
            agent = create_data_cleaning_agent(llm=mock_llm)
            
            assert isinstance(agent, DataCleaningAgent)
            assert agent.llm == mock_llm


class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""
    
    def create_sample_excel_data(self, file_path: str):
        """Create a sample Excel file for testing."""
        # Create sample data that mimics the real histretSP.xls structure
        data = {
            'Year': list(range(2020, 2025)),
            'S&P 500 (includes dividends)': [16.26, 28.71, -18.11, 26.29, 24.23],
            'US Small cap (bottom decile)': [19.96, 39.32, -20.44, 28.27, 25.12],
            '3-month T.Bill': [0.37, 0.05, 1.46, 4.76, 5.00],
            'US T. Bond (10-year)': [13.96, -2.32, -12.99, 1.26, -0.96],
            ' Baa Corporate Bond': [9.89, -1.04, -15.76, 5.53, 1.10],
            'Real Estate': [2.12, 40.16, -25.09, 25.82, 11.58],
            'Gold*': [24.60, -3.55, 0.36, 13.09, 27.13]
        }
        
        df = pd.DataFrame(data)
        
        # Create a temporary Excel file
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Add some header rows to mimic the real file structure
            header_df = pd.DataFrame([[''] * len(df.columns)] * 18)
            header_df.to_excel(writer, sheet_name='Returns by year', 
                             index=False, header=False, startrow=0)
            
            # Add the actual data starting at row 18
            df.to_excel(writer, sheet_name='Returns by year', 
                       index=False, header=True, startrow=18)
    
    def test_integration_with_sample_data(self):
        """Test complete integration with sample Excel data."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            try:
                # Create sample Excel file
                self.create_sample_excel_data(tmp_file.name)
                
                # Test the agent with the sample data
                agent = DataCleaningAgent()
                input_params = DataCleaningInput(
                    file_path=tmp_file.name,
                    missing_value_strategy="forward_fill",
                    outlier_detection_method="iqr",
                    outlier_threshold=3.0
                )
                
                result = agent.clean_data(input_params)
                
                # Verify the results
                assert result.success is True
                assert result.cleaned_data_rows == 5
                assert len(result.outliers_detected) > 0
                
                # Verify cleaned data is accessible
                cleaned_data = agent.get_cleaned_data()
                assert cleaned_data is not None
                assert len(cleaned_data) == 5
                assert 'year' in cleaned_data.columns
                
                # Verify asset returns are created
                asset_returns = agent.get_asset_returns()
                assert asset_returns is not None
                assert len(asset_returns) == 5
                assert all(isinstance(ar, AssetReturns) for ar in asset_returns)
                
                # Verify report generation
                report = agent.generate_cleaning_report()
                assert "DATA CLEANING REPORT" in report
                assert "5 rows processed" in report
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_integration_with_missing_values(self):
        """Test integration with data containing missing values."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            try:
                # Create data with missing values
                data = {
                    'Year': [2020, 2021, 2022, 2023, 2024],
                    'S&P 500 (includes dividends)': [16.26, np.nan, -18.11, 26.29, 24.23],
                    'US Small cap (bottom decile)': [19.96, 39.32, np.nan, 28.27, 25.12],
                    '3-month T.Bill': [0.37, 0.05, 1.46, np.nan, 5.00],
                    'US T. Bond (10-year)': [13.96, -2.32, -12.99, 1.26, np.nan],
                    ' Baa Corporate Bond': [9.89, -1.04, -15.76, 5.53, 1.10],
                    'Real Estate': [2.12, 40.16, -25.09, 25.82, 11.58],
                    'Gold*': [24.60, -3.55, 0.36, 13.09, 27.13]
                }
                
                df = pd.DataFrame(data)
                
                with pd.ExcelWriter(tmp_file.name, engine='openpyxl') as writer:
                    header_df = pd.DataFrame([[''] * len(df.columns)] * 18)
                    header_df.to_excel(writer, sheet_name='Returns by year', 
                                     index=False, header=False, startrow=0)
                    df.to_excel(writer, sheet_name='Returns by year', 
                               index=False, header=True, startrow=18)
                
                # Test with different missing value strategies
                strategies = ["forward_fill", "interpolate"]
                
                for strategy in strategies:
                    agent = DataCleaningAgent()
                    input_params = DataCleaningInput(
                        file_path=tmp_file.name,
                        missing_value_strategy=strategy
                    )
                    
                    result = agent.clean_data(input_params)
                    
                    assert result.success is True
                    assert result.cleaned_data_rows == 5
                    
                    # Verify no missing values remain
                    cleaned_data = agent.get_cleaned_data()
                    assert cleaned_data.isnull().sum().sum() == 0
                    
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)


if __name__ == "__main__":
    pytest.main([__file__])