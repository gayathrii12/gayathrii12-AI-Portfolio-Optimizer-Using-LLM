"""
Integration tests for the main Financial Returns Optimizer application.

This module tests the complete application workflow from user input
processing through orchestration to output generation.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from main import (
    FinancialReturnsOptimizer, ApplicationConfig, 
    get_interactive_input, create_cli_parser
)
from models.data_models import (
    UserInputModel, PortfolioAllocation, ProjectionResult, RiskMetrics
)
from agents.orchestrator import OrchestrationResult, OrchestrationStage


class TestApplicationConfig:
    """Test application configuration management."""
    
    def test_default_config_initialization(self):
        """Test default configuration values."""
        config = ApplicationConfig()
        
        assert config.enable_retry is True
        assert config.max_retries == 3
        assert config.timeout_seconds == 300
        assert "Low" in config.risk_profiles
        assert "Moderate" in config.risk_profiles
        assert "High" in config.risk_profiles
    
    def test_config_update_from_dict(self):
        """Test updating configuration from dictionary."""
        config = ApplicationConfig()
        
        update_dict = {
            'max_retries': 5,
            'timeout_seconds': 600,
            'enable_retry': False
        }
        
        config.update_from_dict(update_dict)
        
        assert config.max_retries == 5
        assert config.timeout_seconds == 600
        assert config.enable_retry is False
    
    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config = ApplicationConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'max_retries' in config_dict
        assert 'timeout_seconds' in config_dict
        assert 'enable_retry' in config_dict
        assert 'risk_profiles' in config_dict


class TestFinancialReturnsOptimizer:
    """Test the main application class."""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ApplicationConfig()
            config.output_directory = Path(temp_dir)
            config.data_file_path = str(Path(__file__).parent.parent / "histretSP.xls")
            yield config
    
    @pytest.fixture
    def app(self, temp_config):
        """Create application instance for testing."""
        return FinancialReturnsOptimizer(temp_config)
    
    @pytest.fixture
    def sample_user_input(self):
        """Sample user input for testing."""
        return {
            "investment_amount": 100000.0,
            "investment_type": "lumpsum",
            "tenure_years": 10,
            "risk_profile": "Moderate",
            "return_expectation": 12.0,
            "rebalancing_preferences": {"frequency": "annual"},
            "withdrawal_preferences": None
        }
    
    @pytest.fixture
    def mock_orchestration_result(self):
        """Mock successful orchestration result."""
        return OrchestrationResult(
            success=True,
            final_stage=OrchestrationStage.COMPLETED,
            total_duration_seconds=45.5,
            stage_results=[],
            portfolio_allocation=PortfolioAllocation(
                sp500=45.0,
                small_cap=7.5,
                bonds=30.0,
                real_estate=12.5,
                gold=5.0
            ),
            projections=[
                ProjectionResult(
                    year=1,
                    portfolio_value=112000.0,
                    annual_return=12.0,
                    cumulative_return=12.0
                ),
                ProjectionResult(
                    year=2,
                    portfolio_value=125440.0,
                    annual_return=12.0,
                    cumulative_return=25.44
                )
            ],
            risk_metrics=RiskMetrics(
                alpha=2.5,
                beta=0.95,
                volatility=15.2,
                sharpe_ratio=0.78,
                max_drawdown=-12.5
            )
        )
    
    def test_initialization(self, app):
        """Test application initialization."""
        assert app.config is not None
        assert app.logger is not None
        assert app.terminal_output is not None
        assert app.json_output is not None
        assert app.orchestrator is not None
    
    def test_validate_user_input_success(self, app, sample_user_input):
        """Test successful user input validation."""
        is_valid, validated_input, error_msg = app.validate_user_input(sample_user_input)
        
        assert is_valid is True
        assert isinstance(validated_input, UserInputModel)
        assert error_msg is None
        assert validated_input.investment_amount == 100000.0
        assert validated_input.risk_profile == "Moderate"
    
    def test_validate_user_input_invalid_amount(self, app):
        """Test user input validation with invalid amount."""
        invalid_input = {
            "investment_amount": -1000.0,  # Invalid negative amount
            "investment_type": "lumpsum",
            "tenure_years": 10,
            "risk_profile": "Moderate",
            "return_expectation": 12.0
        }
        
        is_valid, validated_input, error_msg = app.validate_user_input(invalid_input)
        
        assert is_valid is False
        assert validated_input is None
        assert error_msg is not None
        assert "greater than 0" in error_msg.lower()
    
    def test_validate_user_input_invalid_risk_profile(self, app):
        """Test user input validation with invalid risk profile."""
        invalid_input = {
            "investment_amount": 100000.0,
            "investment_type": "lumpsum",
            "tenure_years": 10,
            "risk_profile": "Invalid",  # Invalid risk profile
            "return_expectation": 12.0
        }
        
        is_valid, validated_input, error_msg = app.validate_user_input(invalid_input)
        
        assert is_valid is False
        assert validated_input is None
        assert error_msg is not None
        assert "validation error" in error_msg.lower() or "should be" in error_msg.lower()
    
    def test_validate_user_input_missing_data_file(self, app):
        """Test user input validation when data file is missing."""
        app.config.data_file_path = "/nonexistent/file.xls"
        
        sample_input = {
            "investment_amount": 100000.0,
            "investment_type": "lumpsum",
            "tenure_years": 10,
            "risk_profile": "Moderate",
            "return_expectation": 12.0
        }
        
        is_valid, validated_input, error_msg = app.validate_user_input(sample_input)
        
        assert is_valid is False
        assert validated_input is None
        assert error_msg is not None
        assert "not found" in error_msg
    
    @patch('main.FinancialReturnsOrchestrator')
    def test_process_investment_request_success(self, mock_orchestrator_class, 
                                              app, sample_user_input, mock_orchestration_result):
        """Test successful investment request processing."""
        # Mock the orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.orchestrate.return_value = mock_orchestration_result
        mock_orchestrator_class.return_value = mock_orchestrator
        app.orchestrator = mock_orchestrator
        
        success, result = app.process_investment_request(sample_user_input)
        
        assert success is True
        assert 'orchestration_result' in result
        assert 'terminal_output' in result
        assert 'json_output' in result
        assert 'output_files' in result
        assert 'processing_time' in result
        assert result['processing_time'] == 45.5
    
    @patch('main.FinancialReturnsOrchestrator')
    def test_process_investment_request_validation_failure(self, mock_orchestrator_class, app):
        """Test investment request processing with validation failure."""
        invalid_input = {
            "investment_amount": -1000.0,  # Invalid
            "investment_type": "lumpsum",
            "tenure_years": 10,
            "risk_profile": "Moderate",
            "return_expectation": 12.0
        }
        
        success, result = app.process_investment_request(invalid_input)
        
        assert success is False
        assert 'error' in result
        assert result['error_type'] == 'validation_error'
    
    @patch('main.FinancialReturnsOrchestrator')
    def test_process_investment_request_orchestration_failure(self, mock_orchestrator_class,
                                                            app, sample_user_input):
        """Test investment request processing with orchestration failure."""
        # Mock failed orchestration
        failed_result = OrchestrationResult(
            success=False,
            final_stage=OrchestrationStage.DATA_CLEANING,
            total_duration_seconds=10.0,
            stage_results=[],
            error_message="Data cleaning failed"
        )
        
        mock_orchestrator = Mock()
        mock_orchestrator.orchestrate.return_value = failed_result
        mock_orchestrator_class.return_value = mock_orchestrator
        app.orchestrator = mock_orchestrator
        
        success, result = app.process_investment_request(sample_user_input)
        
        assert success is False
        assert 'error' in result
        assert result['error_type'] == 'orchestration_error'
        assert 'stage_results' in result
    
    def test_generate_terminal_output(self, app, mock_orchestration_result, sample_user_input):
        """Test terminal output generation."""
        validated_input = UserInputModel(**sample_user_input)
        
        terminal_output = app.generate_terminal_output(mock_orchestration_result, validated_input)
        
        assert isinstance(terminal_output, str)
        assert len(terminal_output) > 0
        assert "PORTFOLIO ALLOCATION" in terminal_output
        assert "PORTFOLIO GROWTH PROJECTIONS" in terminal_output
        assert "RISK ANALYSIS" in terminal_output
    
    def test_generate_terminal_output_incomplete_results(self, app, sample_user_input):
        """Test terminal output generation with incomplete results."""
        incomplete_result = OrchestrationResult(
            success=True,
            final_stage=OrchestrationStage.COMPLETED,
            total_duration_seconds=45.5,
            stage_results=[],
            portfolio_allocation=None,  # Missing
            projections=None,  # Missing
            risk_metrics=None  # Missing
        )
        
        validated_input = UserInputModel(**sample_user_input)
        terminal_output = app.generate_terminal_output(incomplete_result, validated_input)
        
        assert "Error: Incomplete orchestration results" in terminal_output
    
    def test_generate_json_output(self, app, mock_orchestration_result, sample_user_input):
        """Test JSON output generation."""
        validated_input = UserInputModel(**sample_user_input)
        
        json_output = app.generate_json_output(mock_orchestration_result, validated_input)
        
        assert isinstance(json_output, dict)
        assert 'allocation' in json_output
        assert 'projections' in json_output
        assert 'risk_metrics' in json_output
        assert 'metadata' in json_output
    
    def test_generate_json_output_incomplete_results(self, app, sample_user_input):
        """Test JSON output generation with incomplete results."""
        incomplete_result = OrchestrationResult(
            success=True,
            final_stage=OrchestrationStage.COMPLETED,
            total_duration_seconds=45.5,
            stage_results=[],
            portfolio_allocation=None,  # Missing
            projections=None,  # Missing
            risk_metrics=None  # Missing
        )
        
        validated_input = UserInputModel(**sample_user_input)
        json_output = app.generate_json_output(incomplete_result, validated_input)
        
        assert 'error' in json_output
        assert 'Incomplete orchestration results' in json_output['error']
    
    def test_save_outputs(self, app, sample_user_input):
        """Test saving outputs to files."""
        validated_input = UserInputModel(**sample_user_input)
        terminal_output = "Sample terminal output"
        json_output = {"sample": "json output"}
        
        output_files = app.save_outputs(terminal_output, json_output, validated_input)
        
        assert 'terminal_output_file' in output_files
        assert 'json_output_file' in output_files
        
        # Verify files were created
        terminal_path = Path(output_files['terminal_output_file'])
        json_path = Path(output_files['json_output_file'])
        
        assert terminal_path.exists()
        assert json_path.exists()
        
        # Verify file contents
        with open(terminal_path, 'r') as f:
            assert f.read() == terminal_output
        
        with open(json_path, 'r') as f:
            saved_json = json.load(f)
            assert saved_json == json_output
    
    def test_get_sample_input(self, app):
        """Test getting sample input for different risk profiles."""
        for risk_profile in ["Low", "Moderate", "High"]:
            sample_input = app.get_sample_input(risk_profile)
            
            assert isinstance(sample_input, dict)
            assert sample_input['risk_profile'] == risk_profile
            assert sample_input['investment_amount'] > 0
            assert sample_input['tenure_years'] > 0
            assert sample_input['investment_type'] in ['lumpsum', 'sip']


class TestCLIFunctions:
    """Test command-line interface functions."""
    
    def test_create_cli_parser(self):
        """Test CLI parser creation."""
        parser = create_cli_parser()
        
        # Test with sample arguments
        args = parser.parse_args(['--sample', '--risk-profile', 'High'])
        assert args.sample is True
        assert args.risk_profile == 'High'
        
        # Test with full arguments
        args = parser.parse_args([
            '--amount', '50000',
            '--type', 'sip',
            '--years', '15',
            '--risk-profile', 'Low',
            '--return-expectation', '8.5'
        ])
        assert args.amount == 50000.0
        assert args.type == 'sip'
        assert args.years == 15
        assert args.risk_profile == 'Low'
        assert args.return_expectation == 8.5
    
    @patch('builtins.input')
    def test_get_interactive_input_success(self, mock_input):
        """Test successful interactive input collection."""
        # Mock user inputs
        mock_input.side_effect = [
            '75000',      # amount
            '2',          # SIP
            '12',         # years
            '3',          # High risk
            '15.5'        # return expectation
        ]
        
        user_input = get_interactive_input()
        
        assert user_input['investment_amount'] == 75000.0
        assert user_input['investment_type'] == 'sip'
        assert user_input['tenure_years'] == 12
        assert user_input['risk_profile'] == 'High'
        assert user_input['return_expectation'] == 15.5
    
    @patch('builtins.input')
    def test_get_interactive_input_invalid_input(self, mock_input):
        """Test interactive input with invalid values."""
        # Mock invalid input that raises ValueError
        mock_input.side_effect = ValueError("Invalid input")
        
        user_input = get_interactive_input()
        
        # Should return default values when input fails
        assert user_input['investment_amount'] == 100000.0
        assert user_input['risk_profile'] == 'Moderate'


class TestIntegrationWorkflow:
    """Integration tests for complete application workflow."""
    
    @pytest.fixture
    def temp_app(self):
        """Create application with temporary configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ApplicationConfig()
            config.output_directory = Path(temp_dir)
            # Use a mock data file path for testing
            config.data_file_path = str(Path(__file__).parent.parent / "histretSP.xls")
            yield FinancialReturnsOptimizer(config)
    
    @patch('main.FinancialReturnsOrchestrator')
    def test_complete_workflow_success(self, mock_orchestrator_class, temp_app):
        """Test complete workflow from input to output."""
        # Mock successful orchestration
        mock_result = OrchestrationResult(
            success=True,
            final_stage=OrchestrationStage.COMPLETED,
            total_duration_seconds=30.0,
            stage_results=[],
            portfolio_allocation=PortfolioAllocation(
                sp500=50.0, small_cap=20.0, bonds=15.0, real_estate=10.0, gold=5.0
            ),
            projections=[
                ProjectionResult(year=1, portfolio_value=110000.0, annual_return=10.0, cumulative_return=10.0)
            ],
            risk_metrics=RiskMetrics(
                alpha=1.5, beta=1.1, volatility=18.0, sharpe_ratio=0.65, max_drawdown=-15.0
            )
        )
        
        mock_orchestrator = Mock()
        mock_orchestrator.orchestrate.return_value = mock_result
        mock_orchestrator_class.return_value = mock_orchestrator
        temp_app.orchestrator = mock_orchestrator
        
        # Test input
        user_input = {
            "investment_amount": 100000.0,
            "investment_type": "lumpsum",
            "tenure_years": 10,
            "risk_profile": "High",
            "return_expectation": 15.0
        }
        
        # Process request
        success, result = temp_app.process_investment_request(user_input)
        
        # Verify success
        assert success is True
        assert 'terminal_output' in result
        assert 'json_output' in result
        assert 'output_files' in result
        
        # Verify output files were created
        assert Path(result['output_files']['terminal_output_file']).exists()
        assert Path(result['output_files']['json_output_file']).exists()
        
        # Verify orchestrator was called correctly
        mock_orchestrator.orchestrate.assert_called_once()
        call_args = mock_orchestrator.orchestrate.call_args[0][0]
        assert call_args.user_input.investment_amount == 100000.0
        assert call_args.user_input.risk_profile == "High"
    
    def test_error_handling_workflow(self, temp_app):
        """Test error handling throughout the workflow."""
        # Test with invalid input that should fail validation
        invalid_input = {
            "investment_amount": -5000.0,  # Invalid negative amount
            "investment_type": "invalid_type",
            "tenure_years": -5,
            "risk_profile": "Invalid",
            "return_expectation": 150.0  # Unrealistic expectation
        }
        
        success, result = temp_app.process_investment_request(invalid_input)
        
        assert success is False
        assert 'error' in result
        assert result['error_type'] == 'validation_error'
    
    @patch('main.FinancialReturnsOrchestrator')
    def test_partial_failure_workflow(self, mock_orchestrator_class, temp_app):
        """Test workflow with partial orchestration failure."""
        # Mock partial failure (some stages succeed, others fail)
        mock_result = OrchestrationResult(
            success=False,
            final_stage=OrchestrationStage.PORTFOLIO_ALLOCATION,
            total_duration_seconds=15.0,
            stage_results=[],
            error_message="Portfolio allocation failed due to insufficient data"
        )
        
        mock_orchestrator = Mock()
        mock_orchestrator.orchestrate.return_value = mock_result
        mock_orchestrator_class.return_value = mock_orchestrator
        temp_app.orchestrator = mock_orchestrator
        
        user_input = {
            "investment_amount": 50000.0,
            "investment_type": "sip",
            "tenure_years": 5,
            "risk_profile": "Low",
            "return_expectation": 8.0
        }
        
        success, result = temp_app.process_investment_request(user_input)
        
        assert success is False
        assert result['error_type'] == 'orchestration_error'
        assert 'Portfolio allocation failed' in result['error']


if __name__ == "__main__":
    pytest.main([__file__])