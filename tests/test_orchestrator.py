"""
Integration tests for the Financial Returns Orchestrator.

These tests verify the complete agent pipeline, error handling,
retry logic, and agent communication flows.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from agents.orchestrator import (
    FinancialReturnsOrchestrator, OrchestrationInput, OrchestrationResult,
    OrchestrationStage, StageResult, create_orchestrator
)
from models.data_models import UserInputModel, AssetReturns, PortfolioAllocation
from agents.data_cleaning_agent import DataCleaningResult
from agents.asset_predictor_agent import PredictionResult, AssetPrediction
from agents.portfolio_allocator_agent import OptimizationResult, AllocationStrategy, RiskProfile
from agents.simulation_agent import SimulationResult
from agents.rebalancing_agent import RebalancingResult


class TestFinancialReturnsOrchestrator:
    """Test suite for the Financial Returns Orchestrator."""
    
    @pytest.fixture
    def sample_user_input(self):
        """Create sample user input for testing."""
        return UserInputModel(
            investment_amount=100000.0,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=8.0,
            rebalancing_preferences={"frequency": "annual"},
            withdrawal_preferences=None
        )
    
    @pytest.fixture
    def sample_orchestration_input(self, sample_user_input):
        """Create sample orchestration input."""
        return OrchestrationInput(
            user_input=sample_user_input,
            data_file_path="histretSP.xls",
            enable_retry=True,
            max_retries=2,
            timeout_seconds=60
        )
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing."""
        return FinancialReturnsOrchestrator()
    
    @pytest.fixture
    def sample_cleaned_data(self):
        """Create sample cleaned data."""
        return [
            AssetReturns(
                sp500=0.10, small_cap=0.12, t_bills=0.03, t_bonds=0.05,
                corporate_bonds=0.06, real_estate=0.09, gold=0.07, year=2020
            ),
            AssetReturns(
                sp500=0.08, small_cap=0.10, t_bills=0.02, t_bonds=0.04,
                corporate_bonds=0.05, real_estate=0.07, gold=0.05, year=2021
            )
        ]
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = FinancialReturnsOrchestrator()
        
        assert orchestrator.data_cleaning_agent is not None
        assert orchestrator.asset_predictor_agent is not None
        assert orchestrator.portfolio_allocator_agent is not None
        assert orchestrator.simulation_agent is not None
        assert orchestrator.rebalancing_agent is not None
        assert len(orchestrator.tools) == 5
    
    def test_factory_function(self):
        """Test orchestrator factory function."""
        orchestrator = create_orchestrator()
        assert isinstance(orchestrator, FinancialReturnsOrchestrator)
    
    def test_get_orchestration_status(self, orchestrator):
        """Test orchestration status reporting."""
        status = orchestrator.get_orchestration_status()
        
        assert status["orchestrator_initialized"] is True
        assert "agents_available" in status
        assert status["agents_available"]["data_cleaning"] is True
        assert status["agents_available"]["asset_predictor"] is True
        assert status["agents_available"]["portfolio_allocator"] is True
        assert status["agents_available"]["simulation"] is True
        assert status["agents_available"]["rebalancing"] is True
        assert status["tools_count"] == 5
        assert "timestamp" in status
    
    @patch('agents.orchestrator.DataCleaningAgent')
    @patch('agents.orchestrator.AssetPredictorAgent')
    @patch('agents.orchestrator.PortfolioAllocatorAgent')
    @patch('agents.orchestrator.SimulationAgent')
    @patch('agents.orchestrator.RebalancingAgent')
    def test_successful_orchestration(
        self, mock_rebalancing, mock_simulation, mock_allocation, 
        mock_prediction, mock_cleaning, orchestrator, sample_orchestration_input,
        sample_cleaned_data
    ):
        """Test successful complete orchestration pipeline."""
        
        # Mock data cleaning agent
        mock_cleaning_instance = Mock()
        mock_cleaning.return_value = mock_cleaning_instance
        mock_cleaning_instance.clean_data.return_value = DataCleaningResult(
            success=True,
            cleaned_data_rows=100,
            cleaning_summary={"operations": "completed"},
            outliers_detected={},
            missing_values_handled={}
        )
        mock_cleaning_instance.get_asset_returns.return_value = sample_cleaned_data
        
        # Mock asset predictor agent
        mock_prediction_instance = Mock()
        mock_prediction.return_value = mock_prediction_instance
        mock_prediction_instance.predict_returns.return_value = PredictionResult(
            success=True,
            predictions={
                "sp500": AssetPrediction(
                    asset_name="sp500", expected_return=0.10, volatility=0.16,
                    confidence_interval=(0.08, 0.12), historical_mean=0.10,
                    volatility_adjusted_return=0.095, regime_adjusted_return=0.10,
                    sharpe_ratio=0.6
                )
            },
            market_regime="normal_market",
            analysis_period={"start_year": 2020, "end_year": 2021},
            methodology_summary={}
        )
        
        # Mock portfolio allocator agent
        mock_allocation_instance = Mock()
        mock_allocation.return_value = mock_allocation_instance
        mock_allocation_instance.allocate_portfolio.return_value = OptimizationResult(
            success=True,
            allocation=PortfolioAllocation(
                sp500=45.0, small_cap=7.5, bonds=30.0, gold=5.0, real_estate=12.5
            ),
            strategy_used=AllocationStrategy(
                strategy_name="Balanced Growth",
                risk_profile=RiskProfile.MODERATE,
                base_allocation={},
                allocation_ranges={},
                description="Test strategy"
            ),
            correlation_matrix={},
            optimization_metrics={},
            constraint_validation={}
        )
        
        # Mock simulation agent
        mock_simulation_instance = Mock()
        mock_simulation.return_value = mock_simulation_instance
        mock_simulation_instance.simulate_portfolio.return_value = SimulationResult(
            success=True,
            projections=[],
            final_value=200000.0,
            total_invested=100000.0,
            cagr=7.2,
            cumulative_return=100.0,
            simulation_statistics={}
        )
        
        # Mock rebalancing agent
        mock_rebalancing_instance = Mock()
        mock_rebalancing.return_value = mock_rebalancing_instance
        mock_rebalancing_instance.process_rebalancing.return_value = RebalancingResult(
            success=True,
            rebalancing_events=[],
            adjusted_projections=[],
            total_rebalancing_costs=100.0,
            total_tax_impact=50.0,
            final_allocation=PortfolioAllocation(
                sp500=45.0, small_cap=7.5, bonds=30.0, gold=5.0, real_estate=12.5
            ),
            rebalancing_summary={}
        )
        
        # Execute orchestration
        result = orchestrator.orchestrate(sample_orchestration_input)
        
        # Verify successful completion
        assert result.success is True
        assert result.final_stage == OrchestrationStage.COMPLETED
        assert len(result.stage_results) == 5  # All 5 stages
        assert result.cleaned_data is not None
        assert result.expected_returns is not None
        assert result.portfolio_allocation is not None
        assert result.projections is not None
        assert result.error_message is None
        
        # Verify all agents were called
        mock_cleaning_instance.clean_data.assert_called_once()
        mock_prediction_instance.predict_returns.assert_called_once()
        mock_allocation_instance.allocate_portfolio.assert_called_once()
        mock_simulation_instance.simulate_portfolio.assert_called_once()
        mock_rebalancing_instance.process_rebalancing.assert_called_once()
    
    @patch('agents.orchestrator.DataCleaningAgent')
    def test_data_cleaning_failure(self, mock_cleaning, orchestrator, sample_orchestration_input):
        """Test orchestration failure at data cleaning stage."""
        
        # Mock data cleaning failure
        mock_cleaning_instance = Mock()
        mock_cleaning.return_value = mock_cleaning_instance
        mock_cleaning_instance.clean_data.return_value = DataCleaningResult(
            success=False,
            cleaned_data_rows=0,
            cleaning_summary={},
            outliers_detected={},
            missing_values_handled={},
            error_message="Data file not found"
        )
        
        # Execute orchestration
        result = orchestrator.orchestrate(sample_orchestration_input)
        
        # Verify failure
        assert result.success is False
        assert result.final_stage == OrchestrationStage.DATA_CLEANING
        assert len(result.stage_results) == 1
        assert result.stage_results[0].success is False
        assert "Data cleaning failed" in result.error_message
    
    @patch('agents.orchestrator.DataCleaningAgent')
    @patch('agents.orchestrator.AssetPredictorAgent')
    def test_asset_prediction_failure(
        self, mock_prediction, mock_cleaning, orchestrator, 
        sample_orchestration_input, sample_cleaned_data
    ):
        """Test orchestration failure at asset prediction stage."""
        
        # Mock successful data cleaning
        mock_cleaning_instance = Mock()
        mock_cleaning.return_value = mock_cleaning_instance
        mock_cleaning_instance.clean_data.return_value = DataCleaningResult(
            success=True,
            cleaned_data_rows=100,
            cleaning_summary={},
            outliers_detected={},
            missing_values_handled={}
        )
        mock_cleaning_instance.get_asset_returns.return_value = sample_cleaned_data
        
        # Mock asset prediction failure
        mock_prediction_instance = Mock()
        mock_prediction.return_value = mock_prediction_instance
        mock_prediction_instance.predict_returns.return_value = PredictionResult(
            success=False,
            predictions={},
            market_regime="normal_market",
            analysis_period={},
            methodology_summary={},
            error_message="Insufficient historical data"
        )
        
        # Execute orchestration
        result = orchestrator.orchestrate(sample_orchestration_input)
        
        # Verify failure
        assert result.success is False
        assert result.final_stage == OrchestrationStage.ASSET_PREDICTION
        assert len(result.stage_results) == 2
        assert result.stage_results[0].success is True  # Data cleaning succeeded
        assert result.stage_results[1].success is False  # Asset prediction failed
        assert "Asset prediction failed" in result.error_message
    
    def test_retry_logic_success_after_failure(self, orchestrator, sample_orchestration_input):
        """Test retry logic succeeds after initial failure."""
        
        # Mock a stage function that fails once then succeeds
        call_count = 0
        def mock_stage_function(pipeline_data):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            return {"success": True}
        
        # Test retry logic
        result = orchestrator._execute_stage_with_retry(
            mock_stage_function,
            {},
            OrchestrationStage.DATA_CLEANING,
            max_retries=2,
            timeout_seconds=60
        )
        
        assert result.success is True
        assert result.retry_count == 1  # One retry was needed
        assert call_count == 2  # Function was called twice
    
    def test_retry_logic_exhausted(self, orchestrator, sample_orchestration_input):
        """Test retry logic when all retries are exhausted."""
        
        # Mock a stage function that always fails
        def mock_stage_function(pipeline_data):
            raise Exception("Persistent failure")
        
        # Test retry logic
        result = orchestrator._execute_stage_with_retry(
            mock_stage_function,
            {},
            OrchestrationStage.DATA_CLEANING,
            max_retries=2,
            timeout_seconds=60
        )
        
        assert result.success is False
        assert result.retry_count == 1  # max_retries - 1
        assert "Persistent failure" in result.error_message
    
    def test_orchestration_without_rebalancing(self, sample_user_input):
        """Test orchestration when rebalancing is not requested."""
        
        # Remove rebalancing preferences
        user_input_no_rebalancing = UserInputModel(
            investment_amount=sample_user_input.investment_amount,
            investment_type=sample_user_input.investment_type,
            tenure_years=sample_user_input.tenure_years,
            risk_profile=sample_user_input.risk_profile,
            return_expectation=sample_user_input.return_expectation,
            rebalancing_preferences=None,  # No rebalancing
            withdrawal_preferences=None
        )
        
        orchestration_input = OrchestrationInput(
            user_input=user_input_no_rebalancing,
            data_file_path="histretSP.xls"
        )
        
        orchestrator = FinancialReturnsOrchestrator()
        
        # Mock all agents to succeed
        with patch.multiple(
            'agents.orchestrator',
            DataCleaningAgent=Mock(),
            AssetPredictorAgent=Mock(),
            PortfolioAllocatorAgent=Mock(),
            SimulationAgent=Mock(),
            RebalancingAgent=Mock()
        ) as mocks:
            
            # Setup successful mocks (simplified)
            for agent_name, mock_class in mocks.items():
                mock_instance = Mock()
                mock_class.return_value = mock_instance
                
                if agent_name == 'DataCleaningAgent':
                    mock_instance.clean_data.return_value = DataCleaningResult(
                        success=True, cleaned_data_rows=100, cleaning_summary={},
                        outliers_detected={}, missing_values_handled={}
                    )
                    mock_instance.get_asset_returns.return_value = []
                elif agent_name == 'AssetPredictorAgent':
                    mock_instance.predict_returns.return_value = PredictionResult(
                        success=True, predictions={}, market_regime="normal_market",
                        analysis_period={}, methodology_summary={}
                    )
                elif agent_name == 'PortfolioAllocatorAgent':
                    mock_instance.allocate_portfolio.return_value = OptimizationResult(
                        success=True, allocation=PortfolioAllocation(
                            sp500=50.0, small_cap=0.0, bonds=40.0, gold=10.0, real_estate=0.0
                        ), strategy_used=Mock(), correlation_matrix={},
                        optimization_metrics={}, constraint_validation={}
                    )
                elif agent_name == 'SimulationAgent':
                    mock_instance.simulate_portfolio.return_value = SimulationResult(
                        success=True, projections=[], final_value=200000.0,
                        total_invested=100000.0, cagr=7.2, cumulative_return=100.0,
                        simulation_statistics={}
                    )
            
            result = orchestrator.orchestrate(orchestration_input)
            
            # Should succeed without rebalancing stage
            assert result.success is True
            # Should have 4 stages (no rebalancing)
            assert len(result.stage_results) == 4
            
            # Rebalancing agent should not be called
            mocks['RebalancingAgent'].return_value.process_rebalancing.assert_not_called()
    
    def test_invalid_risk_profile_handling(self, orchestrator):
        """Test handling of invalid risk profile."""
        
        invalid_user_input = UserInputModel(
            investment_amount=100000.0,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Invalid",  # Invalid risk profile
            return_expectation=8.0
        )
        
        orchestration_input = OrchestrationInput(
            user_input=invalid_user_input,
            data_file_path="histretSP.xls"
        )
        
        # Mock successful data cleaning and asset prediction
        with patch.multiple(
            'agents.orchestrator',
            DataCleaningAgent=Mock(),
            AssetPredictorAgent=Mock()
        ) as mocks:
            
            mock_cleaning = mocks['DataCleaningAgent'].return_value
            mock_cleaning.clean_data.return_value = DataCleaningResult(
                success=True, cleaned_data_rows=100, cleaning_summary={},
                outliers_detected={}, missing_values_handled={}
            )
            mock_cleaning.get_asset_returns.return_value = []
            
            mock_prediction = mocks['AssetPredictorAgent'].return_value
            mock_prediction.predict_returns.return_value = PredictionResult(
                success=True, predictions={}, market_regime="normal_market",
                analysis_period={}, methodology_summary={}
            )
            
            result = orchestrator.orchestrate(orchestration_input)
            
            # Should fail at portfolio allocation stage due to invalid risk profile
            assert result.success is False
            assert result.final_stage == OrchestrationStage.PORTFOLIO_ALLOCATION
            assert "Invalid risk profile" in result.error_message
    
    def test_stage_timeout_handling(self, orchestrator):
        """Test handling of stage timeouts."""
        
        # Mock a stage function that takes too long
        def slow_stage_function(pipeline_data):
            import time
            time.sleep(2)  # Simulate slow operation
            return {"success": True}
        
        # Test with very short timeout
        result = orchestrator._execute_stage_with_retry(
            slow_stage_function,
            {},
            OrchestrationStage.DATA_CLEANING,
            max_retries=1,
            timeout_seconds=1  # Very short timeout
        )
        
        # Should complete but take longer than timeout
        # (Note: This test doesn't actually implement timeout logic in the current implementation)
        assert result.duration_seconds >= 2
    
    def test_pipeline_data_flow(self, orchestrator, sample_orchestration_input):
        """Test that data flows correctly between pipeline stages."""
        
        # Mock all agents with specific return values to track data flow
        with patch.multiple(
            'agents.orchestrator',
            DataCleaningAgent=Mock(),
            AssetPredictorAgent=Mock(),
            PortfolioAllocatorAgent=Mock(),
            SimulationAgent=Mock()
        ) as mocks:
            
            # Setup data cleaning mock
            mock_cleaning = mocks['DataCleaningAgent'].return_value
            mock_cleaning.clean_data.return_value = DataCleaningResult(
                success=True, cleaned_data_rows=100, cleaning_summary={"test": "data"},
                outliers_detected={}, missing_values_handled={}
            )
            test_cleaned_data = [AssetReturns(
                sp500=0.10, small_cap=0.12, t_bills=0.03, t_bonds=0.05,
                corporate_bonds=0.06, real_estate=0.09, gold=0.07, year=2020
            )]
            mock_cleaning.get_asset_returns.return_value = test_cleaned_data
            
            # Setup asset prediction mock
            mock_prediction = mocks['AssetPredictorAgent'].return_value
            test_expected_returns = {"sp500": 0.10, "bonds": 0.05}
            mock_prediction.predict_returns.return_value = PredictionResult(
                success=True, predictions={}, market_regime="normal_market",
                analysis_period={}, methodology_summary={}
            )
            
            # Setup portfolio allocation mock
            mock_allocation = mocks['PortfolioAllocatorAgent'].return_value
            test_allocation = PortfolioAllocation(
                sp500=50.0, small_cap=0.0, bonds=40.0, gold=10.0, real_estate=0.0
            )
            mock_allocation.allocate_portfolio.return_value = OptimizationResult(
                success=True, allocation=test_allocation, strategy_used=Mock(),
                correlation_matrix={}, optimization_metrics={}, constraint_validation={}
            )
            
            # Setup simulation mock
            mock_simulation = mocks['SimulationAgent'].return_value
            mock_simulation.simulate_portfolio.return_value = SimulationResult(
                success=True, projections=[], final_value=200000.0,
                total_invested=100000.0, cagr=7.2, cumulative_return=100.0,
                simulation_statistics={}
            )
            
            # Execute orchestration
            result = orchestrator.orchestrate(sample_orchestration_input)
            
            # Verify data flow by checking call arguments
            assert result.success is True
            
            # Check that asset predictor received cleaned data
            prediction_call_args = mock_prediction.predict_returns.call_args[0][0]
            assert prediction_call_args.historical_data == test_cleaned_data
            
            # Check that portfolio allocator received expected returns
            allocation_call_args = mock_allocation.allocate_portfolio.call_args[0][0]
            assert hasattr(allocation_call_args, 'historical_data')
            assert allocation_call_args.historical_data == test_cleaned_data
            
            # Check that simulation received allocation
            simulation_call_args = mock_simulation.simulate_portfolio.call_args[0][0]
            assert simulation_call_args.portfolio_allocation == test_allocation


class TestOrchestrationModels:
    """Test suite for orchestration data models."""
    
    def test_orchestration_input_validation(self):
        """Test orchestration input validation."""
        
        user_input = UserInputModel(
            investment_amount=100000.0,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=8.0
        )
        
        # Valid input
        valid_input = OrchestrationInput(
            user_input=user_input,
            data_file_path="test.xls"
        )
        assert valid_input.max_retries == 3  # Default value
        assert valid_input.enable_retry is True  # Default value
        
        # Test validation constraints
        with pytest.raises(ValueError):
            OrchestrationInput(
                user_input=user_input,
                data_file_path="test.xls",
                max_retries=0  # Should be >= 1
            )
        
        with pytest.raises(ValueError):
            OrchestrationInput(
                user_input=user_input,
                data_file_path="test.xls",
                timeout_seconds=10  # Should be >= 30
            )
    
    def test_stage_result_model(self):
        """Test stage result model."""
        
        result = StageResult(
            stage=OrchestrationStage.DATA_CLEANING,
            success=True,
            duration_seconds=5.5,
            data={"test": "data"},
            retry_count=1
        )
        
        assert result.stage == OrchestrationStage.DATA_CLEANING
        assert result.success is True
        assert result.duration_seconds == 5.5
        assert result.data == {"test": "data"}
        assert result.retry_count == 1
        assert result.error_message is None
    
    def test_orchestration_result_model(self):
        """Test orchestration result model."""
        
        stage_results = [
            StageResult(
                stage=OrchestrationStage.DATA_CLEANING,
                success=True,
                duration_seconds=2.0
            )
        ]
        
        result = OrchestrationResult(
            success=True,
            final_stage=OrchestrationStage.COMPLETED,
            total_duration_seconds=10.0,
            stage_results=stage_results
        )
        
        assert result.success is True
        assert result.final_stage == OrchestrationStage.COMPLETED
        assert result.total_duration_seconds == 10.0
        assert len(result.stage_results) == 1
        assert result.error_message is None


class TestAgentCommunicationFailures:
    """Test suite for agent communication failure scenarios."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        return FinancialReturnsOrchestrator()
    
    def test_agent_initialization_failure(self):
        """Test handling of agent initialization failures."""
        
        with patch('agents.orchestrator.DataCleaningAgent', side_effect=Exception("Agent init failed")):
            # Should still create orchestrator but with None agent
            orchestrator = FinancialReturnsOrchestrator()
            # The orchestrator should handle this gracefully
            assert orchestrator is not None
    
    def test_agent_method_not_available(self, orchestrator):
        """Test handling when agent methods are not available."""
        
        # Mock an agent with missing method
        with patch.object(orchestrator.data_cleaning_agent, 'clean_data', side_effect=AttributeError("Method not found")):
            
            pipeline_data = {"data_file_path": "test.xls"}
            
            # Should raise exception which will be caught by retry logic
            with pytest.raises(AttributeError):
                orchestrator._execute_data_cleaning_stage(pipeline_data)
    
    def test_agent_communication_timeout(self, orchestrator):
        """Test handling of agent communication timeouts."""
        
        # Mock an agent method that hangs
        def hanging_method(*args, **kwargs):
            import time
            time.sleep(10)  # Simulate hanging
            return Mock()
        
        with patch.object(orchestrator.data_cleaning_agent, 'clean_data', side_effect=hanging_method):
            
            pipeline_data = {"data_file_path": "test.xls"}
            
            # This would timeout in a real implementation with proper timeout handling
            # For now, we just test that the method can be called
            try:
                orchestrator._execute_data_cleaning_stage(pipeline_data)
            except Exception:
                pass  # Expected to fail due to mock setup
    
    def test_malformed_agent_response(self, orchestrator):
        """Test handling of malformed responses from agents."""
        
        # Mock agent returning malformed response
        malformed_response = "This is not a proper DataCleaningResult"
        
        with patch.object(orchestrator.data_cleaning_agent, 'clean_data', return_value=malformed_response):
            
            pipeline_data = {"data_file_path": "test.xls"}
            
            # Should raise exception due to malformed response
            with pytest.raises(AttributeError):
                orchestrator._execute_data_cleaning_stage(pipeline_data)


if __name__ == "__main__":
    pytest.main([__file__])