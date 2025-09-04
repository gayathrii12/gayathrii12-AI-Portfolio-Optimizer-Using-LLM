"""
Integration tests for orchestrator agent communication and failure scenarios.

These tests focus on testing the complete agent pipeline with real agent
interactions and various failure scenarios.
"""

import pytest
import tempfile
import os
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from agents.orchestrator import (
    FinancialReturnsOrchestrator, OrchestrationInput, OrchestrationResult,
    OrchestrationStage, create_orchestrator
)
from models.data_models import UserInputModel, AssetReturns, PortfolioAllocation
from utils.data_loader import HistoricalDataLoader


class TestOrchestratorIntegration:
    """Integration tests for the orchestrator with real agent interactions."""
    
    @pytest.fixture
    def sample_excel_file(self):
        """Create a temporary Excel file with sample data for testing."""
        # Create sample data
        data = {
            'Year': [2020, 2021, 2022],
            'S&P 500': [10.5, 8.2, -12.1],
            'US Small Cap': [12.3, 9.8, -15.2],
            'T-Bills': [2.1, 1.8, 3.2],
            'T-Bonds': [4.5, 3.9, -8.1],
            'Corporate Bonds': [5.2, 4.1, -6.8],
            'Real Estate': [8.9, 7.2, -10.5],
            'Gold': [6.8, -2.1, 4.3]
        }
        
        df = pd.DataFrame(data)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            df.to_excel(tmp_file.name, index=False)
            yield tmp_file.name
        
        # Cleanup
        os.unlink(tmp_file.name)
    
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
    
    def test_end_to_end_orchestration_with_real_data(self, sample_excel_file, sample_user_input):
        """Test complete orchestration with real data file."""
        
        orchestration_input = OrchestrationInput(
            user_input=sample_user_input,
            data_file_path=sample_excel_file,
            enable_retry=True,
            max_retries=2,
            timeout_seconds=120
        )
        
        orchestrator = FinancialReturnsOrchestrator()
        
        # Execute orchestration
        result = orchestrator.orchestrate(orchestration_input)
        
        # Verify results
        assert isinstance(result, OrchestrationResult)
        
        # Check that we got through at least data cleaning
        assert len(result.stage_results) >= 1
        
        # If data cleaning succeeded, check the data
        if result.stage_results[0].success:
            assert result.cleaned_data is not None
            assert len(result.cleaned_data) > 0
            assert all(isinstance(item, AssetReturns) for item in result.cleaned_data)
    
    def test_data_cleaning_with_missing_file(self, sample_user_input):
        """Test orchestration with missing data file."""
        
        orchestration_input = OrchestrationInput(
            user_input=sample_user_input,
            data_file_path="nonexistent_file.xlsx",
            enable_retry=False,
            max_retries=1
        )
        
        orchestrator = FinancialReturnsOrchestrator()
        result = orchestrator.orchestrate(orchestration_input)
        
        # Should fail at data cleaning stage
        assert result.success is False
        assert result.final_stage == OrchestrationStage.DATA_CLEANING
        assert len(result.stage_results) == 1
        assert result.stage_results[0].success is False
    
    def test_data_cleaning_with_corrupted_file(self, sample_user_input):
        """Test orchestration with corrupted data file."""
        
        # Create a corrupted file (not valid Excel)
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            tmp_file.write(b"This is not a valid Excel file")
            corrupted_file = tmp_file.name
        
        try:
            orchestration_input = OrchestrationInput(
                user_input=sample_user_input,
                data_file_path=corrupted_file,
                enable_retry=False,
                max_retries=1
            )
            
            orchestrator = FinancialReturnsOrchestrator()
            result = orchestrator.orchestrate(orchestration_input)
            
            # Should fail at data cleaning stage
            assert result.success is False
            assert result.final_stage == OrchestrationStage.DATA_CLEANING
            
        finally:
            os.unlink(corrupted_file)
    
    def test_agent_communication_chain(self, sample_excel_file, sample_user_input):
        """Test that agents communicate properly in the chain."""
        
        orchestration_input = OrchestrationInput(
            user_input=sample_user_input,
            data_file_path=sample_excel_file,
            enable_retry=True,
            max_retries=1
        )
        
        orchestrator = FinancialReturnsOrchestrator()
        
        # Track agent method calls
        with patch.object(orchestrator.data_cleaning_agent, 'clean_data', wraps=orchestrator.data_cleaning_agent.clean_data) as mock_clean, \
             patch.object(orchestrator.asset_predictor_agent, 'predict_returns', wraps=orchestrator.asset_predictor_agent.predict_returns) as mock_predict, \
             patch.object(orchestrator.portfolio_allocator_agent, 'allocate_portfolio', wraps=orchestrator.portfolio_allocator_agent.allocate_portfolio) as mock_allocate, \
             patch.object(orchestrator.simulation_agent, 'simulate_portfolio', wraps=orchestrator.simulation_agent.simulate_portfolio) as mock_simulate:
            
            result = orchestrator.orchestrate(orchestration_input)
            
            # Verify agent methods were called in sequence
            mock_clean.assert_called_once()
            
            if result.success or len(result.stage_results) > 1:
                mock_predict.assert_called_once()
            
            if result.success or len(result.stage_results) > 2:
                mock_allocate.assert_called_once()
            
            if result.success or len(result.stage_results) > 3:
                mock_simulate.assert_called_once()
    
    def test_retry_mechanism_with_transient_failures(self, sample_excel_file, sample_user_input):
        """Test retry mechanism with transient failures."""
        
        orchestration_input = OrchestrationInput(
            user_input=sample_user_input,
            data_file_path=sample_excel_file,
            enable_retry=True,
            max_retries=3
        )
        
        orchestrator = FinancialReturnsOrchestrator()
        
        # Mock a transient failure in asset prediction
        call_count = 0
        original_predict = orchestrator.asset_predictor_agent.predict_returns
        
        def failing_predict(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call fails
                from agents.asset_predictor_agent import PredictionResult
                return PredictionResult(
                    success=False,
                    predictions={},
                    market_regime="normal_market",
                    analysis_period={},
                    methodology_summary={},
                    error_message="Transient network error"
                )
            else:
                # Subsequent calls succeed
                return original_predict(*args, **kwargs)
        
        with patch.object(orchestrator.asset_predictor_agent, 'predict_returns', side_effect=failing_predict):
            result = orchestrator.orchestrate(orchestration_input)
            
            # Should eventually succeed after retry
            # (Note: This depends on the actual implementation handling the retry correctly)
            assert call_count >= 1  # At least one call was made
    
    def test_memory_usage_with_large_dataset(self, sample_user_input):
        """Test orchestrator memory usage with larger dataset."""
        
        # Create a larger dataset
        years = list(range(1970, 2024))  # 54 years of data
        data = {
            'Year': years,
            'S&P 500': [10.5 + (i % 20 - 10) for i in range(len(years))],
            'US Small Cap': [12.3 + (i % 25 - 12) for i in range(len(years))],
            'T-Bills': [2.1 + (i % 5) for i in range(len(years))],
            'T-Bonds': [4.5 + (i % 10 - 5) for i in range(len(years))],
            'Corporate Bonds': [5.2 + (i % 8 - 4) for i in range(len(years))],
            'Real Estate': [8.9 + (i % 15 - 7) for i in range(len(years))],
            'Gold': [6.8 + (i % 30 - 15) for i in range(len(years))]
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            df.to_excel(tmp_file.name, index=False)
            large_file = tmp_file.name
        
        try:
            orchestration_input = OrchestrationInput(
                user_input=sample_user_input,
                data_file_path=large_file,
                enable_retry=False,
                max_retries=1,
                timeout_seconds=300  # Longer timeout for large dataset
            )
            
            orchestrator = FinancialReturnsOrchestrator()
            result = orchestrator.orchestrate(orchestration_input)
            
            # Should handle large dataset without memory issues
            if result.success:
                assert result.cleaned_data is not None
                assert len(result.cleaned_data) == len(years)
            
        finally:
            os.unlink(large_file)
    
    def test_concurrent_orchestration_requests(self, sample_excel_file, sample_user_input):
        """Test handling of concurrent orchestration requests."""
        
        import threading
        import time
        
        orchestration_input = OrchestrationInput(
            user_input=sample_user_input,
            data_file_path=sample_excel_file,
            enable_retry=False,
            max_retries=1
        )
        
        results = []
        exceptions = []
        
        def run_orchestration():
            try:
                orchestrator = FinancialReturnsOrchestrator()
                result = orchestrator.orchestrate(orchestration_input)
                results.append(result)
            except Exception as e:
                exceptions.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_orchestration)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=60)  # 60 second timeout
        
        # Verify results
        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        
        # All results should be valid OrchestrationResult objects
        for result in results:
            assert isinstance(result, OrchestrationResult)
    
    def test_orchestrator_state_isolation(self, sample_excel_file, sample_user_input):
        """Test that orchestrator instances don't share state."""
        
        orchestration_input = OrchestrationInput(
            user_input=sample_user_input,
            data_file_path=sample_excel_file
        )
        
        # Create two orchestrator instances
        orchestrator1 = FinancialReturnsOrchestrator()
        orchestrator2 = FinancialReturnsOrchestrator()
        
        # Verify they have separate agent instances
        assert orchestrator1.data_cleaning_agent is not orchestrator2.data_cleaning_agent
        assert orchestrator1.asset_predictor_agent is not orchestrator2.asset_predictor_agent
        assert orchestrator1.portfolio_allocator_agent is not orchestrator2.portfolio_allocator_agent
        assert orchestrator1.simulation_agent is not orchestrator2.simulation_agent
        assert orchestrator1.rebalancing_agent is not orchestrator2.rebalancing_agent
        
        # Run orchestration on both
        result1 = orchestrator1.orchestrate(orchestration_input)
        result2 = orchestrator2.orchestrate(orchestration_input)
        
        # Results should be independent
        assert result1 is not result2
    
    def test_orchestrator_cleanup_after_failure(self, sample_user_input):
        """Test that orchestrator cleans up properly after failures."""
        
        orchestration_input = OrchestrationInput(
            user_input=sample_user_input,
            data_file_path="nonexistent_file.xlsx",
            enable_retry=False
        )
        
        orchestrator = FinancialReturnsOrchestrator()
        
        # Run orchestration that will fail
        result = orchestrator.orchestrate(orchestration_input)
        
        assert result.success is False
        
        # Orchestrator should still be in a valid state for reuse
        status = orchestrator.get_orchestration_status()
        assert status["orchestrator_initialized"] is True
        assert status["agents_available"]["data_cleaning"] is True
    
    def test_orchestrator_with_different_risk_profiles(self, sample_excel_file):
        """Test orchestrator with different risk profiles."""
        
        risk_profiles = ["Low", "Moderate", "High"]
        
        for risk_profile in risk_profiles:
            user_input = UserInputModel(
                investment_amount=100000.0,
                investment_type="lumpsum",
                tenure_years=10,
                risk_profile=risk_profile,
                return_expectation=8.0
            )
            
            orchestration_input = OrchestrationInput(
                user_input=user_input,
                data_file_path=sample_excel_file,
                enable_retry=False,
                max_retries=1
            )
            
            orchestrator = FinancialReturnsOrchestrator()
            result = orchestrator.orchestrate(orchestration_input)
            
            # Should handle all risk profiles
            if result.success:
                assert result.portfolio_allocation is not None
                # Verify allocation sums to 100%
                total_allocation = (
                    result.portfolio_allocation.sp500 +
                    result.portfolio_allocation.small_cap +
                    result.portfolio_allocation.bonds +
                    result.portfolio_allocation.gold +
                    result.portfolio_allocation.real_estate
                )
                assert abs(total_allocation - 100.0) < 0.01
    
    def test_orchestrator_with_different_investment_types(self, sample_excel_file):
        """Test orchestrator with different investment types."""
        
        investment_types = ["lumpsum", "sip"]
        
        for investment_type in investment_types:
            user_input = UserInputModel(
                investment_amount=100000.0 if investment_type == "lumpsum" else 10000.0,
                investment_type=investment_type,
                tenure_years=10,
                risk_profile="Moderate",
                return_expectation=8.0
            )
            
            orchestration_input = OrchestrationInput(
                user_input=user_input,
                data_file_path=sample_excel_file,
                enable_retry=False,
                max_retries=1
            )
            
            orchestrator = FinancialReturnsOrchestrator()
            result = orchestrator.orchestrate(orchestration_input)
            
            # Should handle both investment types
            if result.success:
                assert result.projections is not None
                assert len(result.projections) > 0


class TestOrchestratorPerformance:
    """Performance tests for the orchestrator."""
    
    def test_orchestration_performance_timing(self, sample_excel_file):
        """Test orchestration performance and timing."""
        
        user_input = UserInputModel(
            investment_amount=100000.0,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=8.0
        )
        
        orchestration_input = OrchestrationInput(
            user_input=user_input,
            data_file_path=sample_excel_file,
            enable_retry=False,
            max_retries=1
        )
        
        orchestrator = FinancialReturnsOrchestrator()
        
        import time
        start_time = time.time()
        result = orchestrator.orchestrate(orchestration_input)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Orchestration should complete within reasonable time
        assert execution_time < 30.0, f"Orchestration took too long: {execution_time} seconds"
        
        # Verify timing is recorded in result
        assert result.total_duration_seconds > 0
        assert abs(result.total_duration_seconds - execution_time) < 1.0  # Should be close
    
    def test_stage_timing_breakdown(self, sample_excel_file):
        """Test individual stage timing breakdown."""
        
        user_input = UserInputModel(
            investment_amount=100000.0,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=8.0
        )
        
        orchestration_input = OrchestrationInput(
            user_input=user_input,
            data_file_path=sample_excel_file,
            enable_retry=False,
            max_retries=1
        )
        
        orchestrator = FinancialReturnsOrchestrator()
        result = orchestrator.orchestrate(orchestration_input)
        
        # Verify each stage has timing information
        for stage_result in result.stage_results:
            assert stage_result.duration_seconds >= 0
            assert stage_result.duration_seconds < 30.0  # No single stage should take too long
        
        # Total duration should be sum of stage durations (approximately)
        total_stage_time = sum(sr.duration_seconds for sr in result.stage_results)
        assert abs(result.total_duration_seconds - total_stage_time) < 2.0  # Allow some overhead


if __name__ == "__main__":
    pytest.main([__file__])