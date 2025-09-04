"""
Integration tests for Langgraph Agent Workflow

Tests the complete workflow coordination and agent integration
using the Langgraph framework.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from agents.langgraph_workflow import FinancialPlanningWorkflow, WorkflowState
from agents.return_prediction_agent import ReturnPredictionAgent
from agents.portfolio_allocation_agent import PortfolioAllocationAgent
from agents.rebalancing_agent import RebalancingAgent
from models.asset_return_models import AssetReturnModels


class TestLanggraphWorkflowIntegration:
    """Test suite for Langgraph workflow integration"""
    
    @pytest.fixture
    def mock_asset_models(self):
        """Create mock asset return models"""
        mock_models = Mock(spec=AssetReturnModels)
        mock_models.predict_returns.return_value = 0.08
        return mock_models
    
    @pytest.fixture
    def sample_input_data(self):
        """Sample input data for workflow testing"""
        return {
            'investment_amount': 100000.0,
            'investment_horizon': 10,
            'risk_profile': 'moderate',
            'investment_type': 'lump_sum'
        }
    
    @pytest.fixture
    def workflow(self, mock_asset_models):
        """Create workflow instance for testing"""
        return FinancialPlanningWorkflow(mock_asset_models)
    
    def test_workflow_initialization(self, mock_asset_models):
        """Test workflow initialization with all components"""
        workflow = FinancialPlanningWorkflow(mock_asset_models)
        
        assert workflow.asset_models == mock_asset_models
        assert isinstance(workflow.return_prediction_agent, ReturnPredictionAgent)
        assert isinstance(workflow.portfolio_allocation_agent, PortfolioAllocationAgent)
        assert isinstance(workflow.rebalancing_agent, RebalancingAgent)
        assert workflow.workflow is not None
        assert workflow.memory is not None
    
    def test_complete_workflow_execution_success(self, workflow, sample_input_data):
        """Test successful execution of complete workflow"""
        # Execute workflow
        result = workflow.execute_workflow(sample_input_data, "test_workflow_1")
        
        # Verify workflow completion
        assert result['workflow_complete'] is True
        assert 'error' not in result or result['error'] is None
        
        # Verify all agent outputs are present
        assert 'predicted_returns' in result
        assert 'portfolio_allocation' in result
        assert 'rebalancing_schedule' in result
        
        # Verify predicted returns structure
        predicted_returns = result['predicted_returns']
        assert isinstance(predicted_returns, dict)
        assert len(predicted_returns) > 0
        
        # Verify portfolio allocation structure
        portfolio_allocation = result['portfolio_allocation']
        assert isinstance(portfolio_allocation, dict)
        assert abs(sum(portfolio_allocation.values()) - 100.0) < 0.01  # Should sum to 100%
        
        # Verify rebalancing schedule structure
        rebalancing_schedule = result['rebalancing_schedule']
        assert isinstance(rebalancing_schedule, list)
        assert len(rebalancing_schedule) > 0
        assert all('year' in event and 'allocation' in event for event in rebalancing_schedule)
    
    def test_workflow_with_different_risk_profiles(self, workflow):
        """Test workflow execution with different risk profiles"""
        risk_profiles = ['low', 'moderate', 'high']
        
        for risk_profile in risk_profiles:
            input_data = {
                'investment_amount': 50000.0,
                'investment_horizon': 15,
                'risk_profile': risk_profile,
                'investment_type': 'lump_sum'
            }
            
            result = workflow.execute_workflow(input_data, f"test_{risk_profile}")
            
            # Verify successful completion
            assert result['workflow_complete'] is True
            assert result.get('risk_level') == risk_profile or result.get('risk_profile') == risk_profile
            
            # Verify allocation matches risk profile expectations
            allocation = result['portfolio_allocation']
            equity_allocation = allocation.get('sp500', 0) + allocation.get('small_cap', 0)
            bond_allocation = (allocation.get('t_bills', 0) + 
                             allocation.get('t_bonds', 0) + 
                             allocation.get('corporate_bonds', 0))
            
            if risk_profile == 'low':
                assert bond_allocation > equity_allocation  # More bonds for low risk
            elif risk_profile == 'high':
                assert equity_allocation > bond_allocation  # More equity for high risk
    
    def test_workflow_with_different_investment_horizons(self, workflow):
        """Test workflow execution with different investment horizons"""
        horizons = [5, 10, 20, 30]
        
        for horizon in horizons:
            input_data = {
                'investment_amount': 75000.0,
                'investment_horizon': horizon,
                'risk_profile': 'moderate',
                'investment_type': 'lump_sum'
            }
            
            result = workflow.execute_workflow(input_data, f"test_horizon_{horizon}")
            
            # Verify successful completion
            assert result['workflow_complete'] is True
            assert result.get('investment_horizon') == horizon
            
            # Verify rebalancing schedule considers horizon
            rebalancing_schedule = result['rebalancing_schedule']
            max_year = max(event['year'] for event in rebalancing_schedule)
            assert max_year <= horizon
    
    @patch('agents.return_prediction_agent.ReturnPredictionAgent.predict_returns')
    def test_workflow_error_handling_return_prediction_failure(self, mock_predict, workflow, sample_input_data):
        """Test workflow error handling when return prediction fails"""
        # Mock return prediction failure
        mock_predict.side_effect = Exception("ML model failure")
        
        result = workflow.execute_workflow(sample_input_data, "test_error_1")
        
        # Verify workflow handles error gracefully
        assert result['workflow_complete'] is True
        assert 'error' in result
        
        # Should still provide fallback results
        assert 'predicted_returns' in result
        assert 'portfolio_allocation' in result
    
    @patch('agents.portfolio_allocation_agent.PortfolioAllocationAgent.allocate_portfolio')
    def test_workflow_error_handling_allocation_failure(self, mock_allocate, workflow, sample_input_data):
        """Test workflow error handling when portfolio allocation fails"""
        # Mock allocation failure
        mock_allocate.side_effect = Exception("Allocation calculation failure")
        
        result = workflow.execute_workflow(sample_input_data, "test_error_2")
        
        # Verify workflow handles error gracefully
        assert result['workflow_complete'] is True
        assert 'error' in result
        
        # Should still provide fallback allocation
        assert 'portfolio_allocation' in result
    
    def test_workflow_state_persistence(self, workflow, sample_input_data):
        """Test workflow state persistence and checkpointing"""
        workflow_id = "test_persistence"
        
        # Execute workflow
        result = workflow.execute_workflow(sample_input_data, workflow_id)
        
        # Verify workflow completed
        assert result['workflow_complete'] is True
        
        # Get workflow status
        status = workflow.get_workflow_status(workflow_id)
        
        # Verify status retrieval works
        assert isinstance(status, dict)
    
    def test_workflow_retry_mechanism(self, workflow, sample_input_data):
        """Test workflow retry mechanism for failed executions"""
        workflow_id = "test_retry"
        
        # First execution
        result1 = workflow.execute_workflow(sample_input_data, workflow_id)
        assert result1['workflow_complete'] is True
        
        # Attempt retry (should work even if original succeeded)
        result2 = workflow.retry_failed_workflow(workflow_id)
        
        # Verify retry mechanism works
        assert isinstance(result2, dict)
    
    def test_workflow_with_sip_investment_type(self, workflow):
        """Test workflow execution with SIP investment type"""
        input_data = {
            'investment_amount': 0,  # Not used for SIP
            'monthly_amount': 5000.0,
            'investment_horizon': 10,
            'risk_profile': 'moderate',
            'investment_type': 'sip'
        }
        
        result = workflow.execute_workflow(input_data, "test_sip")
        
        # Verify successful completion
        assert result['workflow_complete'] is True
        assert result.get('investment_type') == 'sip'
        assert result.get('monthly_amount') == 5000.0
    
    def test_workflow_with_swp_investment_type(self, workflow):
        """Test workflow execution with SWP investment type"""
        input_data = {
            'investment_amount': 500000.0,
            'withdrawal_amount': 2000.0,
            'investment_horizon': 20,
            'risk_profile': 'low',
            'investment_type': 'swp'
        }
        
        result = workflow.execute_workflow(input_data, "test_swp")
        
        # Verify successful completion
        assert result['workflow_complete'] is True
        assert result.get('investment_type') == 'swp'
        assert result.get('withdrawal_amount') == 2000.0
    
    def test_workflow_agent_coordination(self, workflow, sample_input_data):
        """Test that agents are properly coordinated and data flows correctly"""
        result = workflow.execute_workflow(sample_input_data, "test_coordination")
        
        # Verify data flow between agents
        assert result['workflow_complete'] is True
        
        # Return prediction agent output should be used by allocation agent
        predicted_returns = result['predicted_returns']
        portfolio_allocation = result['portfolio_allocation']
        
        # Allocation agent output should be used by rebalancing agent
        rebalancing_schedule = result['rebalancing_schedule']
        
        # Verify initial allocation in rebalancing schedule matches portfolio allocation
        initial_event = next((event for event in rebalancing_schedule if event['year'] == 0), None)
        assert initial_event is not None
        
        # Allocations should be similar (allowing for small differences due to processing)
        for asset in portfolio_allocation:
            if asset in initial_event['allocation']:
                assert abs(portfolio_allocation[asset] - initial_event['allocation'][asset]) < 1.0
    
    def test_workflow_performance_with_large_horizon(self, workflow):
        """Test workflow performance with large investment horizon"""
        input_data = {
            'investment_amount': 1000000.0,
            'investment_horizon': 40,  # Large horizon
            'risk_profile': 'moderate',
            'investment_type': 'lump_sum'
        }
        
        result = workflow.execute_workflow(input_data, "test_large_horizon")
        
        # Verify successful completion even with large horizon
        assert result['workflow_complete'] is True
        
        # Verify rebalancing schedule handles large horizon
        rebalancing_schedule = result['rebalancing_schedule']
        assert len(rebalancing_schedule) > 1  # Should have multiple rebalancing events
        
        # Verify final allocation is different from initial (due to rebalancing)
        initial_allocation = rebalancing_schedule[0]['allocation']
        final_allocation = result['final_allocation']
        
        # At least some assets should have different allocations
        differences = [abs(initial_allocation.get(asset, 0) - final_allocation.get(asset, 0)) 
                      for asset in set(initial_allocation.keys()) | set(final_allocation.keys())]
        assert any(diff > 1.0 for diff in differences)
    
    def test_workflow_input_validation(self, workflow):
        """Test workflow input validation and handling of invalid inputs"""
        invalid_inputs = [
            {'investment_amount': -1000},  # Negative amount
            {'investment_horizon': 0},     # Zero horizon
            {'risk_profile': 'invalid'},   # Invalid risk profile
            {}  # Empty input
        ]
        
        for i, invalid_input in enumerate(invalid_inputs):
            result = workflow.execute_workflow(invalid_input, f"test_invalid_{i}")
            
            # Workflow should complete (with defaults or error handling)
            assert result['workflow_complete'] is True
            
            # Should either succeed with defaults or fail gracefully
            assert 'agent_status' in result


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    pytest.main([__file__, "-v"])