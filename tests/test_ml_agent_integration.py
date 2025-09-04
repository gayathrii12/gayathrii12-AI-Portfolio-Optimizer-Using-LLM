"""
Integration tests for ML Models with Agent Workflow

Tests the complete integration of ML models with the agent workflow,
including model predictions, error handling, and end-to-end functionality.
"""

import pytest
import logging
import tempfile
import shutil
from unittest.mock import Mock, patch
from pathlib import Path

from agents.return_prediction_agent import ReturnPredictionAgent
from agents.portfolio_allocation_agent import PortfolioAllocationAgent
from agents.langgraph_workflow import FinancialPlanningWorkflow
from agents.model_manager import ModelManager
from agents.workflow_factory import WorkflowFactory
from models.asset_return_models import AssetReturnModels


class TestMLAgentIntegration:
    """Test ML model integration with agent workflow."""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_asset_models(self):
        """Create mock asset return models with realistic behavior."""
        mock_models = Mock(spec=AssetReturnModels)
        
        # Mock predict_returns with realistic values
        def mock_predict_returns(asset_class, horizon=1):
            returns = {
                'sp500': 0.10,
                'small_cap': 0.12,
                't_bills': 0.03,
                't_bonds': 0.05,
                'corporate_bonds': 0.06,
                'real_estate': 0.08,
                'gold': 0.07
            }
            return returns.get(asset_class, 0.06)
        
        mock_models.predict_returns.side_effect = mock_predict_returns
        
        # Mock model summary
        mock_models.get_model_summary.return_value = {
            'sp500': {'model_type': 'RandomForestRegressor', 'trained': True},
            'small_cap': {'model_type': 'RandomForestRegressor', 'trained': True},
            't_bills': {'model_type': 'RandomForestRegressor', 'trained': True},
            't_bonds': {'model_type': 'RandomForestRegressor', 'trained': True},
            'corporate_bonds': {'model_type': 'RandomForestRegressor', 'trained': True},
            'real_estate': {'model_type': 'RandomForestRegressor', 'trained': True},
            'gold': {'model_type': 'RandomForestRegressor', 'trained': True}
        }
        
        return mock_models
    
    def test_return_prediction_agent_with_ml_models(self, mock_asset_models):
        """Test return prediction agent with ML models."""
        agent = ReturnPredictionAgent(mock_asset_models)
        
        state = {
            'investment_horizon': 10,
            'asset_classes': ['sp500', 'small_cap', 't_bills', 't_bonds']
        }
        
        # Execute prediction
        result = agent.predict_returns(state)
        
        # Verify ML model was called
        assert mock_asset_models.predict_returns.called
        assert mock_asset_models.predict_returns.call_count == 4
        
        # Verify results
        assert 'predicted_returns' in result
        assert 'confidence_scores' in result
        assert 'prediction_rationale' in result
        assert result['agent_status'] == 'return_prediction_complete'
        
        # Verify predictions are reasonable
        predicted_returns = result['predicted_returns']
        assert len(predicted_returns) == 4
        assert all(0.01 <= return_val <= 0.20 for return_val in predicted_returns.values())
        
        # Verify confidence scores
        confidence_scores = result['confidence_scores']
        assert len(confidence_scores) == 4
        assert all(0.1 <= conf <= 1.0 for conf in confidence_scores.values())
    
    def test_return_prediction_agent_with_model_failures(self, mock_asset_models):
        """Test return prediction agent handling ML model failures."""
        # Mock some models to fail
        def mock_predict_with_failures(asset_class, horizon=1):
            if asset_class in ['sp500', 'small_cap']:
                raise Exception(f"ML model failure for {asset_class}")
            returns = {'t_bills': 0.03, 't_bonds': 0.05}
            return returns.get(asset_class, 0.06)
        
        mock_asset_models.predict_returns.side_effect = mock_predict_with_failures
        
        agent = ReturnPredictionAgent(mock_asset_models)
        state = {
            'investment_horizon': 10,
            'asset_classes': ['sp500', 'small_cap', 't_bills', 't_bonds']
        }
        
        # Execute prediction
        result = agent.predict_returns(state)
        
        # Should still complete successfully with fallback values
        assert result['agent_status'] == 'return_prediction_complete'
        assert 'predicted_returns' in result
        
        predicted_returns = result['predicted_returns']
        assert len(predicted_returns) == 4
        
        # Failed models should use fallback values
        assert predicted_returns['sp500'] == 0.10  # Fallback for sp500
        assert predicted_returns['small_cap'] == 0.11  # Fallback for small_cap
        
        # Successful models should use ML predictions
        assert predicted_returns['t_bills'] == 0.03
        assert predicted_returns['t_bonds'] == 0.05
        
        # Confidence should be lower for fallback values
        confidence_scores = result['confidence_scores']
        assert confidence_scores['sp500'] == 0.3  # Lower confidence for fallback
        assert confidence_scores['small_cap'] == 0.3  # Lower confidence for fallback
    
    def test_portfolio_allocation_with_ml_predictions(self, mock_asset_models):
        """Test portfolio allocation agent using ML model predictions."""
        # Create return prediction agent and get predictions
        return_agent = ReturnPredictionAgent(mock_asset_models)
        allocation_agent = PortfolioAllocationAgent()
        
        # Get ML predictions first
        prediction_state = {
            'investment_horizon': 10,
            'asset_classes': ['sp500', 'small_cap', 't_bills', 't_bonds', 
                            'corporate_bonds', 'real_estate', 'gold']
        }
        prediction_result = return_agent.predict_returns(prediction_state)
        
        # Use predictions for allocation
        allocation_state = {
            'risk_profile': 'moderate',
            'predicted_returns': prediction_result['predicted_returns'],
            'investment_amount': 100000,
            'investment_horizon': 10
        }
        
        # Execute allocation
        allocation_result = allocation_agent.allocate_portfolio(allocation_state)
        
        # Verify allocation completed successfully
        assert allocation_result['agent_status'] == 'portfolio_allocation_complete'
        assert 'portfolio_allocation' in allocation_result
        assert 'expected_portfolio_return' in allocation_result
        
        # Verify allocation sums to 100%
        allocation = allocation_result['portfolio_allocation']
        total_allocation = sum(allocation.values())
        assert abs(total_allocation - 100.0) < 0.01
        
        # Verify expected return is calculated from ML predictions
        expected_return = allocation_result['expected_portfolio_return']
        assert 0.04 <= expected_return <= 0.15  # Reasonable range
    
    def test_complete_workflow_with_ml_models(self, mock_asset_models):
        """Test complete workflow execution with ML models."""
        workflow = FinancialPlanningWorkflow(mock_asset_models)
        
        input_data = {
            'investment_amount': 100000,
            'investment_horizon': 10,
            'risk_profile': 'moderate',
            'investment_type': 'lump_sum'
        }
        
        # Execute complete workflow
        result = workflow.execute_workflow(input_data, workflow_id="test_ml_integration")
        
        # Verify workflow completed successfully
        assert result['workflow_complete'] is True
        assert result['agent_status'] == 'completed_successfully'
        
        # Verify all components are present
        assert 'predicted_returns' in result
        assert 'portfolio_allocation' in result
        assert 'rebalancing_schedule' in result
        
        # Verify ML models were used
        assert mock_asset_models.predict_returns.called
        
        # Verify predictions are reasonable
        predicted_returns = result['predicted_returns']
        assert len(predicted_returns) >= 5  # Should have predictions for multiple assets
        assert all(0.01 <= ret <= 0.20 for ret in predicted_returns.values())
    
    def test_workflow_with_complete_ml_failure(self, mock_asset_models):
        """Test workflow handling when all ML models fail."""
        # Mock all models to fail
        mock_asset_models.predict_returns.side_effect = Exception("Complete ML model failure")
        
        workflow = FinancialPlanningWorkflow(mock_asset_models)
        
        input_data = {
            'investment_amount': 100000,
            'investment_horizon': 10,
            'risk_profile': 'moderate',
            'investment_type': 'lump_sum'
        }
        
        # Execute workflow
        result = workflow.execute_workflow(input_data, workflow_id="test_ml_failure")
        
        # Should complete with fallback values
        assert result['workflow_complete'] is True
        assert result['agent_status'] in ['completed_successfully', 'completed_with_errors']
        
        # Should have fallback predictions
        assert 'predicted_returns' in result
        predicted_returns = result['predicted_returns']
        
        # Should use fallback values
        expected_fallbacks = {
            'sp500': 0.10, 'small_cap': 0.11, 't_bills': 0.03,
            't_bonds': 0.05, 'corporate_bonds': 0.06,
            'real_estate': 0.08, 'gold': 0.07
        }
        
        for asset, expected_fallback in expected_fallbacks.items():
            if asset in predicted_returns:
                assert predicted_returns[asset] == expected_fallback
    
    def test_model_manager_initialization(self, temp_model_dir):
        """Test model manager initialization and model loading."""
        with patch('agents.model_manager.os.path.exists') as mock_exists:
            # Mock data file exists
            mock_exists.return_value = True
            
            model_manager = ModelManager("test_data.xls", temp_model_dir)
            
            # Mock the AssetReturnModels
            with patch('agents.model_manager.AssetReturnModels') as mock_models_class:
                mock_instance = Mock()
                mock_models_class.return_value = mock_instance
                
                # Mock successful training
                mock_instance.train_all_models.return_value = {
                    'sp500': {'test_r2': 0.75, 'train_r2': 0.80},
                    'small_cap': {'test_r2': 0.70, 'train_r2': 0.78}
                }
                
                # Initialize models
                success = model_manager.initialize_models()
                
                assert success is True
                assert mock_instance.load_historical_data.called
                assert mock_instance.train_all_models.called
                assert mock_instance.save_models.called
    
    def test_workflow_factory_creation(self):
        """Test workflow factory creates properly initialized workflows."""
        with patch('agents.workflow_factory.get_model_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            
            # Mock successful initialization
            mock_manager.initialize_models.return_value = True
            mock_manager.get_asset_models.return_value = Mock(spec=AssetReturnModels)
            mock_manager.validate_models.return_value = {
                'sp500': {'status': 'valid'},
                'small_cap': {'status': 'valid'}
            }
            
            # Create workflow
            workflow = WorkflowFactory.create_workflow()
            
            assert workflow is not None
            assert mock_manager.initialize_models.called
            assert mock_manager.get_asset_models.called
    
    def test_workflow_factory_model_status(self):
        """Test workflow factory model status reporting."""
        with patch('agents.workflow_factory.get_model_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            
            # Mock model status
            mock_manager.get_asset_models.return_value = Mock(spec=AssetReturnModels)
            mock_manager.get_model_summary.return_value = {'total_models': 7}
            mock_manager.validate_models.return_value = {
                'sp500': {'status': 'valid'},
                'small_cap': {'status': 'valid'},
                't_bills': {'status': 'invalid', 'error': 'Test error'}
            }
            
            # Get status
            status = WorkflowFactory.get_model_status()
            
            assert status['status'] == 'initialized'
            assert status['total_models'] == 3
            assert status['valid_models'] == 2
            assert 'model_summary' in status
            assert 'validation_results' in status
    
    def test_confidence_scoring_with_ml_predictions(self, mock_asset_models):
        """Test confidence scoring for ML model predictions."""
        agent = ReturnPredictionAgent(mock_asset_models)
        
        # Test confidence for different prediction ranges
        test_cases = [
            ('sp500', 0.10, 'good_range'),      # Good equity return
            ('sp500', 0.25, 'high_range'),     # High equity return
            ('sp500', 0.02, 'low_range'),      # Low equity return
            ('t_bills', 0.03, 'good_range'),   # Good bond return
            ('t_bills', 0.15, 'high_range'),   # High bond return
            ('real_estate', 0.08, 'good_range') # Good alternative return
        ]
        
        for asset_class, prediction, expected_range in test_cases:
            confidence = agent._calculate_confidence(asset_class, prediction)
            
            if expected_range == 'good_range':
                assert confidence >= 0.8, f"Expected high confidence for {asset_class} at {prediction}"
            elif expected_range == 'high_range':
                assert confidence <= 0.6, f"Expected lower confidence for {asset_class} at {prediction}"
            elif expected_range == 'low_range':
                assert confidence <= 0.6, f"Expected lower confidence for {asset_class} at {prediction}"
            
            # All confidence scores should be in valid range
            assert 0.1 <= confidence <= 1.0
    
    @pytest.mark.integration
    def test_end_to_end_ml_integration(self, mock_asset_models):
        """Test complete end-to-end ML integration."""
        # This test simulates the complete flow from user input to final results
        workflow = FinancialPlanningWorkflow(mock_asset_models)
        
        # Simulate different user scenarios
        test_scenarios = [
            {
                'name': 'Conservative Investor',
                'input': {
                    'investment_amount': 50000,
                    'investment_horizon': 15,
                    'risk_profile': 'low',
                    'investment_type': 'lump_sum'
                }
            },
            {
                'name': 'Aggressive Investor',
                'input': {
                    'investment_amount': 200000,
                    'investment_horizon': 20,
                    'risk_profile': 'high',
                    'investment_type': 'sip'
                }
            },
            {
                'name': 'Moderate Investor',
                'input': {
                    'investment_amount': 100000,
                    'investment_horizon': 10,
                    'risk_profile': 'moderate',
                    'investment_type': 'lump_sum'
                }
            }
        ]
        
        for scenario in test_scenarios:
            result = workflow.execute_workflow(
                scenario['input'], 
                workflow_id=f"test_{scenario['name'].lower().replace(' ', '_')}"
            )
            
            # Verify successful completion
            assert result['workflow_complete'] is True
            assert result['agent_status'] == 'completed_successfully'
            
            # Verify ML predictions were used
            assert 'predicted_returns' in result
            assert len(result['predicted_returns']) >= 5
            
            # Verify portfolio allocation reflects risk profile
            allocation = result['portfolio_allocation']
            if scenario['input']['risk_profile'] == 'low':
                # Conservative should have more bonds
                bond_allocation = allocation.get('t_bills', 0) + allocation.get('t_bonds', 0) + allocation.get('corporate_bonds', 0)
                assert bond_allocation >= 40  # At least 40% in bonds
            elif scenario['input']['risk_profile'] == 'high':
                # Aggressive should have more equities
                equity_allocation = allocation.get('sp500', 0) + allocation.get('small_cap', 0)
                assert equity_allocation >= 50  # At least 50% in equities
            
            # Verify rebalancing schedule exists
            assert 'rebalancing_schedule' in result
            assert len(result['rebalancing_schedule']) > 0