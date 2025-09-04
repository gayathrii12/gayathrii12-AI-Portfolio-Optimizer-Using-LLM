"""
Unit tests for Return Prediction Agent

Tests the return prediction agent functionality including
ML model integration and prediction logic.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from agents.return_prediction_agent import ReturnPredictionAgent, ReturnPredictionInput, ReturnPredictionOutput
from models.asset_return_models import AssetReturnModels


class TestReturnPredictionAgent:
    """Test suite for Return Prediction Agent"""
    
    @pytest.fixture
    def mock_asset_models(self):
        """Create mock asset return models"""
        mock_models = Mock(spec=AssetReturnModels)
        mock_models.predict_returns.return_value = 0.08
        return mock_models
    
    @pytest.fixture
    def agent(self, mock_asset_models):
        """Create return prediction agent for testing"""
        return ReturnPredictionAgent(mock_asset_models)
    
    @pytest.fixture
    def sample_state(self):
        """Sample state for testing"""
        return {
            'investment_horizon': 10,
            'asset_classes': ['sp500', 'small_cap', 't_bills', 't_bonds', 'corporate_bonds', 'real_estate', 'gold']
        }
    
    def test_agent_initialization(self, mock_asset_models):
        """Test agent initialization"""
        agent = ReturnPredictionAgent(mock_asset_models)
        
        assert agent.asset_models == mock_asset_models
        assert agent.name == "return_prediction_agent"
    
    def test_predict_returns_success(self, agent, sample_state, mock_asset_models):
        """Test successful return prediction"""
        # Mock different returns for different assets
        def mock_predict_returns(asset_class, horizon):
            returns = {
                'sp500': 0.10,
                'small_cap': 0.11,
                't_bills': 0.03,
                't_bonds': 0.05,
                'corporate_bonds': 0.06,
                'real_estate': 0.08,
                'gold': 0.07
            }
            return returns.get(asset_class, 0.06)
        
        mock_asset_models.predict_returns.side_effect = mock_predict_returns
        
        # Execute prediction
        result = agent.predict_returns(sample_state)
        
        # Verify successful completion
        assert result['agent_status'] == 'return_prediction_complete'
        assert 'error' not in result
        
        # Verify predicted returns structure
        predicted_returns = result['predicted_returns']
        assert isinstance(predicted_returns, dict)
        assert len(predicted_returns) == 7
        
        # Verify specific predictions
        assert predicted_returns['sp500'] == 0.10
        assert predicted_returns['small_cap'] == 0.11
        assert predicted_returns['t_bills'] == 0.03
        
        # Verify confidence scores
        confidence_scores = result['confidence_scores']
        assert isinstance(confidence_scores, dict)
        assert len(confidence_scores) == 7
        assert all(0 <= score <= 1 for score in confidence_scores.values())
        
        # Verify rationale is provided
        assert 'prediction_rationale' in result
        assert isinstance(result['prediction_rationale'], str)
        assert len(result['prediction_rationale']) > 0
    
    def test_predict_returns_with_defaults(self, agent, mock_asset_models):
        """Test prediction with default parameters"""
        mock_asset_models.predict_returns.return_value = 0.08
        
        # Execute with minimal state
        result = agent.predict_returns({})
        
        # Should use defaults
        assert result['agent_status'] == 'return_prediction_complete'
        assert 'predicted_returns' in result
        
        # Should predict for all default asset classes
        predicted_returns = result['predicted_returns']
        assert len(predicted_returns) == 7
    
    def test_predict_returns_model_failure(self, agent, sample_state, mock_asset_models):
        """Test handling of ML model prediction failures"""
        # Mock model failure for some assets
        def mock_predict_with_failure(asset_class, horizon):
            if asset_class in ['sp500', 'small_cap']:
                raise Exception("Model prediction failed")
            return 0.06
        
        mock_asset_models.predict_returns.side_effect = mock_predict_with_failure
        
        # Execute prediction
        result = agent.predict_returns(sample_state)
        
        # Should complete successfully with fallback values
        assert result['agent_status'] == 'return_prediction_complete'
        
        # Should have predictions for all assets (including fallbacks)
        predicted_returns = result['predicted_returns']
        assert len(predicted_returns) == 7
        
        # Failed assets should have fallback values
        assert predicted_returns['sp500'] == 0.10  # Fallback for SP500
        assert predicted_returns['small_cap'] == 0.11  # Fallback for Small Cap
        
        # Successful assets should have model predictions
        assert predicted_returns['t_bills'] == 0.06
    
    def test_predict_returns_complete_failure(self, agent, sample_state, mock_asset_models):
        """Test handling when all model predictions fail"""
        mock_asset_models.predict_returns.side_effect = Exception("Complete model failure")
        
        # Execute prediction
        result = agent.predict_returns(sample_state)
        
        # Should complete with all fallback values
        assert result['agent_status'] == 'return_prediction_complete'
        
        # Should have fallback predictions for all assets
        predicted_returns = result['predicted_returns']
        assert len(predicted_returns) == 7
        
        # All confidence scores should be low (0.5 for fallback)
        confidence_scores = result['confidence_scores']
        assert all(score == 0.5 for score in confidence_scores.values())
    
    def test_calculate_confidence_equity_assets(self, agent):
        """Test confidence calculation for equity assets"""
        # Test reasonable equity return
        confidence = agent._calculate_confidence('sp500', 0.10)
        assert confidence > 0.7  # Should have high confidence
        
        # Test unreasonable equity return
        confidence = agent._calculate_confidence('sp500', 0.25)
        assert confidence < 0.7  # Should have lower confidence
        
        confidence = agent._calculate_confidence('small_cap', 0.01)
        assert confidence < 0.7  # Should have lower confidence
    
    def test_calculate_confidence_bond_assets(self, agent):
        """Test confidence calculation for bond assets"""
        # Test reasonable bond return
        confidence = agent._calculate_confidence('t_bonds', 0.05)
        assert confidence > 0.7  # Should have high confidence
        
        # Test unreasonable bond return
        confidence = agent._calculate_confidence('t_bills', 0.15)
        assert confidence < 0.7  # Should have lower confidence
    
    def test_calculate_confidence_alternative_assets(self, agent):
        """Test confidence calculation for alternative assets"""
        # Test reasonable alternative asset return
        confidence = agent._calculate_confidence('real_estate', 0.08)
        assert confidence > 0.7  # Should have high confidence
        
        confidence = agent._calculate_confidence('gold', 0.07)
        assert confidence > 0.7  # Should have high confidence
    
    def test_get_fallback_return(self, agent):
        """Test fallback return values"""
        # Test known asset classes
        assert agent._get_fallback_return('sp500') == 0.10
        assert agent._get_fallback_return('small_cap') == 0.11
        assert agent._get_fallback_return('t_bills') == 0.03
        assert agent._get_fallback_return('t_bonds') == 0.05
        assert agent._get_fallback_return('corporate_bonds') == 0.06
        assert agent._get_fallback_return('real_estate') == 0.08
        assert agent._get_fallback_return('gold') == 0.07
        
        # Test unknown asset class
        assert agent._get_fallback_return('unknown_asset') == 0.06
    
    def test_generate_prediction_rationale(self, agent):
        """Test prediction rationale generation"""
        predictions = {
            'sp500': 0.10,
            'small_cap': 0.11,
            't_bills': 0.03,
            't_bonds': 0.05,
            'corporate_bonds': 0.06,
            'real_estate': 0.08,
            'gold': 0.07
        }
        
        rationale = agent._generate_prediction_rationale(predictions, 10)
        
        # Verify rationale content
        assert isinstance(rationale, str)
        assert len(rationale) > 0
        assert '10-year horizon' in rationale
        assert 'small_cap' in rationale  # Highest return
        assert 't_bills' in rationale    # Lowest return
        
        # Should contain market outlook
        assert any(keyword in rationale.lower() for keyword in ['optimistic', 'conservative', 'moderate'])
    
    def test_generate_prediction_rationale_optimistic_market(self, agent):
        """Test rationale generation for optimistic market conditions"""
        high_predictions = {
            'sp500': 0.12,
            'small_cap': 0.13,
            't_bills': 0.05,
            't_bonds': 0.07,
            'corporate_bonds': 0.08,
            'real_estate': 0.10,
            'gold': 0.09
        }
        
        rationale = agent._generate_prediction_rationale(high_predictions, 5)
        assert 'optimistic' in rationale.lower()
    
    def test_generate_prediction_rationale_conservative_market(self, agent):
        """Test rationale generation for conservative market conditions"""
        low_predictions = {
            'sp500': 0.04,
            'small_cap': 0.05,
            't_bills': 0.02,
            't_bonds': 0.03,
            'corporate_bonds': 0.04,
            'real_estate': 0.04,
            'gold': 0.03
        }
        
        rationale = agent._generate_prediction_rationale(low_predictions, 15)
        assert 'conservative' in rationale.lower()
    
    def test_create_runnable(self, agent):
        """Test creation of LangChain runnable"""
        runnable = agent.create_runnable()
        
        # Verify runnable is created
        assert runnable is not None
        
        # Verify runnable can be invoked
        sample_state = {
            'investment_horizon': 5,
            'asset_classes': ['sp500', 't_bills']
        }
        
        result = runnable.invoke(sample_state)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'predicted_returns' in result
    
    def test_prediction_input_validation(self, agent):
        """Test input validation and edge cases"""
        # Test with empty asset classes
        result = agent.predict_returns({'asset_classes': []})
        assert result['agent_status'] == 'return_prediction_complete'
        
        # Test with invalid horizon
        result = agent.predict_returns({'investment_horizon': -5})
        assert result['agent_status'] == 'return_prediction_complete'
        
        # Test with None values
        result = agent.predict_returns({'investment_horizon': None, 'asset_classes': None})
        assert result['agent_status'] == 'return_prediction_complete'
    
    def test_prediction_logging(self, agent, sample_state, mock_asset_models, caplog):
        """Test logging during prediction process"""
        mock_asset_models.predict_returns.return_value = 0.08
        
        with caplog.at_level(logging.INFO):
            agent.predict_returns(sample_state)
        
        # Verify logging messages
        assert any("Starting return prediction" in record.message for record in caplog.records)
        assert any("Return prediction completed successfully" in record.message for record in caplog.records)


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    pytest.main([__file__, "-v"])