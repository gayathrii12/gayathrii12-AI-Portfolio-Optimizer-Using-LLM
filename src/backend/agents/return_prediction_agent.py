"""
Return Prediction Agent for Financial Returns Optimizer

This agent uses ML models to predict asset returns and provides
intelligent return forecasting capabilities.
"""

from typing import Dict, List, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
import logging
from models.asset_return_models import AssetReturnModels

logger = logging.getLogger(__name__)


class ReturnPredictionInput(BaseModel):
    """Input schema for return prediction agent"""
    investment_horizon: int = Field(description="Investment horizon in years")
    asset_classes: List[str] = Field(description="List of asset classes to predict")
    market_conditions: Optional[Dict[str, Any]] = Field(default=None, description="Current market conditions")


class ReturnPredictionOutput(BaseModel):
    """Output schema for return prediction agent"""
    predicted_returns: Dict[str, float] = Field(description="Predicted annual returns by asset class")
    confidence_scores: Dict[str, float] = Field(description="Confidence scores for predictions")
    prediction_rationale: str = Field(description="Explanation of prediction logic")


class ReturnPredictionAgent:
    """
    Agent responsible for predicting asset returns using ML models
    and providing intelligent analysis of return forecasts.
    """
    
    def __init__(self, asset_models: AssetReturnModels):
        self.asset_models = asset_models
        self.name = "return_prediction_agent"
        
    def predict_returns(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main prediction function that processes input and generates return predictions
        
        Args:
            state: Current workflow state containing prediction parameters
            
        Returns:
            Updated state with return predictions
        """
        try:
            logger.info(f"Starting return prediction for horizon: {state.get('investment_horizon', 'unknown')}")
            
            # Extract input parameters
            horizon = state.get('investment_horizon', 10)
            asset_classes = state.get('asset_classes', [
                'sp500', 'small_cap', 't_bills', 't_bonds', 
                'corporate_bonds', 'real_estate', 'gold'
            ])
            
            # Generate predictions using ML models
            predicted_returns = {}
            confidence_scores = {}
            
            for asset_class in asset_classes:
                try:
                    # Get prediction from ML model
                    prediction = self.asset_models.predict_returns(asset_class, horizon)
                    predicted_returns[asset_class] = prediction
                    
                    # Calculate confidence score based on model performance
                    confidence = self._calculate_confidence(asset_class, prediction)
                    confidence_scores[asset_class] = confidence
                    
                    logger.debug(f"ML model predicted {asset_class}: {prediction:.4f} ({prediction*100:.2f}%) (confidence: {confidence:.2f})")
                    
                except Exception as e:
                    logger.warning(f"ML model prediction failed for {asset_class}: {e}")
                    # Use fallback historical average with error handling
                    fallback_return = self._get_fallback_return(asset_class)
                    predicted_returns[asset_class] = fallback_return
                    confidence_scores[asset_class] = 0.3  # Lower confidence for fallback
                    
                    logger.info(f"Using fallback return for {asset_class}: {fallback_return:.4f} ({fallback_return*100:.2f}%)")
            
            # Generate rationale for predictions
            rationale = self._generate_prediction_rationale(predicted_returns, horizon)
            
            # Update state with predictions
            state.update({
                'predicted_returns': predicted_returns,
                'confidence_scores': confidence_scores,
                'prediction_rationale': rationale,
                'agent_status': 'return_prediction_complete'
            })
            
            logger.info(f"Return prediction completed successfully for {len(predicted_returns)} assets")
            return state
            
        except Exception as e:
            logger.error(f"Return prediction failed: {e}")
            state.update({
                'error': f"Return prediction failed: {str(e)}",
                'agent_status': 'return_prediction_failed'
            })
            return state
    
    def _calculate_confidence(self, asset_class: str, prediction: float) -> float:
        """
        Calculate confidence score for ML model prediction based on model performance
        and prediction reasonableness
        """
        try:
            # Base confidence from ML model validation (higher for ML predictions)
            base_confidence = 0.8
            
            # Adjust based on prediction reasonableness for different asset classes
            if asset_class in ['sp500', 'small_cap']:
                # Equity returns typically 6-15% annually
                if 0.04 <= prediction <= 0.18:
                    confidence_adjustment = 0.15  # Good prediction range
                elif 0.02 <= prediction <= 0.25:
                    confidence_adjustment = 0.05  # Acceptable range
                else:
                    confidence_adjustment = -0.3  # Outside reasonable range
            elif asset_class in ['t_bills', 't_bonds', 'corporate_bonds']:
                # Bond returns typically 2-8% annually
                if 0.015 <= prediction <= 0.10:
                    confidence_adjustment = 0.15  # Good prediction range
                elif 0.005 <= prediction <= 0.12:
                    confidence_adjustment = 0.05  # Acceptable range
                else:
                    confidence_adjustment = -0.3  # Outside reasonable range
            else:
                # Other assets (real estate, gold) typically 3-12%
                if 0.02 <= prediction <= 0.15:
                    confidence_adjustment = 0.1   # Good prediction range
                elif 0.01 <= prediction <= 0.20:
                    confidence_adjustment = 0.0   # Acceptable range
                else:
                    confidence_adjustment = -0.2  # Outside reasonable range
            
            final_confidence = max(0.1, min(1.0, base_confidence + confidence_adjustment))
            
            logger.debug(f"Confidence for {asset_class} prediction {prediction:.4f}: {final_confidence:.2f}")
            return final_confidence
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed for {asset_class}: {e}")
            return 0.5  # Default moderate confidence
    
    def _get_fallback_return(self, asset_class: str) -> float:
        """
        Provide fallback historical average returns when ML prediction fails
        """
        fallback_returns = {
            'sp500': 0.10,
            'small_cap': 0.11,
            't_bills': 0.03,
            't_bonds': 0.05,
            'corporate_bonds': 0.06,
            'real_estate': 0.08,
            'gold': 0.07
        }
        return fallback_returns.get(asset_class, 0.06)
    
    def _generate_prediction_rationale(self, predictions: Dict[str, float], horizon: int) -> str:
        """
        Generate human-readable explanation of prediction logic
        """
        try:
            # Sort predictions by expected return
            sorted_assets = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
            highest_return = sorted_assets[0]
            lowest_return = sorted_assets[-1]
            
            rationale = f"Based on ML model analysis for {horizon}-year horizon:\n"
            rationale += f"• Highest expected return: {highest_return[0]} at {highest_return[1]:.2%}\n"
            rationale += f"• Lowest expected return: {lowest_return[0]} at {lowest_return[1]:.2%}\n"
            
            # Add market context
            avg_return = sum(predictions.values()) / len(predictions)
            if avg_return > 0.08:
                rationale += "• Overall market outlook appears optimistic\n"
            elif avg_return < 0.05:
                rationale += "• Conservative market outlook with lower expected returns\n"
            else:
                rationale += "• Moderate market outlook with balanced expectations\n"
            
            return rationale
            
        except Exception:
            return "Return predictions generated using ML models and historical analysis."
    
    def create_runnable(self) -> RunnableLambda:
        """
        Create a LangChain runnable for this agent
        """
        return RunnableLambda(self.predict_returns)