"""
Portfolio Allocation Agent for Financial Returns Optimizer

This agent makes intelligent portfolio allocation decisions based on
risk profiles, return predictions, and investment objectives.
"""

from typing import Dict, List, Optional, Any
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class PortfolioAllocationInput(BaseModel):
    """Input schema for portfolio allocation agent"""
    risk_profile: str = Field(description="Risk profile: low, moderate, or high")
    predicted_returns: Dict[str, float] = Field(description="Predicted returns by asset class")
    investment_amount: float = Field(description="Total investment amount")
    investment_horizon: int = Field(description="Investment horizon in years")


class PortfolioAllocationOutput(BaseModel):
    """Output schema for portfolio allocation agent"""
    allocation: Dict[str, float] = Field(description="Portfolio allocation percentages")
    expected_return: float = Field(description="Expected portfolio return")
    risk_level: str = Field(description="Portfolio risk level")
    allocation_rationale: str = Field(description="Explanation of allocation decisions")


class PortfolioAllocationAgent:
    """
    Agent responsible for making intelligent portfolio allocation decisions
    based on risk profiles and return predictions.
    """
    
    def __init__(self):
        self.name = "portfolio_allocation_agent"
        
        # Base allocation templates by risk profile
        self.base_allocations = {
            'low': {
                'sp500': 15.0,
                'small_cap': 5.0,
                't_bills': 30.0,
                't_bonds': 25.0,
                'corporate_bonds': 15.0,
                'real_estate': 7.0,
                'gold': 3.0
            },
            'moderate': {
                'sp500': 30.0,
                'small_cap': 10.0,
                't_bills': 15.0,
                't_bonds': 20.0,
                'corporate_bonds': 10.0,
                'real_estate': 10.0,
                'gold': 5.0
            },
            'high': {
                'sp500': 45.0,
                'small_cap': 20.0,
                't_bills': 5.0,
                't_bonds': 10.0,
                'corporate_bonds': 5.0,
                'real_estate': 10.0,
                'gold': 5.0
            }
        }
    
    def allocate_portfolio(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main allocation function that processes input and generates portfolio allocation
        
        Args:
            state: Current workflow state containing allocation parameters
            
        Returns:
            Updated state with portfolio allocation
        """
        try:
            logger.info(f"Starting portfolio allocation for risk profile: {state.get('risk_profile', 'unknown')}")
            
            # Extract input parameters
            risk_profile = state.get('risk_profile', 'moderate').lower()
            predicted_returns = state.get('predicted_returns', {})
            investment_amount = state.get('investment_amount', 0)
            investment_horizon = state.get('investment_horizon', 10)
            
            # Validate risk profile
            if risk_profile not in self.base_allocations:
                logger.warning(f"Invalid risk profile: {risk_profile}, defaulting to moderate")
                risk_profile = 'moderate'
            
            # Get base allocation for risk profile
            base_allocation = self.base_allocations[risk_profile].copy()
            
            # Optimize allocation based on predicted returns
            optimized_allocation = self._optimize_allocation(
                base_allocation, predicted_returns, risk_profile, investment_horizon
            )
            
            # Validate and normalize allocation
            final_allocation = self._validate_allocation(optimized_allocation)
            
            # Calculate expected portfolio return
            expected_return = self._calculate_expected_return(final_allocation, predicted_returns)
            
            # Generate allocation rationale
            rationale = self._generate_allocation_rationale(
                final_allocation, risk_profile, expected_return, investment_horizon
            )
            
            # Update state with allocation results
            state.update({
                'portfolio_allocation': final_allocation,
                'expected_portfolio_return': expected_return,
                'risk_level': risk_profile,
                'allocation_rationale': rationale,
                'agent_status': 'portfolio_allocation_complete'
            })
            
            logger.info(f"Portfolio allocation completed with expected return: {expected_return:.2%}")
            return state
            
        except Exception as e:
            logger.error(f"Portfolio allocation failed: {e}")
            state.update({
                'error': f"Portfolio allocation failed: {str(e)}",
                'agent_status': 'portfolio_allocation_failed'
            })
            return state
    
    def _optimize_allocation(self, base_allocation: Dict[str, float], 
                           predicted_returns: Dict[str, float], 
                           risk_profile: str, horizon: int) -> Dict[str, float]:
        """
        Optimize allocation based on ML model predicted returns while respecting risk constraints
        """
        try:
            optimized = base_allocation.copy()
            
            if not predicted_returns:
                logger.warning("No ML model predictions available, using base allocation")
                return optimized
            
            logger.info(f"Optimizing allocation using ML predictions for {len(predicted_returns)} assets")
            
            # Calculate return-based adjustments using ML predictions
            avg_return = sum(predicted_returns.values()) / len(predicted_returns)
            
            # Log ML predictions for transparency
            for asset, predicted_return in predicted_returns.items():
                logger.debug(f"ML prediction for {asset}: {predicted_return:.4f} ({predicted_return*100:.2f}%)")
            
            for asset, base_weight in base_allocation.items():
                if asset in predicted_returns:
                    predicted_return = predicted_returns[asset]
                    
                    # Calculate relative performance vs average ML prediction
                    relative_performance = predicted_return / avg_return if avg_return > 0 else 1.0
                    
                    # Apply ML-informed adjustments based on risk profile
                    if risk_profile == 'low':
                        # Conservative: small adjustments, favor bonds based on ML predictions
                        adjustment_factor = 0.1 * (relative_performance - 1.0)
                        if asset in ['t_bills', 't_bonds', 'corporate_bonds']:
                            # Favor bonds more if ML predicts higher returns for bonds
                            adjustment_factor *= 1.5
                    elif risk_profile == 'moderate':
                        # Moderate: balanced adjustments based on ML predictions
                        adjustment_factor = 0.15 * (relative_performance - 1.0)
                    else:  # high risk
                        # Aggressive: larger adjustments, favor equities if ML predicts higher returns
                        adjustment_factor = 0.2 * (relative_performance - 1.0)
                        if asset in ['sp500', 'small_cap']:
                            # Favor equities more if ML predicts higher returns
                            adjustment_factor *= 1.3
                    
                    # Apply ML-informed adjustment with bounds
                    new_weight = base_weight * (1 + adjustment_factor)
                    
                    # Set reasonable bounds based on asset type and risk profile
                    min_weight, max_weight = self._get_weight_bounds(asset, risk_profile)
                    optimized[asset] = max(min_weight, min(max_weight, new_weight))
                    
                    logger.debug(f"Adjusted {asset}: {base_weight:.1f}% -> {optimized[asset]:.1f}% (ML factor: {adjustment_factor:.3f})")
            
            logger.info("Portfolio allocation optimized using ML model predictions")
            return optimized
            
        except Exception as e:
            logger.warning(f"ML-based optimization failed, using base allocation: {e}")
            return base_allocation
    
    def _get_weight_bounds(self, asset: str, risk_profile: str) -> tuple:
        """
        Get minimum and maximum weight bounds for an asset based on risk profile
        """
        bounds = {
            'low': {
                'sp500': (5.0, 25.0),
                'small_cap': (0.0, 15.0),
                't_bills': (20.0, 40.0),
                't_bonds': (15.0, 35.0),
                'corporate_bonds': (10.0, 25.0),
                'real_estate': (3.0, 15.0),
                'gold': (0.0, 10.0)
            },
            'moderate': {
                'sp500': (15.0, 45.0),
                'small_cap': (5.0, 20.0),
                't_bills': (5.0, 25.0),
                't_bonds': (10.0, 30.0),
                'corporate_bonds': (5.0, 20.0),
                'real_estate': (5.0, 20.0),
                'gold': (0.0, 15.0)
            },
            'high': {
                'sp500': (25.0, 60.0),
                'small_cap': (10.0, 35.0),
                't_bills': (0.0, 15.0),
                't_bonds': (5.0, 20.0),
                'corporate_bonds': (0.0, 15.0),
                'real_estate': (5.0, 25.0),
                'gold': (0.0, 15.0)
            }
        }
        
        return bounds.get(risk_profile, {}).get(asset, (0.0, 100.0))
    
    def _validate_allocation(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """
        Validate and normalize allocation to ensure it sums to 100%
        """
        try:
            # Remove any negative weights
            allocation = {k: max(0, v) for k, v in allocation.items()}
            
            # Calculate total
            total = sum(allocation.values())
            
            if total == 0:
                logger.error("All allocation weights are zero")
                raise ValueError("Invalid allocation: all weights are zero")
            
            # Normalize to 100%
            normalized = {k: (v / total) * 100 for k, v in allocation.items()}
            
            # Verify sum is approximately 100%
            final_total = sum(normalized.values())
            if abs(final_total - 100.0) > 0.01:
                logger.warning(f"Allocation sum is {final_total:.2f}%, adjusting to 100%")
                # Adjust largest component to make exact 100%
                largest_asset = max(normalized.keys(), key=lambda k: normalized[k])
                normalized[largest_asset] += (100.0 - final_total)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Allocation validation failed: {e}")
            # Return equal weights as fallback
            num_assets = len(allocation)
            return {k: 100.0 / num_assets for k in allocation.keys()}
    
    def _calculate_expected_return(self, allocation: Dict[str, float], 
                                 predicted_returns: Dict[str, float]) -> float:
        """
        Calculate expected portfolio return based on allocation and predicted returns
        """
        try:
            if not predicted_returns:
                logger.warning("No predicted returns available for portfolio return calculation")
                return 0.06  # Default 6% return
            
            expected_return = 0.0
            total_weight = 0.0
            
            for asset, weight in allocation.items():
                if asset in predicted_returns:
                    expected_return += (weight / 100.0) * predicted_returns[asset]
                    total_weight += weight
            
            if total_weight == 0:
                logger.warning("No matching assets between allocation and predictions")
                return 0.06
            
            return expected_return
            
        except Exception as e:
            logger.error(f"Expected return calculation failed: {e}")
            return 0.06
    
    def _generate_allocation_rationale(self, allocation: Dict[str, float], 
                                     risk_profile: str, expected_return: float, 
                                     horizon: int) -> str:
        """
        Generate human-readable explanation of allocation decisions
        """
        try:
            # Sort allocation by weight
            sorted_allocation = sorted(allocation.items(), key=lambda x: x[1], reverse=True)
            
            rationale = f"Portfolio allocation for {risk_profile} risk profile ({horizon}-year horizon):\n"
            
            # Top 3 allocations
            rationale += "• Top allocations:\n"
            for asset, weight in sorted_allocation[:3]:
                rationale += f"  - {asset}: {weight:.1f}%\n"
            
            # Risk profile explanation
            if risk_profile == 'low':
                rationale += "• Conservative approach emphasizing capital preservation\n"
                rationale += "• Higher allocation to bonds and stable assets\n"
            elif risk_profile == 'moderate':
                rationale += "• Balanced approach between growth and stability\n"
                rationale += "• Diversified across asset classes\n"
            else:
                rationale += "• Growth-focused approach for higher returns\n"
                rationale += "• Higher allocation to equities and growth assets\n"
            
            rationale += f"• Expected annual return: {expected_return:.2%}\n"
            
            return rationale
            
        except Exception:
            return f"Portfolio allocated for {risk_profile} risk profile with expected return of {expected_return:.2%}"
    
    def create_runnable(self) -> RunnableLambda:
        """
        Create a LangChain runnable for this agent
        """
        return RunnableLambda(self.allocate_portfolio)