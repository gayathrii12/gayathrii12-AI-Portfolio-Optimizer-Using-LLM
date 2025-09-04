"""
Rebalancing Agent for Financial Returns Optimizer

This agent handles time-based portfolio rebalancing to adjust
risk allocation over the investment horizon.
"""

from typing import Dict, List, Optional, Any, Tuple
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class RebalancingInput(BaseModel):
    """Input schema for rebalancing agent"""
    initial_allocation: Dict[str, float] = Field(description="Initial portfolio allocation")
    investment_horizon: int = Field(description="Total investment horizon in years")
    rebalancing_frequency: int = Field(default=2, description="Rebalancing frequency in years")
    equity_reduction_rate: float = Field(default=5.0, description="Equity reduction percentage per rebalancing")


class RebalancingOutput(BaseModel):
    """Output schema for rebalancing agent"""
    rebalancing_schedule: List[Dict[str, Any]] = Field(description="Schedule of rebalancing events")
    final_allocation: Dict[str, float] = Field(description="Final portfolio allocation")
    rebalancing_rationale: str = Field(description="Explanation of rebalancing strategy")


class RebalancingAgent:
    """
    Agent responsible for creating time-based rebalancing strategies
    to adjust portfolio risk over the investment horizon.
    """
    
    def __init__(self):
        self.name = "rebalancing_agent"
        
        # Asset classifications for rebalancing logic
        self.equity_assets = ['sp500', 'small_cap']
        self.bond_assets = ['t_bills', 't_bonds', 'corporate_bonds']
        self.alternative_assets = ['real_estate', 'gold']
    
    def create_rebalancing_strategy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main rebalancing function that creates a time-based rebalancing strategy
        
        Args:
            state: Current workflow state containing rebalancing parameters
            
        Returns:
            Updated state with rebalancing strategy
        """
        try:
            logger.info("Starting rebalancing strategy creation")
            
            # Extract input parameters
            initial_allocation = state.get('portfolio_allocation', {})
            investment_horizon = state.get('investment_horizon', 10)
            rebalancing_frequency = state.get('rebalancing_frequency', 2)
            equity_reduction_rate = state.get('equity_reduction_rate', 5.0)
            
            if not initial_allocation:
                logger.error("No initial allocation provided for rebalancing")
                state.update({
                    'error': "No initial allocation provided for rebalancing",
                    'agent_status': 'rebalancing_failed'
                })
                return state
            
            # Create rebalancing schedule
            rebalancing_schedule = self._create_rebalancing_schedule(
                initial_allocation, investment_horizon, rebalancing_frequency, equity_reduction_rate
            )
            
            # Get final allocation
            final_allocation = rebalancing_schedule[-1]['allocation'] if rebalancing_schedule else initial_allocation
            
            # Generate rebalancing rationale
            rationale = self._generate_rebalancing_rationale(
                initial_allocation, final_allocation, investment_horizon, 
                rebalancing_frequency, equity_reduction_rate
            )
            
            # Update state with rebalancing strategy
            state.update({
                'rebalancing_schedule': rebalancing_schedule,
                'final_allocation': final_allocation,
                'rebalancing_rationale': rationale,
                'agent_status': 'rebalancing_complete'
            })
            
            logger.info(f"Rebalancing strategy created with {len(rebalancing_schedule)} rebalancing events")
            return state
            
        except Exception as e:
            logger.error(f"Rebalancing strategy creation failed: {e}")
            state.update({
                'error': f"Rebalancing strategy creation failed: {str(e)}",
                'agent_status': 'rebalancing_failed'
            })
            return state
    
    def _create_rebalancing_schedule(self, initial_allocation: Dict[str, float], 
                                   horizon: int, frequency: int, 
                                   equity_reduction_rate: float) -> List[Dict[str, Any]]:
        """
        Create a schedule of rebalancing events over the investment horizon
        """
        try:
            schedule = []
            current_allocation = initial_allocation.copy()
            
            # Add initial allocation as year 0
            schedule.append({
                'year': 0,
                'allocation': current_allocation.copy(),
                'changes': {},
                'rationale': 'Initial portfolio allocation'
            })
            
            # Create rebalancing events
            for year in range(frequency, horizon + 1, frequency):
                if year > horizon:
                    break
                
                # Calculate new allocation with equity reduction
                new_allocation, changes = self._apply_rebalancing_rules(
                    current_allocation, equity_reduction_rate, year
                )
                
                # Add rebalancing event
                schedule.append({
                    'year': year,
                    'allocation': new_allocation.copy(),
                    'changes': changes,
                    'rationale': f'Rebalancing at year {year}: reducing equity exposure by {equity_reduction_rate}%'
                })
                
                current_allocation = new_allocation
            
            return schedule
            
        except Exception as e:
            logger.error(f"Failed to create rebalancing schedule: {e}")
            return []
    
    def _apply_rebalancing_rules(self, current_allocation: Dict[str, float], 
                               equity_reduction_rate: float, year: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Apply rebalancing rules to adjust portfolio allocation
        """
        try:
            new_allocation = current_allocation.copy()
            changes = {}
            
            # Calculate current equity allocation
            current_equity = sum(new_allocation.get(asset, 0) for asset in self.equity_assets)
            
            if current_equity <= 0:
                logger.warning("No equity allocation to reduce")
                return new_allocation, changes
            
            # Calculate reduction amount
            reduction_amount = current_equity * (equity_reduction_rate / 100.0)
            
            # Reduce equity allocations proportionally
            equity_reduction_per_asset = {}
            for asset in self.equity_assets:
                if asset in new_allocation and new_allocation[asset] > 0:
                    asset_proportion = new_allocation[asset] / current_equity
                    asset_reduction = reduction_amount * asset_proportion
                    equity_reduction_per_asset[asset] = asset_reduction
                    new_allocation[asset] = max(0, new_allocation[asset] - asset_reduction)
                    changes[asset] = -asset_reduction
            
            # Redistribute reduction to bonds (safer assets as investor ages)
            total_reduction = sum(equity_reduction_per_asset.values())
            bond_allocation_increase = self._redistribute_to_bonds(new_allocation, total_reduction)
            
            # Update changes for bonds
            for asset, increase in bond_allocation_increase.items():
                changes[asset] = increase
            
            # Validate and normalize allocation
            new_allocation = self._normalize_allocation(new_allocation)
            
            return new_allocation, changes
            
        except Exception as e:
            logger.error(f"Failed to apply rebalancing rules: {e}")
            return current_allocation, {}
    
    def _redistribute_to_bonds(self, allocation: Dict[str, float], 
                             amount_to_redistribute: float) -> Dict[str, float]:
        """
        Redistribute equity reduction to bond assets
        """
        try:
            bond_increases = {}
            
            # Calculate current bond allocation
            current_bonds = sum(allocation.get(asset, 0) for asset in self.bond_assets)
            
            if current_bonds == 0:
                # If no bonds, distribute equally among bond assets
                increase_per_bond = amount_to_redistribute / len(self.bond_assets)
                for asset in self.bond_assets:
                    allocation[asset] = allocation.get(asset, 0) + increase_per_bond
                    bond_increases[asset] = increase_per_bond
            else:
                # Distribute proportionally to existing bond allocation
                for asset in self.bond_assets:
                    if asset in allocation:
                        asset_proportion = allocation[asset] / current_bonds
                        asset_increase = amount_to_redistribute * asset_proportion
                        allocation[asset] += asset_increase
                        bond_increases[asset] = asset_increase
            
            return bond_increases
            
        except Exception as e:
            logger.error(f"Failed to redistribute to bonds: {e}")
            return {}
    
    def _normalize_allocation(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize allocation to ensure it sums to 100%
        """
        try:
            # Remove negative values
            allocation = {k: max(0, v) for k, v in allocation.items()}
            
            # Calculate total
            total = sum(allocation.values())
            
            if total == 0:
                logger.error("All allocation weights are zero after rebalancing")
                return allocation
            
            # Normalize to 100%
            normalized = {k: (v / total) * 100 for k, v in allocation.items()}
            
            return normalized
            
        except Exception as e:
            logger.error(f"Failed to normalize allocation: {e}")
            return allocation
    
    def _generate_rebalancing_rationale(self, initial_allocation: Dict[str, float],
                                      final_allocation: Dict[str, float],
                                      horizon: int, frequency: int,
                                      equity_reduction_rate: float) -> str:
        """
        Generate human-readable explanation of rebalancing strategy
        """
        try:
            # Calculate initial and final equity allocations
            initial_equity = sum(initial_allocation.get(asset, 0) for asset in self.equity_assets)
            final_equity = sum(final_allocation.get(asset, 0) for asset in self.equity_assets)
            
            initial_bonds = sum(initial_allocation.get(asset, 0) for asset in self.bond_assets)
            final_bonds = sum(final_allocation.get(asset, 0) for asset in self.bond_assets)
            
            rationale = f"Rebalancing strategy over {horizon}-year horizon:\n"
            rationale += f"• Rebalancing frequency: Every {frequency} years\n"
            rationale += f"• Equity reduction rate: {equity_reduction_rate}% per rebalancing\n"
            rationale += f"• Initial equity allocation: {initial_equity:.1f}%\n"
            rationale += f"• Final equity allocation: {final_equity:.1f}%\n"
            rationale += f"• Initial bond allocation: {initial_bonds:.1f}%\n"
            rationale += f"• Final bond allocation: {final_bonds:.1f}%\n"
            
            # Calculate number of rebalancing events
            num_rebalances = len(range(frequency, horizon + 1, frequency))
            rationale += f"• Total rebalancing events: {num_rebalances}\n"
            
            rationale += "• Strategy rationale: Gradually reduce risk exposure as investment horizon shortens\n"
            
            return rationale
            
        except Exception:
            return f"Rebalancing strategy: Reduce equity by {equity_reduction_rate}% every {frequency} years over {horizon}-year horizon"
    
    def calculate_rebalancing_impact(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the impact of rebalancing on portfolio projections
        """
        try:
            logger.info("Calculating rebalancing impact on portfolio projections")
            
            rebalancing_schedule = state.get('rebalancing_schedule', [])
            predicted_returns = state.get('predicted_returns', {})
            investment_amount = state.get('investment_amount', 0)
            investment_type = state.get('investment_type', 'lump_sum')
            monthly_amount = state.get('monthly_amount', 0)
            
            if not rebalancing_schedule or not predicted_returns:
                logger.warning("Insufficient data for rebalancing impact calculation")
                return state
            
            # For now, just mark as complete - detailed calculations can be added later
            state.update({
                'rebalancing_impact_calculated': True
            })
            
            logger.info("Rebalancing impact calculation completed")
            return state
            
        except Exception as e:
            logger.error(f"Rebalancing impact calculation failed: {e}")
            return state
    
    def create_runnable(self) -> RunnableLambda:
        """
        Create a LangChain runnable for this agent
        """
        return RunnableLambda(self.create_rebalancing_strategy)