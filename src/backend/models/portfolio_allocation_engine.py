"""
Portfolio Allocation Engine for Financial Returns Optimizer

This module provides risk-based portfolio allocation strategies with validation
to ensure proper allocation percentages across different asset classes.
"""

from typing import Dict, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


class PortfolioAllocation(BaseModel):
    """Model for detailed portfolio allocation across all asset classes."""
    
    sp500: float = Field(
        ge=0,
        le=100,
        description="S&P 500 allocation percentage"
    )
    small_cap: float = Field(
        ge=0,
        le=100,
        description="US Small Cap allocation percentage"
    )
    t_bills: float = Field(
        ge=0,
        le=100,
        description="Treasury Bills allocation percentage"
    )
    t_bonds: float = Field(
        ge=0,
        le=100,
        description="Treasury Bonds allocation percentage"
    )
    corporate_bonds: float = Field(
        ge=0,
        le=100,
        description="Corporate Bonds allocation percentage"
    )
    real_estate: float = Field(
        ge=0,
        le=100,
        description="Real Estate allocation percentage"
    )
    gold: float = Field(
        ge=0,
        le=100,
        description="Gold allocation percentage"
    )
    
    @model_validator(mode='after')
    def validate_total_allocation(self):
        """Ensure total allocation equals 100%."""
        total = (self.sp500 + self.small_cap + self.t_bills + self.t_bonds + 
                self.corporate_bonds + self.real_estate + self.gold)
        
        if abs(total - 100.0) > 0.01:  # Allow for small floating point errors
            raise ValueError(f"Total allocation must equal 100%, got {total:.2f}%")
        return self
    
    @field_validator('sp500', 'small_cap', 't_bills', 't_bonds', 'corporate_bonds', 'real_estate', 'gold', mode='before')
    @classmethod
    def round_to_two_decimals(cls, v):
        """Round allocation percentages to 2 decimal places."""
        if isinstance(v, (int, float)):
            return round(float(v), 2)
        return v
    
    def to_dict(self) -> Dict[str, float]:
        """Convert allocation to dictionary format."""
        return {
            'sp500': self.sp500,
            'small_cap': self.small_cap,
            't_bills': self.t_bills,
            't_bonds': self.t_bonds,
            'corporate_bonds': self.corporate_bonds,
            'real_estate': self.real_estate,
            'gold': self.gold
        }
    
    def get_equity_percentage(self) -> float:
        """Calculate total equity allocation (stocks)."""
        return self.sp500 + self.small_cap
    
    def get_bonds_percentage(self) -> float:
        """Calculate total bonds allocation."""
        return self.t_bills + self.t_bonds + self.corporate_bonds
    
    def get_alternatives_percentage(self) -> float:
        """Calculate total alternatives allocation (real estate + gold)."""
        return self.real_estate + self.gold


class PortfolioAllocationEngine:
    """
    Engine for generating risk-based portfolio allocations.
    
    Provides three predefined risk profiles:
    - Low Risk (Conservative): 70% bonds, 30% equity
    - Moderate Risk (Balanced): 50% bonds, 50% equity  
    - High Risk (Aggressive): 20% bonds, 80% equity
    """
    
    def __init__(self):
        """Initialize the portfolio allocation engine."""
        self.name = "portfolio_allocation_engine"
        
        # Predefined allocation strategies based on risk profiles
        self._allocations = {
            'low': {
                'sp500': 20.0,
                'small_cap': 10.0,
                't_bills': 25.0,
                't_bonds': 30.0,
                'corporate_bonds': 15.0,
                'real_estate': 0.0,
                'gold': 0.0
            },
            'moderate': {
                'sp500': 35.0,
                'small_cap': 15.0,
                't_bills': 15.0,
                't_bonds': 20.0,
                'corporate_bonds': 15.0,
                'real_estate': 0.0,
                'gold': 0.0
            },
            'high': {
                'sp500': 60.0,
                'small_cap': 20.0,
                't_bills': 5.0,
                't_bonds': 10.0,
                'corporate_bonds': 5.0,
                'real_estate': 0.0,
                'gold': 0.0
            }
        }
        
        logger.info("Portfolio Allocation Engine initialized")
    
    def get_low_risk_allocation(self) -> PortfolioAllocation:
        """
        Get conservative portfolio allocation (70% bonds, 30% equity).
        
        Returns:
            PortfolioAllocation: Conservative allocation with emphasis on capital preservation
        """
        try:
            allocation_dict = self._allocations['low'].copy()
            allocation = PortfolioAllocation(**allocation_dict)
            
            logger.info(f"Generated low risk allocation: {allocation.get_equity_percentage():.1f}% equity, "
                       f"{allocation.get_bonds_percentage():.1f}% bonds")
            
            return allocation
            
        except Exception as e:
            logger.error(f"Failed to generate low risk allocation: {e}")
            raise
    
    def get_moderate_risk_allocation(self) -> PortfolioAllocation:
        """
        Get balanced portfolio allocation (50% bonds, 50% equity).
        
        Returns:
            PortfolioAllocation: Balanced allocation between growth and stability
        """
        try:
            allocation_dict = self._allocations['moderate'].copy()
            allocation = PortfolioAllocation(**allocation_dict)
            
            logger.info(f"Generated moderate risk allocation: {allocation.get_equity_percentage():.1f}% equity, "
                       f"{allocation.get_bonds_percentage():.1f}% bonds")
            
            return allocation
            
        except Exception as e:
            logger.error(f"Failed to generate moderate risk allocation: {e}")
            raise
    
    def get_high_risk_allocation(self) -> PortfolioAllocation:
        """
        Get aggressive portfolio allocation (20% bonds, 80% equity).
        
        Returns:
            PortfolioAllocation: Growth-focused allocation with higher equity exposure
        """
        try:
            allocation_dict = self._allocations['high'].copy()
            allocation = PortfolioAllocation(**allocation_dict)
            
            logger.info(f"Generated high risk allocation: {allocation.get_equity_percentage():.1f}% equity, "
                       f"{allocation.get_bonds_percentage():.1f}% bonds")
            
            return allocation
            
        except Exception as e:
            logger.error(f"Failed to generate high risk allocation: {e}")
            raise
    
    def get_allocation_by_risk_profile(self, risk_profile: Literal['low', 'moderate', 'high']) -> PortfolioAllocation:
        """
        Get portfolio allocation based on risk profile.
        
        Args:
            risk_profile: Risk tolerance level ('low', 'moderate', 'high')
            
        Returns:
            PortfolioAllocation: Allocation matching the specified risk profile
            
        Raises:
            ValueError: If risk profile is not supported
        """
        risk_profile = risk_profile.lower()
        
        if risk_profile == 'low':
            return self.get_low_risk_allocation()
        elif risk_profile == 'moderate':
            return self.get_moderate_risk_allocation()
        elif risk_profile == 'high':
            return self.get_high_risk_allocation()
        else:
            raise ValueError(f"Unsupported risk profile: {risk_profile}. "
                           f"Supported profiles: 'low', 'moderate', 'high'")
    
    def validate_allocation(self, allocation: Dict[str, float]) -> bool:
        """
        Validate that allocation percentages sum to 100%.
        
        Args:
            allocation: Dictionary of asset allocations
            
        Returns:
            bool: True if allocation is valid, False otherwise
        """
        try:
            # Create PortfolioAllocation object to leverage validation
            PortfolioAllocation(**allocation)
            return True
            
        except Exception as e:
            logger.warning(f"Allocation validation failed: {e}")
            return False
    
    def get_supported_risk_profiles(self) -> list:
        """
        Get list of supported risk profiles.
        
        Returns:
            list: List of supported risk profile strings
        """
        return list(self._allocations.keys())
    
    def get_allocation_summary(self, risk_profile: Literal['low', 'moderate', 'high']) -> Dict[str, float]:
        """
        Get high-level allocation summary by asset category.
        
        Args:
            risk_profile: Risk tolerance level
            
        Returns:
            Dict[str, float]: Summary with equity, bonds, and alternatives percentages
        """
        allocation = self.get_allocation_by_risk_profile(risk_profile)
        
        return {
            'equity': allocation.get_equity_percentage(),
            'bonds': allocation.get_bonds_percentage(),
            'alternatives': allocation.get_alternatives_percentage()
        }