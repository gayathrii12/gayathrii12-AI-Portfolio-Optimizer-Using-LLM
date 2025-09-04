"""
Investment Calculators for Financial Returns Optimizer

This module provides calculators for different investment strategies:
- Lump Sum: Single initial investment with compound growth
- SIP: Systematic Investment Plan with monthly contributions
- SWP: Systematic Withdrawal Plan with regular withdrawals

Each calculator generates year-by-year projections based on expected returns.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import logging
import math

logger = logging.getLogger(__name__)


class InvestmentProjection(BaseModel):
    """Model for investment projection results over time."""
    
    year: int = Field(
        ge=1,
        description="Year number in the projection timeline"
    )
    portfolio_value: float = Field(
        ge=0,
        description="Portfolio value at end of year"
    )
    annual_contribution: float = Field(
        ge=0,
        description="Total contributions made during this year"
    )
    annual_withdrawal: float = Field(
        ge=0,
        description="Total withdrawals made during this year"
    )
    annual_return: float = Field(
        description="Annual return percentage for this year"
    )
    cumulative_contributions: float = Field(
        ge=0,
        description="Total contributions made from start to this year"
    )
    cumulative_withdrawals: float = Field(
        ge=0,
        description="Total withdrawals made from start to this year"
    )
    
    @field_validator('portfolio_value')
    @classmethod
    def validate_portfolio_value(cls, v):
        """Validate portfolio value is reasonable."""
        if v > 1_000_000_000_000:  # 1 trillion limit
            raise ValueError("Portfolio value exceeds reasonable limits")
        return v


class InvestmentCalculators:
    """
    Investment calculators for different investment strategies.
    
    This class provides methods to calculate portfolio growth for:
    - Lump sum investments
    - Systematic Investment Plans (SIP)
    - Systematic Withdrawal Plans (SWP)
    """
    
    def __init__(self):
        """Initialize the investment calculators."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_lump_sum(
        self, 
        amount: float, 
        returns: Dict[str, float], 
        years: int
    ) -> List[InvestmentProjection]:
        """
        Calculate portfolio growth for a lump sum investment.
        
        Args:
            amount: Initial investment amount
            returns: Dictionary of asset returns (e.g., {'sp500': 10.5, 'bonds': 4.2})
            years: Investment tenure in years
            
        Returns:
            List of InvestmentProjection objects showing year-by-year growth
            
        Raises:
            ValueError: If inputs are invalid
        """
        if amount <= 0:
            raise ValueError("Investment amount must be positive")
        if years <= 0:
            raise ValueError("Investment years must be positive")
        if not returns:
            raise ValueError("Returns dictionary cannot be empty")
        
        # Calculate simple average return (assuming equal weights)
        # In a real implementation, this would take allocation weights as a separate parameter
        if any(ret < -100 for ret in returns.values()):
            raise ValueError("Returns cannot be less than -100%")
        
        weighted_return = sum(returns.values()) / len(returns)
        annual_return_rate = weighted_return / 100  # Convert percentage to decimal
        
        projections = []
        current_value = amount
        
        for year in range(1, years + 1):
            # Apply compound growth
            current_value = current_value * (1 + annual_return_rate)
            
            projection = InvestmentProjection(
                year=year,
                portfolio_value=round(current_value, 2),
                annual_contribution=amount if year == 1 else 0,
                annual_withdrawal=0,
                annual_return=weighted_return,
                cumulative_contributions=amount,
                cumulative_withdrawals=0
            )
            projections.append(projection)
        
        self.logger.info(f"Calculated lump sum projection: {amount} over {years} years, "
                        f"final value: {current_value:.2f}")
        
        return projections
    
    def calculate_sip(
        self, 
        monthly_amount: float, 
        returns: Dict[str, float], 
        years: int
    ) -> List[InvestmentProjection]:
        """
        Calculate portfolio growth for Systematic Investment Plan (SIP).
        
        Args:
            monthly_amount: Monthly investment amount
            returns: Dictionary of asset returns
            years: Investment tenure in years
            
        Returns:
            List of InvestmentProjection objects showing year-by-year growth
            
        Raises:
            ValueError: If inputs are invalid
        """
        if monthly_amount <= 0:
            raise ValueError("Monthly investment amount must be positive")
        if years <= 0:
            raise ValueError("Investment years must be positive")
        if not returns:
            raise ValueError("Returns dictionary cannot be empty")
        
        # Calculate simple average return (assuming equal weights)
        # In a real implementation, this would take allocation weights as a separate parameter
        if any(ret < -100 for ret in returns.values()):
            raise ValueError("Returns cannot be less than -100%")
        
        weighted_return = sum(returns.values()) / len(returns)
        annual_return_rate = weighted_return / 100  # Convert percentage to decimal
        monthly_return_rate = annual_return_rate / 12  # Monthly compounding
        
        projections = []
        current_value = 0
        total_contributions = 0
        
        for year in range(1, years + 1):
            year_start_value = current_value
            annual_contribution = monthly_amount * 12
            
            # Calculate monthly SIP growth for this year
            for month in range(12):
                current_value += monthly_amount  # Add monthly contribution
                current_value = current_value * (1 + monthly_return_rate)  # Apply monthly return
            
            total_contributions += annual_contribution
            
            projection = InvestmentProjection(
                year=year,
                portfolio_value=round(current_value, 2),
                annual_contribution=annual_contribution,
                annual_withdrawal=0,
                annual_return=weighted_return,
                cumulative_contributions=total_contributions,
                cumulative_withdrawals=0
            )
            projections.append(projection)
        
        self.logger.info(f"Calculated SIP projection: {monthly_amount}/month over {years} years, "
                        f"final value: {current_value:.2f}")
        
        return projections
    
    def calculate_swp(
        self, 
        initial_amount: float, 
        monthly_withdrawal: float, 
        returns: Dict[str, float], 
        years: int
    ) -> List[InvestmentProjection]:
        """
        Calculate portfolio value for Systematic Withdrawal Plan (SWP).
        
        Args:
            initial_amount: Initial investment corpus
            monthly_withdrawal: Monthly withdrawal amount
            returns: Dictionary of asset returns
            years: Withdrawal period in years
            
        Returns:
            List of InvestmentProjection objects showing year-by-year values
            
        Raises:
            ValueError: If inputs are invalid or withdrawals exceed growth
        """
        if initial_amount <= 0:
            raise ValueError("Initial investment amount must be positive")
        if monthly_withdrawal <= 0:
            raise ValueError("Monthly withdrawal amount must be positive")
        if years <= 0:
            raise ValueError("Withdrawal years must be positive")
        if not returns:
            raise ValueError("Returns dictionary cannot be empty")
        
        # Calculate simple average return (assuming equal weights)
        # In a real implementation, this would take allocation weights as a separate parameter
        if any(ret < -100 for ret in returns.values()):
            raise ValueError("Returns cannot be less than -100%")
        
        weighted_return = sum(returns.values()) / len(returns)
        annual_return_rate = weighted_return / 100  # Convert percentage to decimal
        monthly_return_rate = annual_return_rate / 12  # Monthly compounding
        
        # Check if withdrawal rate is sustainable
        annual_withdrawal = monthly_withdrawal * 12
        if annual_withdrawal > initial_amount * annual_return_rate:
            self.logger.warning(f"High withdrawal rate detected: {annual_withdrawal:.2f} vs "
                              f"expected annual return: {initial_amount * annual_return_rate:.2f}")
        
        projections = []
        current_value = initial_amount
        total_withdrawals = 0
        
        for year in range(1, years + 1):
            annual_withdrawal_amount = 0
            
            # Calculate monthly SWP for this year
            for month in range(12):
                if current_value >= monthly_withdrawal:
                    current_value -= monthly_withdrawal  # Withdraw monthly amount
                    annual_withdrawal_amount += monthly_withdrawal
                    current_value = current_value * (1 + monthly_return_rate)  # Apply monthly return
                else:
                    # Portfolio exhausted
                    remaining_withdrawal = current_value
                    annual_withdrawal_amount += remaining_withdrawal
                    current_value = 0
                    self.logger.warning(f"Portfolio exhausted in year {year}, month {month + 1}")
                    break
            
            total_withdrawals += annual_withdrawal_amount
            
            projection = InvestmentProjection(
                year=year,
                portfolio_value=round(max(current_value, 0), 2),
                annual_contribution=0,
                annual_withdrawal=annual_withdrawal_amount,
                annual_return=weighted_return,
                cumulative_contributions=initial_amount,
                cumulative_withdrawals=total_withdrawals
            )
            projections.append(projection)
            
            # Stop if portfolio is exhausted
            if current_value <= 0:
                break
        
        self.logger.info(f"Calculated SWP projection: {initial_amount} initial, "
                        f"{monthly_withdrawal}/month over {years} years, "
                        f"final value: {current_value:.2f}")
        
        return projections
    
    def generate_investment_summary(
        self, 
        projections: List[InvestmentProjection]
    ) -> Dict[str, float]:
        """
        Generate summary statistics for investment projections.
        
        Args:
            projections: List of investment projections
            
        Returns:
            Dictionary containing summary statistics
        """
        if not projections:
            return {}
        
        final_projection = projections[-1]
        initial_value = projections[0].cumulative_contributions
        
        total_return = final_projection.portfolio_value - initial_value + final_projection.cumulative_withdrawals
        total_return_percentage = (total_return / initial_value) * 100 if initial_value > 0 else 0
        
        # Calculate CAGR (Compound Annual Growth Rate)
        years = len(projections)
        if years > 0 and initial_value > 0:
            cagr = (((final_projection.portfolio_value + final_projection.cumulative_withdrawals) / initial_value) ** (1/years) - 1) * 100
        else:
            cagr = 0
        
        return {
            'initial_investment': initial_value,
            'final_value': final_projection.portfolio_value,
            'total_contributions': final_projection.cumulative_contributions,
            'total_withdrawals': final_projection.cumulative_withdrawals,
            'total_return': total_return,
            'total_return_percentage': round(total_return_percentage, 2),
            'cagr': round(cagr, 2),
            'investment_years': years
        }