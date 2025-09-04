"""
Core data models for the Financial Returns Optimizer system.

This module contains Pydantic models for data validation and type safety
across the multi-agent system.
"""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime


class UserInputModel(BaseModel):
    """Model for user investment parameters and preferences."""
    
    investment_amount: float = Field(
        gt=0,
        description="Investment amount in currency units"
    )
    investment_type: Literal["lumpsum", "sip"] = Field(
        description="Type of investment: lumpsum or systematic investment plan"
    )
    tenure_years: int = Field(
        ge=1,
        le=50,
        description="Investment tenure in years"
    )
    risk_profile: Literal["Low", "Moderate", "High"] = Field(
        description="Risk tolerance level"
    )
    return_expectation: float = Field(
        ge=0,
        le=100,
        description="Expected annual return percentage"
    )
    rebalancing_preferences: Optional[Dict[str, Any]] = Field(
        default=None,
        description="User-defined rebalancing rules and preferences"
    )
    withdrawal_preferences: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Withdrawal schedule and preferences"
    )
    
    @field_validator('investment_amount')
    @classmethod
    def validate_investment_amount(cls, v):
        """Validate investment amount is reasonable."""
        if v > 1_000_000_000:  # 1 billion limit
            raise ValueError("Investment amount exceeds maximum limit")
        return v
    
    @field_validator('return_expectation')
    @classmethod
    def validate_return_expectation(cls, v):
        """Validate return expectation is reasonable."""
        if v > 50:  # 50% annual return seems unrealistic
            raise ValueError("Return expectation exceeds reasonable limits")
        return v


class AssetReturns(BaseModel):
    """Model for historical asset returns data."""
    
    sp500: float = Field(description="S&P 500 annual return percentage")
    small_cap: float = Field(description="US Small Cap annual return percentage")
    t_bills: float = Field(description="Treasury Bills annual return percentage")
    t_bonds: float = Field(description="Treasury Bonds annual return percentage")
    corporate_bonds: float = Field(description="Corporate Bonds annual return percentage")
    real_estate: float = Field(description="Real Estate annual return percentage")
    gold: float = Field(description="Gold annual return percentage")
    year: int = Field(
        ge=1900,
        le=2100,
        description="Year of the returns data"
    )
    
    @field_validator('sp500', 'small_cap', 't_bills', 't_bonds', 'corporate_bonds', 'real_estate', 'gold')
    @classmethod
    def validate_return_ranges(cls, v):
        """Validate return values are within reasonable ranges."""
        if v < -100:  # Cannot lose more than 100%
            raise ValueError("Return cannot be less than -100%")
        if v > 1000:  # 1000% return seems unrealistic for annual data
            raise ValueError("Return exceeds reasonable upper limit")
        return v


class PortfolioAllocation(BaseModel):
    """Model for portfolio allocation with validation constraints."""
    
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
    bonds: float = Field(
        ge=0,
        le=100,
        description="Combined bonds allocation percentage"
    )
    gold: float = Field(
        ge=0,
        le=100,
        description="Gold allocation percentage"
    )
    real_estate: float = Field(
        ge=0,
        le=100,
        description="Real Estate allocation percentage"
    )
    
    @model_validator(mode='after')
    def validate_total_allocation(self):
        """Ensure total allocation equals 100%."""
        total = self.sp500 + self.small_cap + self.bonds + self.gold + self.real_estate
        if abs(total - 100) > 0.01:  # Allow for small floating point errors
            raise ValueError(f"Total allocation must equal 100%, got {total}%")
        return self
    
    @field_validator('sp500', 'small_cap', 'bonds', 'gold', 'real_estate', mode='before')
    @classmethod
    def round_to_two_decimals(cls, v):
        """Round allocation percentages to 2 decimal places."""
        if isinstance(v, (int, float)):
            return round(float(v), 2)
        return v


class ProjectionResult(BaseModel):
    """Model for portfolio projection results over time."""
    
    year: int = Field(
        ge=1,
        description="Year number in the projection timeline"
    )
    portfolio_value: float = Field(
        ge=0,
        description="Portfolio value at end of year"
    )
    annual_return: float = Field(
        description="Annual return percentage for this year"
    )
    cumulative_return: float = Field(
        description="Cumulative return percentage from start"
    )
    
    @field_validator('portfolio_value')
    @classmethod
    def validate_portfolio_value(cls, v):
        """Validate portfolio value is reasonable."""
        if v > 1_000_000_000_000:  # 1 trillion limit
            raise ValueError("Portfolio value exceeds reasonable limits")
        return v


class RiskMetrics(BaseModel):
    """Model for portfolio risk analysis metrics."""
    
    alpha: float = Field(
        description="Alpha relative to benchmark (excess return)"
    )
    beta: float = Field(
        ge=0,
        description="Beta relative to benchmark (volatility correlation)"
    )
    volatility: float = Field(
        ge=0,
        le=100,
        description="Portfolio volatility (standard deviation of returns)"
    )
    sharpe_ratio: float = Field(
        description="Risk-adjusted return metric"
    )
    max_drawdown: float = Field(
        le=0,
        description="Maximum peak-to-trough decline percentage"
    )
    
    @field_validator('alpha')
    @classmethod
    def validate_alpha(cls, v):
        """Validate alpha is within reasonable range."""
        if abs(v) > 50:  # Alpha beyond ±50% seems unrealistic
            raise ValueError("Alpha value exceeds reasonable range")
        return v
    
    @field_validator('beta')
    @classmethod
    def validate_beta(cls, v):
        """Validate beta is within reasonable range."""
        if v > 5:  # Beta > 5 seems unrealistic
            raise ValueError("Beta value exceeds reasonable range")
        return v
    
    @field_validator('sharpe_ratio')
    @classmethod
    def validate_sharpe_ratio(cls, v):
        """Validate Sharpe ratio is within reasonable range."""
        if abs(v) > 10:  # Sharpe ratio beyond ±10 seems unrealistic
            raise ValueError("Sharpe ratio exceeds reasonable range")
        return v
    
    @field_validator('max_drawdown')
    @classmethod
    def validate_max_drawdown(cls, v):
        """Validate max drawdown is negative and reasonable."""
        if v > 0:
            raise ValueError("Maximum drawdown must be negative or zero")
        if v < -100:
            raise ValueError("Maximum drawdown cannot exceed -100%")
        return v


class ErrorResponse(BaseModel):
    """Model for structured error responses."""
    
    error_type: str = Field(description="Type/category of error")
    error_message: str = Field(description="Human-readable error message")
    error_code: int = Field(description="Numeric error code")
    suggested_action: str = Field(description="Suggested action to resolve error")
    timestamp: datetime = Field(default_factory=datetime.now)