"""
Unit tests for data models validation logic.

Tests all Pydantic models for proper validation, constraints,
and error handling.
"""

import pytest
from pydantic import ValidationError
from models.data_models import (
    UserInputModel,
    AssetReturns,
    PortfolioAllocation,
    ProjectionResult,
    RiskMetrics,
    ErrorResponse
)


class TestUserInputModel:
    """Test cases for UserInputModel validation."""
    
    def test_valid_user_input(self):
        """Test valid user input creation."""
        user_input = UserInputModel(
            investment_amount=100000.0,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=12.0
        )
        assert user_input.investment_amount == 100000.0
        assert user_input.investment_type == "lumpsum"
        assert user_input.tenure_years == 10
        assert user_input.risk_profile == "Moderate"
        assert user_input.return_expectation == 12.0
    
    def test_invalid_investment_amount_negative(self):
        """Test validation fails for negative investment amount."""
        with pytest.raises(ValidationError) as exc_info:
            UserInputModel(
                investment_amount=-1000.0,
                investment_type="lumpsum",
                tenure_years=10,
                risk_profile="Low",
                return_expectation=8.0
            )
        assert "greater than 0" in str(exc_info.value)
    
    def test_invalid_investment_amount_too_large(self):
        """Test validation fails for excessive investment amount."""
        with pytest.raises(ValidationError) as exc_info:
            UserInputModel(
                investment_amount=2_000_000_000.0,  # 2 billion
                investment_type="lumpsum",
                tenure_years=10,
                risk_profile="Low",
                return_expectation=8.0
            )
        assert "exceeds maximum limit" in str(exc_info.value)
    
    def test_invalid_investment_type(self):
        """Test validation fails for invalid investment type."""
        with pytest.raises(ValidationError) as exc_info:
            UserInputModel(
                investment_amount=100000.0,
                investment_type="invalid_type",
                tenure_years=10,
                risk_profile="Low",
                return_expectation=8.0
            )
        assert "Input should be 'lumpsum' or 'sip'" in str(exc_info.value)
    
    def test_invalid_tenure_years_zero(self):
        """Test validation fails for zero tenure years."""
        with pytest.raises(ValidationError) as exc_info:
            UserInputModel(
                investment_amount=100000.0,
                investment_type="sip",
                tenure_years=0,
                risk_profile="High",
                return_expectation=15.0
            )
        assert "greater than or equal to 1" in str(exc_info.value)
    
    def test_invalid_tenure_years_too_large(self):
        """Test validation fails for excessive tenure years."""
        with pytest.raises(ValidationError) as exc_info:
            UserInputModel(
                investment_amount=100000.0,
                investment_type="sip",
                tenure_years=100,
                risk_profile="High",
                return_expectation=15.0
            )
        assert "less than or equal to 50" in str(exc_info.value)
    
    def test_invalid_risk_profile(self):
        """Test validation fails for invalid risk profile."""
        with pytest.raises(ValidationError) as exc_info:
            UserInputModel(
                investment_amount=100000.0,
                investment_type="lumpsum",
                tenure_years=10,
                risk_profile="Invalid",
                return_expectation=12.0
            )
        assert "Input should be 'Low', 'Moderate' or 'High'" in str(exc_info.value)
    
    def test_invalid_return_expectation_negative(self):
        """Test validation fails for negative return expectation."""
        with pytest.raises(ValidationError) as exc_info:
            UserInputModel(
                investment_amount=100000.0,
                investment_type="lumpsum",
                tenure_years=10,
                risk_profile="Low",
                return_expectation=-5.0
            )
        assert "greater than or equal to 0" in str(exc_info.value)
    
    def test_invalid_return_expectation_too_high(self):
        """Test validation fails for unrealistic return expectation."""
        with pytest.raises(ValidationError) as exc_info:
            UserInputModel(
                investment_amount=100000.0,
                investment_type="lumpsum",
                tenure_years=10,
                risk_profile="High",
                return_expectation=75.0
            )
        assert "exceeds reasonable limits" in str(exc_info.value)
    
    def test_optional_fields(self):
        """Test optional fields can be None."""
        user_input = UserInputModel(
            investment_amount=50000.0,
            investment_type="sip",
            tenure_years=5,
            risk_profile="Moderate",
            return_expectation=10.0,
            rebalancing_preferences={"frequency": "annual"},
            withdrawal_preferences=None
        )
        assert user_input.rebalancing_preferences == {"frequency": "annual"}
        assert user_input.withdrawal_preferences is None


class TestAssetReturns:
    """Test cases for AssetReturns validation."""
    
    def test_valid_asset_returns(self):
        """Test valid asset returns creation."""
        returns = AssetReturns(
            sp500=12.5,
            small_cap=15.2,
            t_bills=2.1,
            t_bonds=4.8,
            corporate_bonds=6.3,
            real_estate=8.9,
            gold=3.2,
            year=2023
        )
        assert returns.sp500 == 12.5
        assert returns.year == 2023
    
    def test_invalid_return_too_negative(self):
        """Test validation fails for returns below -100%."""
        with pytest.raises(ValidationError) as exc_info:
            AssetReturns(
                sp500=-150.0,  # Cannot lose more than 100%
                small_cap=15.2,
                t_bills=2.1,
                t_bonds=4.8,
                corporate_bonds=6.3,
                real_estate=8.9,
                gold=3.2,
                year=2008
            )
        assert "cannot be less than -100%" in str(exc_info.value)
    
    def test_invalid_return_too_high(self):
        """Test validation fails for unrealistic high returns."""
        with pytest.raises(ValidationError) as exc_info:
            AssetReturns(
                sp500=12.5,
                small_cap=1500.0,  # 1500% return is unrealistic
                t_bills=2.1,
                t_bonds=4.8,
                corporate_bonds=6.3,
                real_estate=8.9,
                gold=3.2,
                year=2023
            )
        assert "exceeds reasonable upper limit" in str(exc_info.value)
    
    def test_invalid_year_too_old(self):
        """Test validation fails for years before 1900."""
        with pytest.raises(ValidationError) as exc_info:
            AssetReturns(
                sp500=12.5,
                small_cap=15.2,
                t_bills=2.1,
                t_bonds=4.8,
                corporate_bonds=6.3,
                real_estate=8.9,
                gold=3.2,
                year=1850
            )
        assert "greater than or equal to 1900" in str(exc_info.value)
    
    def test_invalid_year_too_future(self):
        """Test validation fails for years beyond 2100."""
        with pytest.raises(ValidationError) as exc_info:
            AssetReturns(
                sp500=12.5,
                small_cap=15.2,
                t_bills=2.1,
                t_bonds=4.8,
                corporate_bonds=6.3,
                real_estate=8.9,
                gold=3.2,
                year=2150
            )
        assert "less than or equal to 2100" in str(exc_info.value)


class TestPortfolioAllocation:
    """Test cases for PortfolioAllocation validation."""
    
    def test_valid_portfolio_allocation(self):
        """Test valid portfolio allocation creation."""
        allocation = PortfolioAllocation(
            sp500=40.0,
            small_cap=20.0,
            bonds=25.0,
            gold=5.0,
            real_estate=10.0
        )
        assert allocation.sp500 == 40.0
        assert sum([allocation.sp500, allocation.small_cap, allocation.bonds, 
                   allocation.gold, allocation.real_estate]) == 100.0
    
    def test_invalid_allocation_not_100_percent(self):
        """Test validation fails when total allocation != 100%."""
        with pytest.raises(ValidationError) as exc_info:
            PortfolioAllocation(
                sp500=40.0,
                small_cap=20.0,
                bonds=25.0,
                gold=5.0,
                real_estate=15.0  # Total = 105%
            )
        assert "must equal 100%" in str(exc_info.value)
    
    def test_invalid_negative_allocation(self):
        """Test validation fails for negative allocations."""
        with pytest.raises(ValidationError) as exc_info:
            PortfolioAllocation(
                sp500=45.0,
                small_cap=-5.0,  # Negative allocation
                bonds=35.0,
                gold=10.0,
                real_estate=15.0
            )
        assert "greater than or equal to 0" in str(exc_info.value)
    
    def test_invalid_allocation_over_100_percent(self):
        """Test validation fails for individual allocation > 100%."""
        with pytest.raises(ValidationError) as exc_info:
            PortfolioAllocation(
                sp500=150.0,  # Over 100%
                small_cap=0.0,
                bonds=0.0,
                gold=0.0,
                real_estate=0.0
            )
        assert "less than or equal to 100" in str(exc_info.value)
    
    def test_allocation_rounding(self):
        """Test allocation values are rounded to 2 decimal places."""
        allocation = PortfolioAllocation(
            sp500=40.123456,
            small_cap=20.987654,
            bonds=24.444444,
            gold=5.555555,
            real_estate=8.888891
        )
        assert allocation.sp500 == 40.12
        assert allocation.small_cap == 20.99
        assert allocation.bonds == 24.44
        assert allocation.gold == 5.56
        assert allocation.real_estate == 8.89
    
    def test_allocation_floating_point_tolerance(self):
        """Test small floating point errors are tolerated."""
        # This should pass despite tiny floating point error
        allocation = PortfolioAllocation(
            sp500=33.33,
            small_cap=33.33,
            bonds=33.33,
            gold=0.01,  # Total = 100.00, exactly
            real_estate=0.00
        )
        assert allocation.sp500 == 33.33


class TestProjectionResult:
    """Test cases for ProjectionResult validation."""
    
    def test_valid_projection_result(self):
        """Test valid projection result creation."""
        result = ProjectionResult(
            year=5,
            portfolio_value=150000.0,
            annual_return=12.5,
            cumulative_return=50.0
        )
        assert result.year == 5
        assert result.portfolio_value == 150000.0
        assert result.annual_return == 12.5
        assert result.cumulative_return == 50.0
    
    def test_invalid_year_zero(self):
        """Test validation fails for year 0."""
        with pytest.raises(ValidationError) as exc_info:
            ProjectionResult(
                year=0,
                portfolio_value=100000.0,
                annual_return=10.0,
                cumulative_return=0.0
            )
        assert "greater than or equal to 1" in str(exc_info.value)
    
    def test_invalid_negative_portfolio_value(self):
        """Test validation fails for negative portfolio value."""
        with pytest.raises(ValidationError) as exc_info:
            ProjectionResult(
                year=1,
                portfolio_value=-50000.0,
                annual_return=-50.0,
                cumulative_return=-50.0
            )
        assert "greater than or equal to 0" in str(exc_info.value)
    
    def test_invalid_excessive_portfolio_value(self):
        """Test validation fails for unrealistic portfolio value."""
        with pytest.raises(ValidationError) as exc_info:
            ProjectionResult(
                year=10,
                portfolio_value=2_000_000_000_000.0,  # 2 trillion
                annual_return=1000.0,
                cumulative_return=10000.0
            )
        assert "exceeds reasonable limits" in str(exc_info.value)


class TestRiskMetrics:
    """Test cases for RiskMetrics validation."""
    
    def test_valid_risk_metrics(self):
        """Test valid risk metrics creation."""
        metrics = RiskMetrics(
            alpha=2.5,
            beta=1.2,
            volatility=15.8,
            sharpe_ratio=1.4,
            max_drawdown=-12.3
        )
        assert metrics.alpha == 2.5
        assert metrics.beta == 1.2
        assert metrics.volatility == 15.8
        assert metrics.sharpe_ratio == 1.4
        assert metrics.max_drawdown == -12.3
    
    def test_invalid_alpha_too_high(self):
        """Test validation fails for unrealistic alpha."""
        with pytest.raises(ValidationError) as exc_info:
            RiskMetrics(
                alpha=75.0,  # Unrealistic alpha
                beta=1.0,
                volatility=15.0,
                sharpe_ratio=1.0,
                max_drawdown=-10.0
            )
        assert "exceeds reasonable range" in str(exc_info.value)
    
    def test_invalid_negative_beta(self):
        """Test validation fails for negative beta."""
        with pytest.raises(ValidationError) as exc_info:
            RiskMetrics(
                alpha=2.0,
                beta=-0.5,  # Beta cannot be negative
                volatility=15.0,
                sharpe_ratio=1.0,
                max_drawdown=-10.0
            )
        assert "greater than or equal to 0" in str(exc_info.value)
    
    def test_invalid_beta_too_high(self):
        """Test validation fails for unrealistic beta."""
        with pytest.raises(ValidationError) as exc_info:
            RiskMetrics(
                alpha=2.0,
                beta=10.0,  # Unrealistic beta
                volatility=15.0,
                sharpe_ratio=1.0,
                max_drawdown=-10.0
            )
        assert "exceeds reasonable range" in str(exc_info.value)
    
    def test_invalid_negative_volatility(self):
        """Test validation fails for negative volatility."""
        with pytest.raises(ValidationError) as exc_info:
            RiskMetrics(
                alpha=2.0,
                beta=1.0,
                volatility=-5.0,  # Volatility cannot be negative
                sharpe_ratio=1.0,
                max_drawdown=-10.0
            )
        assert "greater than or equal to 0" in str(exc_info.value)
    
    def test_invalid_volatility_over_100(self):
        """Test validation fails for volatility > 100%."""
        with pytest.raises(ValidationError) as exc_info:
            RiskMetrics(
                alpha=2.0,
                beta=1.0,
                volatility=150.0,  # Over 100%
                sharpe_ratio=1.0,
                max_drawdown=-10.0
            )
        assert "less than or equal to 100" in str(exc_info.value)
    
    def test_invalid_sharpe_ratio_too_high(self):
        """Test validation fails for unrealistic Sharpe ratio."""
        with pytest.raises(ValidationError) as exc_info:
            RiskMetrics(
                alpha=2.0,
                beta=1.0,
                volatility=15.0,
                sharpe_ratio=15.0,  # Unrealistic Sharpe ratio
                max_drawdown=-10.0
            )
        assert "exceeds reasonable range" in str(exc_info.value)
    
    def test_invalid_positive_max_drawdown(self):
        """Test validation fails for positive max drawdown."""
        with pytest.raises(ValidationError) as exc_info:
            RiskMetrics(
                alpha=2.0,
                beta=1.0,
                volatility=15.0,
                sharpe_ratio=1.0,
                max_drawdown=5.0  # Drawdown must be negative
            )
        assert "Input should be less than or equal to 0" in str(exc_info.value)
    
    def test_invalid_max_drawdown_below_minus_100(self):
        """Test validation fails for max drawdown < -100%."""
        with pytest.raises(ValidationError) as exc_info:
            RiskMetrics(
                alpha=2.0,
                beta=1.0,
                volatility=15.0,
                sharpe_ratio=1.0,
                max_drawdown=-150.0  # Cannot exceed -100%
            )
        assert "cannot exceed -100%" in str(exc_info.value)


class TestErrorResponse:
    """Test cases for ErrorResponse model."""
    
    def test_valid_error_response(self):
        """Test valid error response creation."""
        error = ErrorResponse(
            error_type="ValidationError",
            error_message="Invalid input provided",
            error_code=400,
            suggested_action="Check input parameters and try again"
        )
        assert error.error_type == "ValidationError"
        assert error.error_message == "Invalid input provided"
        assert error.error_code == 400
        assert error.suggested_action == "Check input parameters and try again"
        assert error.timestamp is not None