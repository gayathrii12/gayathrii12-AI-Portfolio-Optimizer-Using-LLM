"""
Unit tests for Portfolio Allocation Engine

Tests all risk profile allocations, validation logic, and edge cases
to ensure proper portfolio allocation functionality.
"""

import pytest
from models.portfolio_allocation_engine import PortfolioAllocationEngine, PortfolioAllocation


class TestPortfolioAllocationEngine:
    """Test suite for Portfolio Allocation Engine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = PortfolioAllocationEngine()
    
    def test_engine_initialization(self):
        """Test that engine initializes correctly."""
        assert self.engine.name == "portfolio_allocation_engine"
        assert len(self.engine.get_supported_risk_profiles()) == 3
        assert set(self.engine.get_supported_risk_profiles()) == {'low', 'moderate', 'high'}
    
    def test_low_risk_allocation(self):
        """Test low risk (conservative) allocation strategy."""
        allocation = self.engine.get_low_risk_allocation()
        
        # Verify it's a PortfolioAllocation object
        assert isinstance(allocation, PortfolioAllocation)
        
        # Test equity/bonds ratio (should be ~30% equity, 70% bonds)
        equity_pct = allocation.get_equity_percentage()
        bonds_pct = allocation.get_bonds_percentage()
        
        assert equity_pct == 30.0  # 20% SP500 + 10% Small Cap
        assert bonds_pct == 70.0   # 25% T-Bills + 30% T-Bonds + 15% Corporate Bonds
        
        # Test individual allocations
        assert allocation.sp500 == 20.0
        assert allocation.small_cap == 10.0
        assert allocation.t_bills == 25.0
        assert allocation.t_bonds == 30.0
        assert allocation.corporate_bonds == 15.0
        assert allocation.real_estate == 0.0
        assert allocation.gold == 0.0
        
        # Test total sums to 100%
        total = sum(allocation.to_dict().values())
        assert abs(total - 100.0) < 0.01
    
    def test_moderate_risk_allocation(self):
        """Test moderate risk (balanced) allocation strategy."""
        allocation = self.engine.get_moderate_risk_allocation()
        
        # Verify it's a PortfolioAllocation object
        assert isinstance(allocation, PortfolioAllocation)
        
        # Test equity/bonds ratio (should be 50% equity, 50% bonds)
        equity_pct = allocation.get_equity_percentage()
        bonds_pct = allocation.get_bonds_percentage()
        
        assert equity_pct == 50.0  # 35% SP500 + 15% Small Cap
        assert bonds_pct == 50.0   # 15% T-Bills + 20% T-Bonds + 15% Corporate Bonds
        
        # Test individual allocations
        assert allocation.sp500 == 35.0
        assert allocation.small_cap == 15.0
        assert allocation.t_bills == 15.0
        assert allocation.t_bonds == 20.0
        assert allocation.corporate_bonds == 15.0
        assert allocation.real_estate == 0.0
        assert allocation.gold == 0.0
        
        # Test total sums to 100%
        total = sum(allocation.to_dict().values())
        assert abs(total - 100.0) < 0.01
    
    def test_high_risk_allocation(self):
        """Test high risk (aggressive) allocation strategy."""
        allocation = self.engine.get_high_risk_allocation()
        
        # Verify it's a PortfolioAllocation object
        assert isinstance(allocation, PortfolioAllocation)
        
        # Test equity/bonds ratio (should be 80% equity, 20% bonds)
        equity_pct = allocation.get_equity_percentage()
        bonds_pct = allocation.get_bonds_percentage()
        
        assert equity_pct == 80.0  # 60% SP500 + 20% Small Cap
        assert bonds_pct == 20.0   # 5% T-Bills + 10% T-Bonds + 5% Corporate Bonds
        
        # Test individual allocations
        assert allocation.sp500 == 60.0
        assert allocation.small_cap == 20.0
        assert allocation.t_bills == 5.0
        assert allocation.t_bonds == 10.0
        assert allocation.corporate_bonds == 5.0
        assert allocation.real_estate == 0.0
        assert allocation.gold == 0.0
        
        # Test total sums to 100%
        total = sum(allocation.to_dict().values())
        assert abs(total - 100.0) < 0.01
    
    def test_get_allocation_by_risk_profile(self):
        """Test getting allocation by risk profile string."""
        # Test all valid risk profiles
        low_allocation = self.engine.get_allocation_by_risk_profile('low')
        moderate_allocation = self.engine.get_allocation_by_risk_profile('moderate')
        high_allocation = self.engine.get_allocation_by_risk_profile('high')
        
        # Verify they match direct method calls
        assert low_allocation.to_dict() == self.engine.get_low_risk_allocation().to_dict()
        assert moderate_allocation.to_dict() == self.engine.get_moderate_risk_allocation().to_dict()
        assert high_allocation.to_dict() == self.engine.get_high_risk_allocation().to_dict()
        
        # Test case insensitivity
        assert self.engine.get_allocation_by_risk_profile('LOW').to_dict() == low_allocation.to_dict()
        assert self.engine.get_allocation_by_risk_profile('MODERATE').to_dict() == moderate_allocation.to_dict()
        assert self.engine.get_allocation_by_risk_profile('HIGH').to_dict() == high_allocation.to_dict()
    
    def test_invalid_risk_profile(self):
        """Test handling of invalid risk profiles."""
        with pytest.raises(ValueError, match="Unsupported risk profile"):
            self.engine.get_allocation_by_risk_profile('invalid')
        
        with pytest.raises(ValueError, match="Unsupported risk profile"):
            self.engine.get_allocation_by_risk_profile('medium')
        
        with pytest.raises(ValueError, match="Unsupported risk profile"):
            self.engine.get_allocation_by_risk_profile('')
    
    def test_allocation_validation_valid(self):
        """Test allocation validation with valid allocations."""
        # Test valid allocation that sums to 100%
        valid_allocation = {
            'sp500': 40.0,
            'small_cap': 10.0,
            't_bills': 20.0,
            't_bonds': 20.0,
            'corporate_bonds': 10.0,
            'real_estate': 0.0,
            'gold': 0.0
        }
        
        assert self.engine.validate_allocation(valid_allocation) is True
        
        # Test all predefined allocations are valid
        for risk_profile in self.engine.get_supported_risk_profiles():
            allocation = self.engine.get_allocation_by_risk_profile(risk_profile)
            assert self.engine.validate_allocation(allocation.to_dict()) is True
    
    def test_allocation_validation_invalid(self):
        """Test allocation validation with invalid allocations."""
        # Test allocation that doesn't sum to 100%
        invalid_allocation_high = {
            'sp500': 50.0,
            'small_cap': 20.0,
            't_bills': 20.0,
            't_bonds': 20.0,
            'corporate_bonds': 10.0,
            'real_estate': 0.0,
            'gold': 0.0
        }  # Sums to 120%
        
        assert self.engine.validate_allocation(invalid_allocation_high) is False
        
        # Test allocation that sums to less than 100%
        invalid_allocation_low = {
            'sp500': 30.0,
            'small_cap': 10.0,
            't_bills': 10.0,
            't_bonds': 10.0,
            'corporate_bonds': 10.0,
            'real_estate': 0.0,
            'gold': 0.0
        }  # Sums to 70%
        
        assert self.engine.validate_allocation(invalid_allocation_low) is False
        
        # Test allocation with negative values
        invalid_allocation_negative = {
            'sp500': -10.0,
            'small_cap': 20.0,
            't_bills': 30.0,
            't_bonds': 30.0,
            'corporate_bonds': 30.0,
            'real_estate': 0.0,
            'gold': 0.0
        }
        
        assert self.engine.validate_allocation(invalid_allocation_negative) is False
        
        # Test allocation with values over 100%
        invalid_allocation_over = {
            'sp500': 150.0,
            'small_cap': 0.0,
            't_bills': 0.0,
            't_bonds': 0.0,
            'corporate_bonds': 0.0,
            'real_estate': 0.0,
            'gold': 0.0
        }
        
        assert self.engine.validate_allocation(invalid_allocation_over) is False
    
    def test_get_allocation_summary(self):
        """Test allocation summary by asset category."""
        # Test low risk summary
        low_summary = self.engine.get_allocation_summary('low')
        assert low_summary['equity'] == 30.0
        assert low_summary['bonds'] == 70.0
        assert low_summary['alternatives'] == 0.0
        
        # Test moderate risk summary
        moderate_summary = self.engine.get_allocation_summary('moderate')
        assert moderate_summary['equity'] == 50.0
        assert moderate_summary['bonds'] == 50.0
        assert moderate_summary['alternatives'] == 0.0
        
        # Test high risk summary
        high_summary = self.engine.get_allocation_summary('high')
        assert high_summary['equity'] == 80.0
        assert high_summary['bonds'] == 20.0
        assert high_summary['alternatives'] == 0.0
    
    def test_portfolio_allocation_model_validation(self):
        """Test PortfolioAllocation model validation."""
        # Test valid allocation
        valid_data = {
            'sp500': 40.0,
            'small_cap': 10.0,
            't_bills': 20.0,
            't_bonds': 20.0,
            'corporate_bonds': 10.0,
            'real_estate': 0.0,
            'gold': 0.0
        }
        
        allocation = PortfolioAllocation(**valid_data)
        assert allocation.sp500 == 40.0
        assert sum(allocation.to_dict().values()) == 100.0
        
        # Test invalid allocation (doesn't sum to 100%)
        invalid_data = {
            'sp500': 50.0,
            'small_cap': 10.0,
            't_bills': 20.0,
            't_bonds': 20.0,
            'corporate_bonds': 10.0,
            'real_estate': 0.0,
            'gold': 0.0
        }  # Sums to 110%
        
        with pytest.raises(ValueError, match="Total allocation must equal 100%"):
            PortfolioAllocation(**invalid_data)
        
        # Test negative values
        negative_data = {
            'sp500': -10.0,
            'small_cap': 10.0,
            't_bills': 40.0,
            't_bonds': 30.0,
            'corporate_bonds': 30.0,
            'real_estate': 0.0,
            'gold': 0.0
        }
        
        with pytest.raises(ValueError):
            PortfolioAllocation(**negative_data)
        
        # Test values over 100%
        over_data = {
            'sp500': 150.0,
            'small_cap': 0.0,
            't_bills': 0.0,
            't_bonds': 0.0,
            'corporate_bonds': 0.0,
            'real_estate': 0.0,
            'gold': 0.0
        }
        
        with pytest.raises(ValueError):
            PortfolioAllocation(**over_data)
    
    def test_portfolio_allocation_helper_methods(self):
        """Test PortfolioAllocation helper methods."""
        allocation = self.engine.get_moderate_risk_allocation()
        
        # Test to_dict method
        allocation_dict = allocation.to_dict()
        assert isinstance(allocation_dict, dict)
        assert len(allocation_dict) == 7
        assert all(isinstance(v, float) for v in allocation_dict.values())
        
        # Test equity percentage calculation
        equity_pct = allocation.get_equity_percentage()
        assert equity_pct == allocation.sp500 + allocation.small_cap
        
        # Test bonds percentage calculation
        bonds_pct = allocation.get_bonds_percentage()
        assert bonds_pct == allocation.t_bills + allocation.t_bonds + allocation.corporate_bonds
        
        # Test alternatives percentage calculation
        alternatives_pct = allocation.get_alternatives_percentage()
        assert alternatives_pct == allocation.real_estate + allocation.gold
        
        # Test that all categories sum to 100%
        total_pct = equity_pct + bonds_pct + alternatives_pct
        assert abs(total_pct - 100.0) < 0.01
    
    def test_risk_profile_progression(self):
        """Test that risk profiles show proper progression from conservative to aggressive."""
        low_allocation = self.engine.get_low_risk_allocation()
        moderate_allocation = self.engine.get_moderate_risk_allocation()
        high_allocation = self.engine.get_high_risk_allocation()
        
        # Equity allocation should increase with risk
        assert low_allocation.get_equity_percentage() < moderate_allocation.get_equity_percentage()
        assert moderate_allocation.get_equity_percentage() < high_allocation.get_equity_percentage()
        
        # Bonds allocation should decrease with risk
        assert low_allocation.get_bonds_percentage() > moderate_allocation.get_bonds_percentage()
        assert moderate_allocation.get_bonds_percentage() > high_allocation.get_bonds_percentage()
        
        # Verify specific ratios match requirements
        assert low_allocation.get_equity_percentage() == 30.0  # 30% equity
        assert low_allocation.get_bonds_percentage() == 70.0   # 70% bonds
        
        assert moderate_allocation.get_equity_percentage() == 50.0  # 50% equity
        assert moderate_allocation.get_bonds_percentage() == 50.0   # 50% bonds
        
        assert high_allocation.get_equity_percentage() == 80.0  # 80% equity
        assert high_allocation.get_bonds_percentage() == 20.0   # 20% bonds