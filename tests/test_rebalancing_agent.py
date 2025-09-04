"""
Unit tests for Rebalancing Agent

Tests the rebalancing agent functionality including
time-based rebalancing and allocation adjustments.
"""

import pytest
import logging
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from agents.rebalancing_agent import RebalancingAgent, RebalancingInput, RebalancingOutput


class TestRebalancingAgent:
    """Test suite for Rebalancing Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create rebalancing agent for testing"""
        return RebalancingAgent()
    
    @pytest.fixture
    def sample_initial_allocation(self):
        """Sample initial portfolio allocation"""
        return {
            'sp500': 40.0,
            'small_cap': 20.0,
            't_bills': 15.0,
            't_bonds': 15.0,
            'corporate_bonds': 5.0,
            'real_estate': 3.0,
            'gold': 2.0
        }
    
    @pytest.fixture
    def sample_state(self, sample_initial_allocation):
        """Sample state for testing"""
        return {
            'portfolio_allocation': sample_initial_allocation,
            'investment_horizon': 10,
            'rebalancing_frequency': 2,
            'equity_reduction_rate': 5.0
        }
    
    @pytest.fixture
    def sample_predicted_returns(self):
        """Sample predicted returns for testing"""
        return {
            'sp500': 0.10,
            'small_cap': 0.11,
            't_bills': 0.03,
            't_bonds': 0.05,
            'corporate_bonds': 0.06,
            'real_estate': 0.08,
            'gold': 0.07
        }
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        agent = RebalancingAgent()
        
        assert agent.name == "rebalancing_agent"
        assert agent.equity_assets == ['sp500', 'small_cap']
        assert agent.bond_assets == ['t_bills', 't_bonds', 'corporate_bonds']
        assert agent.alternative_assets == ['real_estate', 'gold']
    
    def test_create_rebalancing_strategy_success(self, agent, sample_state):
        """Test successful rebalancing strategy creation"""
        result = agent.create_rebalancing_strategy(sample_state)
        
        # Verify successful completion
        assert result['agent_status'] == 'rebalancing_complete'
        assert 'error' not in result
        
        # Verify rebalancing schedule structure
        schedule = result['rebalancing_schedule']
        assert isinstance(schedule, list)
        assert len(schedule) > 1  # Should have initial + rebalancing events
        
        # Verify initial event (year 0)
        initial_event = schedule[0]
        assert initial_event['year'] == 0
        assert initial_event['rationale'] == 'Initial portfolio allocation'
        assert 'allocation' in initial_event
        
        # Verify rebalancing events
        rebalancing_events = [event for event in schedule if event['year'] > 0]
        assert len(rebalancing_events) > 0
        
        for event in rebalancing_events:
            assert 'year' in event
            assert 'allocation' in event
            assert 'changes' in event
            assert 'rationale' in event
            assert event['year'] % 2 == 0  # Should be every 2 years
        
        # Verify final allocation
        assert 'final_allocation' in result
        final_allocation = result['final_allocation']
        assert abs(sum(final_allocation.values()) - 100.0) < 0.01
        
        # Verify rationale
        assert 'rebalancing_rationale' in result
        assert isinstance(result['rebalancing_rationale'], str)
    
    def test_create_rebalancing_strategy_no_initial_allocation(self, agent):
        """Test handling when no initial allocation is provided"""
        state = {
            'investment_horizon': 10,
            'rebalancing_frequency': 2,
            'equity_reduction_rate': 5.0
        }
        
        result = agent.create_rebalancing_strategy(state)
        
        # Should fail gracefully
        assert result['agent_status'] == 'rebalancing_failed'
        assert 'error' in result
    
    def test_create_rebalancing_schedule(self, agent, sample_initial_allocation):
        """Test rebalancing schedule creation"""
        schedule = agent._create_rebalancing_schedule(
            sample_initial_allocation, 10, 2, 5.0
        )
        
        # Verify schedule structure
        assert isinstance(schedule, list)
        assert len(schedule) == 6  # Year 0, 2, 4, 6, 8, 10
        
        # Verify years are correct
        years = [event['year'] for event in schedule]
        assert years == [0, 2, 4, 6, 8, 10]
        
        # Verify equity reduction over time
        initial_equity = (sample_initial_allocation['sp500'] + 
                         sample_initial_allocation['small_cap'])
        
        for i, event in enumerate(schedule):
            if i > 0:  # Skip initial allocation
                current_equity = (event['allocation']['sp500'] + 
                                event['allocation']['small_cap'])
                # Equity should decrease over time
                assert current_equity < initial_equity
    
    def test_apply_rebalancing_rules(self, agent, sample_initial_allocation):
        """Test application of rebalancing rules"""
        new_allocation, changes = agent._apply_rebalancing_rules(
            sample_initial_allocation, 5.0, 2
        )
        
        # Verify allocation structure
        assert isinstance(new_allocation, dict)
        assert isinstance(changes, dict)
        assert abs(sum(new_allocation.values()) - 100.0) < 0.01
        
        # Verify equity reduction
        initial_equity = (sample_initial_allocation['sp500'] + 
                         sample_initial_allocation['small_cap'])
        new_equity = new_allocation['sp500'] + new_allocation['small_cap']
        
        assert new_equity < initial_equity
        
        # Verify bond increase
        initial_bonds = (sample_initial_allocation['t_bills'] + 
                        sample_initial_allocation['t_bonds'] + 
                        sample_initial_allocation['corporate_bonds'])
        new_bonds = (new_allocation['t_bills'] + 
                    new_allocation['t_bonds'] + 
                    new_allocation['corporate_bonds'])
        
        assert new_bonds > initial_bonds
        
        # Verify changes are recorded
        assert len(changes) > 0
        equity_changes = sum(changes.get(asset, 0) for asset in agent.equity_assets)
        bond_changes = sum(changes.get(asset, 0) for asset in agent.bond_assets)
        
        assert equity_changes < 0  # Equity should decrease
        assert bond_changes > 0    # Bonds should increase
    
    def test_apply_rebalancing_rules_no_equity(self, agent):
        """Test rebalancing when there's no equity allocation"""
        no_equity_allocation = {
            'sp500': 0.0,
            'small_cap': 0.0,
            't_bills': 50.0,
            't_bonds': 30.0,
            'corporate_bonds': 20.0
        }
        
        new_allocation, changes = agent._apply_rebalancing_rules(
            no_equity_allocation, 5.0, 2
        )
        
        # Should return unchanged allocation
        assert new_allocation == no_equity_allocation
        assert changes == {}
    
    def test_redistribute_to_bonds(self, agent):
        """Test redistribution of equity reduction to bonds"""
        allocation = {
            'sp500': 40.0,
            'small_cap': 20.0,
            't_bills': 15.0,
            't_bonds': 15.0,
            'corporate_bonds': 10.0
        }
        
        bond_increases = agent._redistribute_to_bonds(allocation, 10.0)
        
        # Verify redistribution
        assert isinstance(bond_increases, dict)
        assert len(bond_increases) == 3  # Three bond assets
        
        # Total increase should equal redistribution amount
        total_increase = sum(bond_increases.values())
        assert abs(total_increase - 10.0) < 0.01
        
        # All increases should be positive
        assert all(increase > 0 for increase in bond_increases.values())
    
    def test_redistribute_to_bonds_no_existing_bonds(self, agent):
        """Test redistribution when no bonds exist initially"""
        allocation = {
            'sp500': 60.0,
            'small_cap': 40.0,
            't_bills': 0.0,
            't_bonds': 0.0,
            'corporate_bonds': 0.0
        }
        
        bond_increases = agent._redistribute_to_bonds(allocation, 15.0)
        
        # Should distribute equally among bond assets
        assert len(bond_increases) == 3
        expected_increase = 15.0 / 3
        
        for increase in bond_increases.values():
            assert abs(increase - expected_increase) < 0.01
    
    def test_normalize_allocation(self, agent):
        """Test allocation normalization"""
        # Test normal case
        allocation = {'sp500': 50.0, 'bonds': 30.0, 'other': 20.0}
        normalized = agent._normalize_allocation(allocation)
        
        assert abs(sum(normalized.values()) - 100.0) < 0.01
        
        # Test with values that don't sum to 100%
        allocation = {'sp500': 60.0, 'bonds': 30.0, 'other': 15.0}  # Sums to 105%
        normalized = agent._normalize_allocation(allocation)
        
        assert abs(sum(normalized.values()) - 100.0) < 0.01
        
        # Test with negative values
        allocation = {'sp500': 60.0, 'bonds': -10.0, 'other': 50.0}
        normalized = agent._normalize_allocation(allocation)
        
        assert all(value >= 0 for value in normalized.values())
        assert abs(sum(normalized.values()) - 100.0) < 0.01
    
    def test_generate_rebalancing_rationale(self, agent, sample_initial_allocation):
        """Test rebalancing rationale generation"""
        final_allocation = {
            'sp500': 30.0,  # Reduced from 40%
            'small_cap': 15.0,  # Reduced from 20%
            't_bills': 20.0,  # Increased from 15%
            't_bonds': 20.0,  # Increased from 15%
            'corporate_bonds': 10.0,  # Increased from 5%
            'real_estate': 3.0,
            'gold': 2.0
        }
        
        rationale = agent._generate_rebalancing_rationale(
            sample_initial_allocation, final_allocation, 10, 2, 5.0
        )
        
        # Verify rationale content
        assert isinstance(rationale, str)
        assert len(rationale) > 0
        assert '10-year horizon' in rationale
        assert 'Every 2 years' in rationale
        assert '5.0%' in rationale
        assert 'equity allocation' in rationale.lower()
        assert 'bond allocation' in rationale.lower()
    
    def test_calculate_rebalancing_impact(self, agent, sample_state, sample_predicted_returns):
        """Test calculation of rebalancing impact on portfolio projections"""
        # First create rebalancing strategy
        state_with_schedule = agent.create_rebalancing_strategy(sample_state)
        state_with_schedule['predicted_returns'] = sample_predicted_returns
        state_with_schedule['investment_amount'] = 100000.0
        
        # Calculate impact
        result = agent.calculate_rebalancing_impact(state_with_schedule)
        
        # Verify projections are added
        assert 'portfolio_projections_with_rebalancing' in result
        projections = result['portfolio_projections_with_rebalancing']
        
        assert isinstance(projections, list)
        assert len(projections) > 0
        
        # Verify projection structure
        for projection in projections:
            assert 'year' in projection
            assert 'portfolio_value' in projection
            assert 'allocation' in projection
            assert 'expected_return' in projection
    
    def test_calculate_portfolio_projections(self, agent, sample_predicted_returns):
        """Test portfolio value projections with rebalancing"""
        rebalancing_schedule = [
            {
                'year': 0,
                'allocation': {'sp500': 50.0, 't_bills': 50.0}
            },
            {
                'year': 2,
                'allocation': {'sp500': 40.0, 't_bills': 60.0}
            }
        ]
        
        projections = agent._calculate_portfolio_projections(
            rebalancing_schedule, sample_predicted_returns, 100000.0
        )
        
        # Verify projections structure
        assert isinstance(projections, list)
        assert len(projections) == 2
        
        for projection in projections:
            assert 'year' in projection
            assert 'portfolio_value' in projection
            assert 'expected_return' in projection
            assert projection['portfolio_value'] > 0
            assert projection['expected_return'] > 0
    
    def test_create_runnable(self, agent):
        """Test creation of LangChain runnable"""
        runnable = agent.create_runnable()
        
        # Verify runnable is created
        assert runnable is not None
        
        # Verify runnable can be invoked
        sample_state = {
            'portfolio_allocation': {'sp500': 50.0, 't_bills': 50.0},
            'investment_horizon': 5,
            'rebalancing_frequency': 2,
            'equity_reduction_rate': 5.0
        }
        
        result = runnable.invoke(sample_state)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'rebalancing_schedule' in result
    
    def test_rebalancing_with_different_frequencies(self, agent, sample_initial_allocation):
        """Test rebalancing with different frequencies"""
        frequencies = [1, 2, 3, 5]
        horizon = 10
        
        for frequency in frequencies:
            schedule = agent._create_rebalancing_schedule(
                sample_initial_allocation, horizon, frequency, 5.0
            )
            
            # Verify correct number of events
            expected_events = len(list(range(0, horizon + 1, frequency)))
            if horizon % frequency != 0:
                expected_events += 1  # Add final year if not aligned
            
            # Should have at least initial event plus rebalancing events
            assert len(schedule) >= 2
            
            # Verify years are multiples of frequency
            rebalancing_years = [event['year'] for event in schedule if event['year'] > 0]
            for year in rebalancing_years[:-1]:  # Exclude potential final year
                assert year % frequency == 0
    
    def test_rebalancing_with_different_reduction_rates(self, agent, sample_initial_allocation):
        """Test rebalancing with different equity reduction rates"""
        reduction_rates = [2.0, 5.0, 10.0, 15.0]
        
        for rate in reduction_rates:
            schedule = agent._create_rebalancing_schedule(
                sample_initial_allocation, 10, 2, rate
            )
            
            # Verify equity reduction is proportional to rate
            initial_equity = (sample_initial_allocation['sp500'] + 
                             sample_initial_allocation['small_cap'])
            
            # Check first rebalancing event
            first_rebalance = schedule[1]  # Index 1 is first rebalancing (year 2)
            new_equity = (first_rebalance['allocation']['sp500'] + 
                         first_rebalance['allocation']['small_cap'])
            
            # Equity should be reduced by approximately the specified rate
            expected_reduction = initial_equity * (rate / 100.0)
            actual_reduction = initial_equity - new_equity
            
            # Allow some tolerance for rounding and redistribution
            assert abs(actual_reduction - expected_reduction) < 2.0
    
    def test_rebalancing_edge_cases(self, agent):
        """Test edge cases in rebalancing"""
        # Test with minimal state
        result = agent.create_rebalancing_strategy({})
        assert result['agent_status'] == 'rebalancing_failed'
        
        # Test with zero horizon
        state = {
            'portfolio_allocation': {'sp500': 100.0},
            'investment_horizon': 0,
            'rebalancing_frequency': 2,
            'equity_reduction_rate': 5.0
        }
        result = agent.create_rebalancing_strategy(state)
        assert result['agent_status'] == 'rebalancing_complete'
        
        # Should have only initial allocation
        schedule = result['rebalancing_schedule']
        assert len(schedule) == 1
        assert schedule[0]['year'] == 0
    
    def test_rebalancing_logging(self, agent, sample_state, caplog):
        """Test logging during rebalancing process"""
        with caplog.at_level(logging.INFO):
            agent.create_rebalancing_strategy(sample_state)
        
        # Verify logging messages
        assert any("Starting rebalancing strategy creation" in record.message for record in caplog.records)
        assert any("Rebalancing strategy created" in record.message for record in caplog.records)
    
    def test_sip_calculation(self, agent):
        """Test SIP (Systematic Investment Plan) calculation"""
        current_value = 100000.0
        monthly_amount = 5000.0
        annual_return = 0.10
        years = 2
        
        result = agent._calculate_sip_value(current_value, monthly_amount, annual_return, years)
        
        # Should be greater than just compound growth of current value
        compound_only = current_value * ((1 + annual_return) ** years)
        assert result > compound_only
        
        # Should include SIP contributions
        expected_sip_contribution = monthly_amount * 12 * years  # Minimum without compounding
        assert result > compound_only + expected_sip_contribution
    
    def test_swp_calculation(self, agent):
        """Test SWP (Systematic Withdrawal Plan) calculation"""
        current_value = 500000.0
        monthly_withdrawal = 2000.0
        annual_return = 0.08
        years = 2
        
        result = agent._calculate_swp_value(current_value, monthly_withdrawal, annual_return, years)
        
        # Should be less than compound growth without withdrawals
        compound_only = current_value * ((1 + annual_return) ** years)
        assert result < compound_only
        
        # Should be positive if withdrawals are sustainable
        assert result > 0
    
    def test_swp_depletion(self, agent):
        """Test SWP calculation when portfolio gets depleted"""
        current_value = 50000.0
        monthly_withdrawal = 10000.0  # Unsustainable withdrawal
        annual_return = 0.05
        years = 1
        
        result = agent._calculate_swp_value(current_value, monthly_withdrawal, annual_return, years)
        
        # Should return 0 when portfolio is depleted
        assert result == 0
    
    def test_no_rebalancing_projections(self, agent, sample_initial_allocation, sample_predicted_returns):
        """Test calculation of projections without rebalancing"""
        projections = agent._calculate_no_rebalancing_projections(
            sample_initial_allocation, sample_predicted_returns, 100000.0, 'lump_sum', 0, 5
        )
        
        # Verify structure
        assert isinstance(projections, list)
        assert len(projections) == 6  # Years 0-5
        
        # Verify all projections use same allocation
        initial_allocation = projections[0]['allocation']
        for projection in projections:
            assert projection['allocation'] == initial_allocation
        
        # Verify portfolio value grows
        for i in range(1, len(projections)):
            assert projections[i]['portfolio_value'] > projections[i-1]['portfolio_value']
    
    def test_rebalancing_benefits_calculation(self, agent):
        """Test calculation of rebalancing benefits"""
        with_rebalancing = [
            {'year': 0, 'portfolio_value': 100000, 'end_period_value': 110000},
            {'year': 2, 'portfolio_value': 110000, 'end_period_value': 125000}
        ]
        
        without_rebalancing = [
            {'year': 0, 'portfolio_value': 100000},
            {'year': 2, 'portfolio_value': 120000}
        ]
        
        benefits = agent._calculate_rebalancing_benefits(with_rebalancing, without_rebalancing)
        
        # Verify benefit calculation
        assert 'final_value_with_rebalancing' in benefits
        assert 'final_value_without_rebalancing' in benefits
        assert 'benefit_amount' in benefits
        assert 'benefit_percentage' in benefits
        assert 'recommendation' in benefits
        
        assert benefits['final_value_with_rebalancing'] == 125000
        assert benefits['final_value_without_rebalancing'] == 120000
        assert benefits['benefit_amount'] == 5000
        assert abs(benefits['benefit_percentage'] - 4.17) < 0.1  # Approximately 4.17%
    
    def test_risk_reduction_calculation(self, agent):
        """Test risk reduction score calculation"""
        projections = [
            {'allocation': {'sp500': 60, 'small_cap': 20, 't_bills': 20}},
            {'allocation': {'sp500': 55, 'small_cap': 15, 't_bills': 30}},
            {'allocation': {'sp500': 50, 'small_cap': 10, 't_bills': 40}}
        ]
        
        risk_reduction = agent._calculate_risk_reduction(projections)
        
        # Should return a positive score for varying equity allocations
        assert isinstance(risk_reduction, float)
        assert risk_reduction >= 0
    
    def test_rebalancing_recommendation_generation(self, agent):
        """Test rebalancing recommendation generation"""
        # High benefit scenario
        recommendation = agent._generate_rebalancing_recommendation(8.0, 15.0)
        assert "Highly recommended" in recommendation
        
        # Moderate benefit scenario
        recommendation = agent._generate_rebalancing_recommendation(3.0, 7.0)
        assert "Recommended" in recommendation
        
        # Small benefit scenario
        recommendation = agent._generate_rebalancing_recommendation(1.0, 2.0)
        assert "Consider" in recommendation
        
        # Minimal benefit scenario
        recommendation = agent._generate_rebalancing_recommendation(-1.0, 1.0)
        assert "Optional" in recommendation
    
    def test_allocation_visualization_data_preparation(self, agent, sample_state):
        """Test preparation of allocation visualization data"""
        # First create rebalancing strategy
        state_with_schedule = agent.create_rebalancing_strategy(sample_state)
        
        # Prepare visualization data
        result = agent.prepare_allocation_visualization_data(state_with_schedule)
        
        # Verify visualization data is added
        assert 'allocation_timeline_data' in result
        assert 'allocation_changes_summary' in result
        
        timeline_data = result['allocation_timeline_data']
        assert isinstance(timeline_data, list)
        assert len(timeline_data) > 0
        
        # Verify timeline data structure
        for point in timeline_data:
            assert 'year' in point
            assert 'sp500' in point
            assert 'small_cap' in point
            assert 't_bills' in point
            assert 't_bonds' in point
            assert 'corporate_bonds' in point
        
        # Verify summary data
        summary = result['allocation_changes_summary']
        assert 'total_equity_change' in summary
        assert 'total_bond_change' in summary
        assert 'rebalancing_events' in summary
    
    def test_allocation_changes_summary(self, agent):
        """Test summarization of allocation changes"""
        rebalancing_schedule = [
            {
                'year': 0,
                'allocation': {'sp500': 50, 'small_cap': 20, 't_bills': 30}
            },
            {
                'year': 2,
                'allocation': {'sp500': 45, 'small_cap': 15, 't_bills': 40}
            },
            {
                'year': 4,
                'allocation': {'sp500': 40, 'small_cap': 10, 't_bills': 50}
            }
        ]
        
        summary = agent._summarize_allocation_changes(rebalancing_schedule)
        
        # Verify summary structure
        assert 'total_equity_change' in summary
        assert 'total_bond_change' in summary
        assert 'largest_increases' in summary
        assert 'largest_decreases' in summary
        assert 'rebalancing_events' in summary
        assert 'years_covered' in summary
        
        # Verify calculations
        assert summary['total_equity_change'] == -20  # 70% to 50%
        assert summary['total_bond_change'] == 20    # 30% to 50%
        assert summary['rebalancing_events'] == 2
        assert summary['years_covered'] == 4
    
    def test_enhanced_calculate_rebalancing_impact(self, agent, sample_state, sample_predicted_returns):
        """Test enhanced rebalancing impact calculation with comparisons"""
        # Setup state with all required data
        state_with_schedule = agent.create_rebalancing_strategy(sample_state)
        state_with_schedule.update({
            'predicted_returns': sample_predicted_returns,
            'investment_amount': 100000.0,
            'investment_type': 'lump_sum',
            'monthly_amount': 0
        })
        
        # Calculate impact
        result = agent.calculate_rebalancing_impact(state_with_schedule)
        
        # Verify all projection types are calculated
        assert 'portfolio_projections_with_rebalancing' in result
        assert 'portfolio_projections_without_rebalancing' in result
        assert 'rebalancing_benefits' in result
        
        # Verify benefits structure
        benefits = result['rebalancing_benefits']
        assert 'final_value_with_rebalancing' in benefits
        assert 'final_value_without_rebalancing' in benefits
        assert 'benefit_amount' in benefits
        assert 'benefit_percentage' in benefits
        assert 'recommendation' in benefits


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    pytest.main([__file__, "-v"])