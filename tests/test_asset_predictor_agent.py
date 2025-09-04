"""
Unit tests for the Asset Predictor Agent.

This module contains comprehensive tests for asset return prediction functionality,
including historical analysis, volatility adjustments, market regime detection,
and confidence interval calculations.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from agents.asset_predictor_agent import (
    AssetPredictorAgent,
    PredictionInput,
    AssetPrediction,
    PredictionResult,
    MarketRegime,
    create_asset_predictor_agent
)
from models.data_models import AssetReturns


class TestAssetPredictorAgent:
    """Test suite for AssetPredictorAgent class."""
    
    @pytest.fixture
    def sample_historical_data(self) -> List[AssetReturns]:
        """Create sample historical data for testing."""
        data = []
        
        # Create 15 years of sample data with realistic returns
        base_year = 2009
        
        # Sample returns based on historical patterns
        sample_returns = [
            # Year, SP500, SmallCap, TBills, TBonds, CorpBonds, RealEstate, Gold
            (2009, 0.264, 0.278, 0.001, -0.111, 0.201, 0.276, 0.236),
            (2010, 0.151, 0.269, 0.001, 0.085, 0.154, 0.178, 0.296),
            (2011, 0.021, -0.033, 0.001, 0.169, 0.075, 0.088, 0.101),
            (2012, 0.160, 0.163, 0.001, 0.021, 0.154, 0.176, 0.070),
            (2013, 0.322, 0.434, 0.001, -0.090, -0.015, 0.021, -0.281),
            (2014, 0.136, 0.049, 0.001, 0.251, 0.075, 0.301, -0.017),
            (2015, 0.014, -0.041, 0.001, 0.013, -0.005, 0.024, -0.103),
            (2016, 0.120, 0.213, 0.003, 0.006, 0.134, 0.076, 0.085),
            (2017, 0.217, 0.143, 0.010, 0.024, 0.064, 0.050, 0.133),
            (2018, -0.043, -0.111, 0.018, 0.001, -0.021, -0.041, -0.018),
            (2019, 0.315, 0.226, 0.021, 0.069, 0.137, 0.226, 0.184),
            (2020, 0.184, 0.199, 0.006, 0.080, 0.094, -0.021, 0.249),
            (2021, 0.288, 0.143, 0.001, -0.024, -0.010, 0.434, -0.037),
            (2022, -0.181, -0.206, 0.015, -0.130, -0.156, -0.256, 0.001),
            (2023, 0.264, 0.169, 0.046, -0.031, 0.084, 0.111, 0.134)
        ]
        
        for i, returns in enumerate(sample_returns):
            year, sp500, small_cap, t_bills, t_bonds, corp_bonds, real_estate, gold = returns
            
            asset_return = AssetReturns(
                year=year,
                sp500=sp500,
                small_cap=small_cap,
                t_bills=t_bills,
                t_bonds=t_bonds,
                corporate_bonds=corp_bonds,
                real_estate=real_estate,
                gold=gold
            )
            data.append(asset_return)
        
        return data
    
    @pytest.fixture
    def agent(self) -> AssetPredictorAgent:
        """Create AssetPredictorAgent instance for testing."""
        return AssetPredictorAgent()
    
    @pytest.fixture
    def prediction_input(self, sample_historical_data) -> PredictionInput:
        """Create PredictionInput for testing."""
        return PredictionInput(
            historical_data=sample_historical_data,
            lookback_years=10,
            volatility_adjustment=True,
            market_regime_analysis=True,
            risk_free_rate=0.02
        )
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = AssetPredictorAgent()
        
        assert agent.llm is None
        assert len(agent.tools) == 4
        assert agent.prompt is not None
        assert agent.agent_executor is None
    
    def test_agent_initialization_with_llm(self):
        """Test agent initialization with LLM."""
        mock_llm = Mock()
        agent = AssetPredictorAgent(llm=mock_llm)
        
        assert agent.llm == mock_llm
        assert len(agent.tools) == 4
        assert agent.agent_executor is not None
    
    def test_convert_to_dataframe(self, agent, sample_historical_data):
        """Test conversion of AssetReturns list to DataFrame."""
        df = agent._convert_to_dataframe(sample_historical_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_historical_data)
        assert 'year' in df.columns
        assert 'sp500' in df.columns
        assert 'small_cap' in df.columns
        assert df['year'].is_monotonic_increasing  # Should be sorted by year
    
    def test_calculate_historical_statistics(self, agent, sample_historical_data):
        """Test historical statistics calculation."""
        df = agent._convert_to_dataframe(sample_historical_data)
        stats = agent._calculate_historical_statistics(df, lookback_years=10)
        
        # Check that all asset classes are included
        expected_assets = ['sp500', 'small_cap', 't_bills', 't_bonds', 
                          'corporate_bonds', 'real_estate', 'gold']
        
        for asset in expected_assets:
            assert asset in stats
            assert 'mean' in stats[asset]
            assert 'std' in stats[asset]
            assert 'count' in stats[asset]
            assert stats[asset]['count'] == 10  # Should use last 10 years
        
        # Check that statistics are reasonable
        assert stats['sp500']['mean'] > 0  # S&P 500 should have positive mean return
        assert stats['sp500']['std'] > 0   # Should have positive volatility
        assert stats['t_bills']['std'] < stats['sp500']['std']  # T-bills less volatile
    
    def test_apply_volatility_adjustments(self, agent):
        """Test volatility adjustment calculations."""
        # Create mock historical statistics
        historical_stats = {
            'sp500': {'mean': 0.10, 'std': 0.16, 'count': 10},
            't_bills': {'mean': 0.02, 'std': 0.01, 'count': 10}
        }
        
        adjusted_stats = agent._apply_volatility_adjustments(historical_stats, risk_free_rate=0.02)
        
        # Check that adjustments were applied
        assert 'volatility_adjusted_mean' in adjusted_stats['sp500']
        assert 'volatility_penalty' in adjusted_stats['sp500']
        
        # High volatility asset should have lower adjusted return
        sp500_original = historical_stats['sp500']['mean']
        sp500_adjusted = adjusted_stats['sp500']['volatility_adjusted_mean']
        assert sp500_adjusted < sp500_original
        
        # Low volatility asset should have minimal adjustment
        tbills_penalty = adjusted_stats['t_bills']['volatility_penalty']
        assert tbills_penalty >= 0  # Should have minimal or no penalty
    
    def test_calculate_regime_indicators(self, agent, sample_historical_data):
        """Test market regime indicator calculations."""
        df = agent._convert_to_dataframe(sample_historical_data)
        recent_data = df.tail(3)  # Last 3 years
        
        indicators = agent._calculate_regime_indicators(recent_data)
        
        assert 'avg_volatility' in indicators
        assert 'avg_return' in indicators
        assert 'correlation_breakdown' in indicators
        assert 'trend_strength' in indicators
        
        # Check that indicators are within reasonable ranges
        assert 0 <= indicators['trend_strength'] <= 1
        assert 0 <= indicators['correlation_breakdown'] <= 1
        assert indicators['avg_volatility'] >= 0
    
    def test_detect_market_regime(self, agent):
        """Test market regime detection logic."""
        # Test high volatility regime
        high_vol_indicators = {
            'avg_volatility': 0.25,
            'avg_return': 0.05,
            'trend_strength': 0.5,
            'correlation_breakdown': 0.2
        }
        regime = agent._detect_market_regime(high_vol_indicators)
        assert regime == MarketRegime.HIGH_VOLATILITY
        
        # Test bull market regime
        bull_indicators = {
            'avg_volatility': 0.15,
            'avg_return': 0.08,
            'trend_strength': 0.8,
            'correlation_breakdown': 0.2
        }
        regime = agent._detect_market_regime(bull_indicators)
        assert regime == MarketRegime.BULL_MARKET
        
        # Test bear market regime
        bear_indicators = {
            'avg_volatility': 0.15,
            'avg_return': -0.08,
            'trend_strength': 0.8,
            'correlation_breakdown': 0.2
        }
        regime = agent._detect_market_regime(bear_indicators)
        assert regime == MarketRegime.BEAR_MARKET
        
        # Test normal market regime
        normal_indicators = {
            'avg_volatility': 0.15,
            'avg_return': 0.03,
            'trend_strength': 0.6,
            'correlation_breakdown': 0.2
        }
        regime = agent._detect_market_regime(normal_indicators)
        assert regime == MarketRegime.NORMAL_MARKET
    
    def test_apply_regime_adjustments(self, agent):
        """Test regime-based return adjustments."""
        # Create mock statistics
        stats = {
            'sp500': {'mean': 0.10, 'volatility_adjusted_mean': 0.095},
            'gold': {'mean': 0.07, 'volatility_adjusted_mean': 0.065},
            't_bonds': {'mean': 0.05, 'volatility_adjusted_mean': 0.048}
        }
        
        # Test bull market adjustments
        bull_adjusted = agent._apply_regime_adjustments(stats, MarketRegime.BULL_MARKET)
        
        # Equities should get boost in bull market
        assert bull_adjusted['sp500']['regime_adjusted_mean'] > stats['sp500']['volatility_adjusted_mean']
        # Gold should get penalty in bull market
        assert bull_adjusted['gold']['regime_adjusted_mean'] < stats['gold']['volatility_adjusted_mean']
        
        # Test bear market adjustments
        bear_adjusted = agent._apply_regime_adjustments(stats, MarketRegime.BEAR_MARKET)
        
        # Equities should get penalty in bear market
        assert bear_adjusted['sp500']['regime_adjusted_mean'] < stats['sp500']['volatility_adjusted_mean']
        # Gold should get boost in bear market
        assert bear_adjusted['gold']['regime_adjusted_mean'] > stats['gold']['volatility_adjusted_mean']
    
    def test_calculate_confidence_intervals(self, agent, sample_historical_data):
        """Test confidence interval calculations."""
        df = agent._convert_to_dataframe(sample_historical_data)
        
        # Create mock final statistics
        final_stats = {
            'sp500': {'regime_adjusted_mean': 0.095, 'std': 0.16, 'count': 10, 'mean': 0.10},
            't_bills': {'regime_adjusted_mean': 0.02, 'std': 0.01, 'count': 10, 'mean': 0.025}
        }
        
        intervals = agent._calculate_confidence_intervals(df, final_stats, lookback_years=10)
        
        # Check that intervals are calculated for all assets
        assert 'sp500' in intervals
        assert 't_bills' in intervals
        
        # Check interval structure
        sp500_lower, sp500_upper = intervals['sp500']
        assert sp500_lower < final_stats['sp500']['regime_adjusted_mean']
        assert sp500_upper > final_stats['sp500']['regime_adjusted_mean']
        assert sp500_lower < sp500_upper
        
        # Higher volatility asset should have wider interval
        sp500_width = sp500_upper - sp500_lower
        tbills_lower, tbills_upper = intervals['t_bills']
        tbills_width = tbills_upper - tbills_lower
        assert sp500_width > tbills_width
    
    def test_generate_asset_predictions(self, agent):
        """Test final asset prediction generation."""
        # Create mock statistics for all steps
        historical_stats = {
            'sp500': {'mean': 0.10, 'std': 0.16, 'count': 10}
        }
        
        volatility_adjusted = {
            'sp500': {'volatility_adjusted_mean': 0.095, 'std': 0.16, 'count': 10}
        }
        
        regime_adjusted = {
            'sp500': {'regime_adjusted_mean': 0.098, 'std': 0.16, 'count': 10, 'mean': 0.10}
        }
        
        confidence_intervals = {
            'sp500': (0.078, 0.118)
        }
        
        predictions = agent._generate_asset_predictions(
            historical_stats,
            volatility_adjusted,
            regime_adjusted,
            confidence_intervals,
            risk_free_rate=0.02
        )
        
        assert 'sp500' in predictions
        prediction = predictions['sp500']
        
        assert isinstance(prediction, AssetPrediction)
        assert prediction.asset_name == 'S&P 500'
        assert prediction.expected_return == 0.098
        assert prediction.volatility == 0.16
        assert prediction.confidence_interval == (0.078, 0.118)
        assert prediction.sharpe_ratio > 0  # Should be positive for equity
    
    def test_predict_returns_success(self, agent, prediction_input):
        """Test successful return prediction pipeline."""
        result = agent.predict_returns(prediction_input)
        
        assert isinstance(result, PredictionResult)
        assert result.success is True
        assert result.error_message is None
        
        # Check that predictions were generated for all asset classes
        expected_assets = ['sp500', 'small_cap', 't_bills', 't_bonds', 
                          'corporate_bonds', 'real_estate', 'gold']
        
        for asset in expected_assets:
            assert asset in result.predictions
            prediction = result.predictions[asset]
            assert isinstance(prediction, AssetPrediction)
            assert prediction.expected_return is not None
            assert prediction.volatility > 0
        
        # Check metadata
        assert result.market_regime in MarketRegime
        assert result.analysis_period['lookback_years'] == 10
        assert result.methodology_summary['volatility_adjustment'] is True
        assert result.methodology_summary['market_regime_analysis'] is True
    
    def test_predict_returns_with_different_parameters(self, agent, sample_historical_data):
        """Test prediction with different parameter combinations."""
        # Test without volatility adjustment
        input_no_vol = PredictionInput(
            historical_data=sample_historical_data,
            lookback_years=5,
            volatility_adjustment=False,
            market_regime_analysis=True,
            risk_free_rate=0.025
        )
        
        result = agent.predict_returns(input_no_vol)
        assert result.success is True
        assert result.methodology_summary['volatility_adjustment'] is False
        assert result.analysis_period['lookback_years'] == 5
        
        # Test without regime analysis
        input_no_regime = PredictionInput(
            historical_data=sample_historical_data,
            lookback_years=15,
            volatility_adjustment=True,
            market_regime_analysis=False,
            risk_free_rate=0.015
        )
        
        result = agent.predict_returns(input_no_regime)
        assert result.success is True
        assert result.methodology_summary['market_regime_analysis'] is False
    
    def test_predict_returns_with_insufficient_data(self, agent):
        """Test prediction with insufficient historical data."""
        # Create minimal data (only 2 years)
        minimal_data = [
            AssetReturns(year=2022, sp500=0.1, small_cap=0.12, t_bills=0.02, 
                        t_bonds=0.05, corporate_bonds=0.06, real_estate=0.09, gold=0.07),
            AssetReturns(year=2023, sp500=0.15, small_cap=0.18, t_bills=0.025, 
                        t_bonds=0.04, corporate_bonds=0.055, real_estate=0.11, gold=0.08)
        ]
        
        input_minimal = PredictionInput(
            historical_data=minimal_data,
            lookback_years=10,  # More than available data
            volatility_adjustment=True,
            market_regime_analysis=True,
            risk_free_rate=0.02
        )
        
        result = agent.predict_returns(input_minimal)
        
        # Should still succeed but use available data
        assert result.success is True
        assert len(result.predictions) > 0
    
    def test_get_prediction_summary(self, agent, prediction_input):
        """Test prediction summary generation."""
        result = agent.predict_returns(prediction_input)
        summary = agent.get_prediction_summary(result)
        
        assert isinstance(summary, str)
        assert "ASSET RETURN PREDICTIONS" in summary
        assert "Expected Annual Returns" in summary
        assert "Methodology Applied" in summary
        
        # Check that all assets are included in summary
        for asset_name in ['S&P 500', 'US Small Cap', 'Treasury Bills']:
            assert asset_name in summary
    
    def test_get_prediction_summary_failure(self, agent):
        """Test prediction summary for failed prediction."""
        failed_result = PredictionResult(
            success=False,
            predictions={},
            market_regime=MarketRegime.NORMAL_MARKET,
            analysis_period={},
            methodology_summary={},
            error_message="Test error"
        )
        
        summary = agent.get_prediction_summary(failed_result)
        assert "Prediction failed: Test error" in summary
    
    def test_create_asset_predictor_agent_factory(self):
        """Test factory function for creating agent."""
        agent = create_asset_predictor_agent()
        assert isinstance(agent, AssetPredictorAgent)
        assert agent.llm is None
        
        mock_llm = Mock()
        agent_with_llm = create_asset_predictor_agent(llm=mock_llm)
        assert isinstance(agent_with_llm, AssetPredictorAgent)
        assert agent_with_llm.llm == mock_llm


class TestPredictionModels:
    """Test suite for prediction-related Pydantic models."""
    
    def test_prediction_input_validation(self):
        """Test PredictionInput model validation."""
        # Valid input
        valid_data = [
            AssetReturns(year=2023, sp500=0.1, small_cap=0.12, t_bills=0.02, 
                        t_bonds=0.05, corporate_bonds=0.06, real_estate=0.09, gold=0.07)
        ]
        
        input_model = PredictionInput(
            historical_data=valid_data,
            lookback_years=10,
            volatility_adjustment=True,
            market_regime_analysis=True,
            risk_free_rate=0.02
        )
        
        assert input_model.lookback_years == 10
        assert input_model.risk_free_rate == 0.02
        
        # Test validation errors
        with pytest.raises(ValueError):
            PredictionInput(
                historical_data=valid_data,
                lookback_years=2,  # Below minimum
                risk_free_rate=0.02
            )
        
        with pytest.raises(ValueError):
            PredictionInput(
                historical_data=valid_data,
                lookback_years=10,
                risk_free_rate=0.15  # Above maximum
            )
    
    def test_asset_prediction_model(self):
        """Test AssetPrediction model creation."""
        prediction = AssetPrediction(
            asset_name="S&P 500",
            expected_return=0.095,
            volatility=0.16,
            confidence_interval=(0.075, 0.115),
            historical_mean=0.10,
            volatility_adjusted_return=0.095,
            regime_adjusted_return=0.098,
            sharpe_ratio=0.47
        )
        
        assert prediction.asset_name == "S&P 500"
        assert prediction.expected_return == 0.095
        assert prediction.confidence_interval == (0.075, 0.115)
    
    def test_prediction_result_model(self):
        """Test PredictionResult model creation."""
        predictions = {
            'sp500': AssetPrediction(
                asset_name="S&P 500",
                expected_return=0.095,
                volatility=0.16,
                confidence_interval=(0.075, 0.115),
                historical_mean=0.10,
                volatility_adjusted_return=0.095,
                regime_adjusted_return=0.098,
                sharpe_ratio=0.47
            )
        }
        
        result = PredictionResult(
            success=True,
            predictions=predictions,
            market_regime=MarketRegime.NORMAL_MARKET,
            analysis_period={'start_year': 2009, 'end_year': 2023, 'lookback_years': 10},
            methodology_summary={'volatility_adjustment': True}
        )
        
        assert result.success is True
        assert len(result.predictions) == 1
        assert result.market_regime == MarketRegime.NORMAL_MARKET


class TestMarketRegimeAnalysis:
    """Test suite for market regime analysis functionality."""
    
    def test_market_regime_enum(self):
        """Test MarketRegime enum values."""
        assert MarketRegime.BULL_MARKET.value == "bull_market"
        assert MarketRegime.BEAR_MARKET.value == "bear_market"
        assert MarketRegime.NORMAL_MARKET.value == "normal_market"
        assert MarketRegime.HIGH_VOLATILITY.value == "high_volatility"
        assert MarketRegime.LOW_VOLATILITY.value == "low_volatility"
    
    def test_regime_detection_edge_cases(self):
        """Test regime detection with edge case indicators."""
        agent = AssetPredictorAgent()
        
        # Test with zero volatility
        zero_vol_indicators = {
            'avg_volatility': 0.0,
            'avg_return': 0.05,
            'trend_strength': 0.5,
            'correlation_breakdown': 0.2
        }
        regime = agent._detect_market_regime(zero_vol_indicators)
        assert regime == MarketRegime.LOW_VOLATILITY
        
        # Test with extreme values
        extreme_indicators = {
            'avg_volatility': 1.0,  # 100% volatility
            'avg_return': 0.5,      # 50% return
            'trend_strength': 1.0,  # Perfect trend
            'correlation_breakdown': 1.0  # Complete breakdown
        }
        regime = agent._detect_market_regime(extreme_indicators)
        assert regime == MarketRegime.HIGH_VOLATILITY  # Volatility takes precedence


if __name__ == "__main__":
    pytest.main([__file__])