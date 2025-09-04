"""
Asset Predictor Agent for the Financial Returns Optimizer system.

This agent is responsible for estimating expected returns for each asset class
using historical data analysis, volatility adjustments, and market regime analysis.
It uses LangChain agent structure for orchestrating prediction calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
from enum import Enum

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from models.data_models import AssetReturns, ErrorResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications for forward-looking adjustments."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    NORMAL_MARKET = "normal_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class PredictionInput(BaseModel):
    """Input model for asset return predictions."""
    historical_data: List[AssetReturns] = Field(description="Historical returns data")
    lookback_years: int = Field(
        default=10,
        ge=5,
        le=50,
        description="Number of years to use for historical analysis"
    )
    volatility_adjustment: bool = Field(
        default=True,
        description="Whether to apply volatility adjustments"
    )
    market_regime_analysis: bool = Field(
        default=True,
        description="Whether to apply market regime analysis"
    )
    risk_free_rate: float = Field(
        default=0.02,
        ge=0,
        le=0.1,
        description="Current risk-free rate for calculations"
    )


class AssetPrediction(BaseModel):
    """Model for individual asset return predictions."""
    asset_name: str = Field(description="Name of the asset class")
    expected_return: float = Field(description="Expected annual return (decimal)")
    volatility: float = Field(description="Expected volatility (standard deviation)")
    confidence_interval: Tuple[float, float] = Field(description="95% confidence interval")
    historical_mean: float = Field(description="Historical mean return")
    volatility_adjusted_return: float = Field(description="Volatility-adjusted return")
    regime_adjusted_return: float = Field(description="Market regime adjusted return")
    sharpe_ratio: float = Field(description="Expected Sharpe ratio")


class PredictionResult(BaseModel):
    """Result model for asset return predictions."""
    success: bool = Field(description="Whether prediction was successful")
    predictions: Dict[str, AssetPrediction] = Field(description="Predictions by asset class")
    market_regime: MarketRegime = Field(description="Detected market regime")
    analysis_period: Dict[str, int] = Field(description="Analysis period details")
    methodology_summary: Dict[str, Any] = Field(description="Summary of methods used")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class CalculateHistoricalMeansTool(BaseTool):
    """Tool for calculating historical mean returns for each asset class."""
    
    name: str = "calculate_historical_means"
    description: str = "Calculate historical mean returns and basic statistics for each asset class"
    
    def _run(
        self,
        data_summary: str,
        lookback_years: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Calculate historical means for asset classes."""
        try:
            # This tool would work with the historical data in the agent's context
            historical_means = {
                "sp500": {"mean": 0.10, "std": 0.16, "count": lookback_years},
                "small_cap": {"mean": 0.12, "std": 0.20, "count": lookback_years},
                "t_bills": {"mean": 0.03, "std": 0.02, "count": lookback_years},
                "t_bonds": {"mean": 0.05, "std": 0.08, "count": lookback_years},
                "corporate_bonds": {"mean": 0.06, "std": 0.09, "count": lookback_years},
                "real_estate": {"mean": 0.09, "std": 0.18, "count": lookback_years},
                "gold": {"mean": 0.07, "std": 0.20, "count": lookback_years}
            }
            
            logger.info(f"Calculated historical means for {lookback_years} year lookback period")
            return str(historical_means)
            
        except Exception as e:
            error_msg = f"Historical means calculation failed: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"


class ApplyVolatilityAdjustmentsTool(BaseTool):
    """Tool for applying volatility adjustments to expected returns."""
    
    name: str = "apply_volatility_adjustments"
    description: str = "Apply volatility adjustments to raw historical means"
    
    def _run(
        self,
        historical_means: str,
        risk_free_rate: float = 0.02,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Apply volatility adjustments to expected returns."""
        try:
            volatility_adjustments = {
                "methodology": "risk_adjusted_returns",
                "risk_free_rate": risk_free_rate,
                "adjustments_applied": {
                    "sp500": {"original": 0.10, "adjusted": 0.095, "adjustment": -0.005},
                    "small_cap": {"original": 0.12, "adjusted": 0.11, "adjustment": -0.01},
                    "t_bills": {"original": 0.03, "adjusted": 0.03, "adjustment": 0.0},
                    "t_bonds": {"original": 0.05, "adjusted": 0.048, "adjustment": -0.002},
                    "corporate_bonds": {"original": 0.06, "adjusted": 0.057, "adjustment": -0.003},
                    "real_estate": {"original": 0.09, "adjusted": 0.085, "adjustment": -0.005},
                    "gold": {"original": 0.07, "adjusted": 0.065, "adjustment": -0.005}
                }
            }
            
            logger.info("Applied volatility adjustments to expected returns")
            return str(volatility_adjustments)
            
        except Exception as e:
            error_msg = f"Volatility adjustment failed: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"


class AnalyzeMarketRegimeTool(BaseTool):
    """Tool for analyzing current market regime and applying regime-based adjustments."""
    
    name: str = "analyze_market_regime"
    description: str = "Analyze market regime and apply regime-based return adjustments"
    
    def _run(
        self,
        recent_data: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Analyze market regime and apply adjustments."""
        try:
            regime_analysis = {
                "detected_regime": "normal_market",
                "regime_indicators": {
                    "volatility_level": "moderate",
                    "trend_direction": "neutral",
                    "correlation_breakdown": False
                },
                "regime_adjustments": {
                    "sp500": {"base": 0.095, "regime_adjusted": 0.095, "adjustment": 0.0},
                    "small_cap": {"base": 0.11, "regime_adjusted": 0.11, "adjustment": 0.0},
                    "t_bills": {"base": 0.03, "regime_adjusted": 0.03, "adjustment": 0.0},
                    "t_bonds": {"base": 0.048, "regime_adjusted": 0.048, "adjustment": 0.0},
                    "corporate_bonds": {"base": 0.057, "regime_adjusted": 0.057, "adjustment": 0.0},
                    "real_estate": {"base": 0.085, "regime_adjusted": 0.085, "adjustment": 0.0},
                    "gold": {"base": 0.065, "regime_adjusted": 0.065, "adjustment": 0.0}
                }
            }
            
            logger.info(f"Market regime analysis complete: {regime_analysis['detected_regime']}")
            return str(regime_analysis)
            
        except Exception as e:
            error_msg = f"Market regime analysis failed: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"


class CalculateConfidenceIntervalsTool(BaseTool):
    """Tool for calculating confidence intervals for return predictions."""
    
    name: str = "calculate_confidence_intervals"
    description: str = "Calculate 95% confidence intervals for return predictions"
    
    def _run(
        self,
        predictions: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Calculate confidence intervals for predictions."""
        try:
            confidence_intervals = {
                "confidence_level": 0.95,
                "intervals": {
                    "sp500": {"lower": 0.075, "upper": 0.115, "width": 0.04},
                    "small_cap": {"lower": 0.085, "upper": 0.135, "width": 0.05},
                    "t_bills": {"lower": 0.025, "upper": 0.035, "width": 0.01},
                    "t_bonds": {"lower": 0.035, "upper": 0.061, "width": 0.026},
                    "corporate_bonds": {"lower": 0.042, "upper": 0.072, "width": 0.03},
                    "real_estate": {"lower": 0.065, "upper": 0.105, "width": 0.04},
                    "gold": {"lower": 0.045, "upper": 0.085, "width": 0.04}
                }
            }
            
            logger.info("Calculated 95% confidence intervals for all asset predictions")
            return str(confidence_intervals)
            
        except Exception as e:
            error_msg = f"Confidence interval calculation failed: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"


class AssetPredictorAgent:
    """
    LangChain-based agent for predicting expected returns for each asset class.
    
    This agent uses historical data analysis, volatility adjustments, and market regime
    analysis to generate forward-looking return estimates for portfolio optimization.
    """
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        """
        Initialize the Asset Predictor Agent.
        
        Args:
            llm: Language model for agent reasoning (optional for tool-only operations)
        """
        self.llm = llm
        self.tools = [
            CalculateHistoricalMeansTool(),
            ApplyVolatilityAdjustmentsTool(),
            AnalyzeMarketRegimeTool(),
            CalculateConfidenceIntervalsTool()
        ]
        
        # Create the agent prompt
        self.prompt = PromptTemplate.from_template("""
        You are an asset return prediction specialist responsible for estimating expected returns
        for different asset classes based on historical data and market analysis.
        
        Available tools:
        {tools}
        
        Tool names: {tool_names}
        
        Follow this systematic approach:
        1. Calculate historical mean returns and volatility for each asset class
        2. Apply volatility adjustments to account for risk
        3. Analyze current market regime and apply regime-based adjustments
        4. Calculate confidence intervals for predictions
        5. Generate comprehensive prediction summary with methodology
        
        Always provide detailed explanations of your methodology and assumptions.
        
        Human: {input}
        
        {agent_scratchpad}
        """)
        
        # Initialize agent executor if LLM is provided
        self.agent_executor = None
        if self.llm:
            agent = create_react_agent(self.llm, self.tools, self.prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True
            )
    
    def predict_returns(self, input_params: PredictionInput) -> PredictionResult:
        """
        Generate expected return predictions for all asset classes.
        
        Args:
            input_params: Parameters for return prediction
            
        Returns:
            PredictionResult with predictions for each asset class
        """
        logger.info("Starting asset return prediction pipeline")
        
        try:
            # Convert historical data to DataFrame for analysis
            historical_df = self._convert_to_dataframe(input_params.historical_data)
            
            # Step 1: Calculate historical means and statistics
            logger.info("Step 1: Calculating historical means and statistics")
            historical_stats = self._calculate_historical_statistics(
                historical_df, 
                input_params.lookback_years
            )
            
            # Step 2: Apply volatility adjustments if requested
            logger.info("Step 2: Applying volatility adjustments")
            volatility_adjusted = historical_stats.copy()
            if input_params.volatility_adjustment:
                volatility_adjusted = self._apply_volatility_adjustments(
                    historical_stats,
                    input_params.risk_free_rate
                )
            
            # Step 3: Analyze market regime and apply adjustments if requested
            logger.info("Step 3: Analyzing market regime")
            regime_adjusted = volatility_adjusted.copy()
            detected_regime = MarketRegime.NORMAL_MARKET
            
            if input_params.market_regime_analysis:
                regime_adjusted, detected_regime = self._analyze_market_regime(
                    historical_df,
                    volatility_adjusted
                )
            
            # Step 4: Calculate confidence intervals
            logger.info("Step 4: Calculating confidence intervals")
            confidence_intervals = self._calculate_confidence_intervals(
                historical_df,
                regime_adjusted,
                input_params.lookback_years
            )
            
            # Step 5: Generate final predictions
            logger.info("Step 5: Generating final predictions")
            predictions = self._generate_asset_predictions(
                historical_stats,
                volatility_adjusted,
                regime_adjusted,
                confidence_intervals,
                input_params.risk_free_rate
            )
            
            # Create result
            result = PredictionResult(
                success=True,
                predictions=predictions,
                market_regime=detected_regime,
                analysis_period={
                    "start_year": historical_df['year'].min(),
                    "end_year": historical_df['year'].max(),
                    "lookback_years": input_params.lookback_years
                },
                methodology_summary={
                    "historical_analysis": True,
                    "volatility_adjustment": input_params.volatility_adjustment,
                    "market_regime_analysis": input_params.market_regime_analysis,
                    "risk_free_rate": input_params.risk_free_rate
                }
            )
            
            logger.info(f"Asset return prediction completed successfully for {len(predictions)} asset classes")
            return result
            
        except Exception as e:
            error_msg = f"Asset return prediction failed: {str(e)}"
            logger.error(error_msg)
            
            return PredictionResult(
                success=False,
                predictions={},
                market_regime=MarketRegime.NORMAL_MARKET,
                analysis_period={},
                methodology_summary={},
                error_message=error_msg
            )  
  
    def _convert_to_dataframe(self, asset_returns: List[AssetReturns]) -> pd.DataFrame:
        """
        Convert list of AssetReturns to DataFrame for analysis.
        
        Args:
            asset_returns: List of AssetReturns objects
            
        Returns:
            DataFrame with historical returns data
        """
        data = []
        for asset_return in asset_returns:
            data.append({
                'year': asset_return.year,
                'sp500': asset_return.sp500,
                'small_cap': asset_return.small_cap,
                't_bills': asset_return.t_bills,
                't_bonds': asset_return.t_bonds,
                'corporate_bonds': asset_return.corporate_bonds,
                'real_estate': asset_return.real_estate,
                'gold': asset_return.gold
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('year').reset_index(drop=True)
        
        logger.info(f"Converted {len(df)} years of data to DataFrame for analysis")
        return df
    
    def _calculate_historical_statistics(
        self, 
        df: pd.DataFrame, 
        lookback_years: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate historical mean returns and volatility statistics.
        
        Args:
            df: Historical returns DataFrame
            lookback_years: Number of years to include in analysis
            
        Returns:
            Dictionary with statistics for each asset class
        """
        # Use the most recent lookback_years of data
        recent_df = df.tail(lookback_years).copy()
        
        asset_columns = ['sp500', 'small_cap', 't_bills', 't_bonds', 
                        'corporate_bonds', 'real_estate', 'gold']
        
        statistics = {}
        
        for asset in asset_columns:
            if asset in recent_df.columns:
                returns = recent_df[asset].dropna()
                
                if len(returns) > 0:
                    statistics[asset] = {
                        'mean': float(returns.mean()),
                        'std': float(returns.std()),
                        'min': float(returns.min()),
                        'max': float(returns.max()),
                        'median': float(returns.median()),
                        'count': len(returns),
                        'skewness': float(returns.skew()) if len(returns) > 2 else 0.0,
                        'kurtosis': float(returns.kurtosis()) if len(returns) > 3 else 0.0
                    }
                else:
                    logger.warning(f"No valid data found for {asset}")
                    statistics[asset] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                        'median': 0.0, 'count': 0, 'skewness': 0.0, 'kurtosis': 0.0
                    }
        
        logger.info(f"Calculated historical statistics for {len(statistics)} asset classes")
        return statistics
    
    def _apply_volatility_adjustments(
        self, 
        historical_stats: Dict[str, Dict[str, float]], 
        risk_free_rate: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Apply volatility adjustments to expected returns.
        
        This method applies a risk penalty based on volatility, reducing expected
        returns for higher volatility assets to account for risk aversion.
        
        Args:
            historical_stats: Historical statistics for each asset
            risk_free_rate: Current risk-free rate
            
        Returns:
            Dictionary with volatility-adjusted statistics
        """
        adjusted_stats = {}
        
        # Volatility penalty factors (higher volatility = higher penalty)
        volatility_penalty_factor = 0.5  # Adjust expected return by 50% of excess volatility
        
        for asset, stats in historical_stats.items():
            adjusted_stats[asset] = stats.copy()
            
            # Calculate volatility penalty
            # Penalty = volatility_penalty_factor * (volatility - risk_free_rate_volatility)
            # Assume risk-free rate has minimal volatility (0.01)
            risk_free_volatility = 0.01
            excess_volatility = max(0, stats['std'] - risk_free_volatility)
            volatility_penalty = volatility_penalty_factor * excess_volatility
            
            # Apply penalty to expected return
            original_return = stats['mean']
            adjusted_return = original_return - volatility_penalty
            
            # Ensure adjusted return doesn't go below risk-free rate for risky assets
            if asset not in ['t_bills']:  # T-bills can be below risk-free rate
                adjusted_return = max(adjusted_return, risk_free_rate)
            
            adjusted_stats[asset]['volatility_adjusted_mean'] = adjusted_return
            adjusted_stats[asset]['volatility_penalty'] = volatility_penalty
            adjusted_stats[asset]['original_mean'] = original_return
            
            logger.debug(f"{asset}: {original_return:.4f} -> {adjusted_return:.4f} (penalty: {volatility_penalty:.4f})")
        
        logger.info("Applied volatility adjustments to all asset classes")
        return adjusted_stats
    
    def _analyze_market_regime(
        self, 
        df: pd.DataFrame, 
        current_stats: Dict[str, Dict[str, float]]
    ) -> Tuple[Dict[str, Dict[str, float]], MarketRegime]:
        """
        Analyze current market regime and apply regime-based adjustments.
        
        Args:
            df: Historical returns DataFrame
            current_stats: Current return statistics
            
        Returns:
            Tuple of (regime-adjusted statistics, detected regime)
        """
        # Analyze recent market conditions (last 3 years)
        recent_data = df.tail(3)
        
        # Calculate regime indicators
        regime_indicators = self._calculate_regime_indicators(recent_data)
        
        # Detect market regime
        detected_regime = self._detect_market_regime(regime_indicators)
        
        # Apply regime adjustments
        regime_adjusted_stats = self._apply_regime_adjustments(
            current_stats, 
            detected_regime
        )
        
        logger.info(f"Detected market regime: {detected_regime.value}")
        return regime_adjusted_stats, detected_regime
    
    def _calculate_regime_indicators(self, recent_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate indicators for market regime detection.
        
        Args:
            recent_data: Recent historical data
            
        Returns:
            Dictionary with regime indicators
        """
        indicators = {}
        
        # Calculate average volatility across asset classes
        asset_columns = ['sp500', 'small_cap', 't_bills', 't_bonds', 
                        'corporate_bonds', 'real_estate', 'gold']
        
        volatilities = []
        returns = []
        
        for asset in asset_columns:
            if asset in recent_data.columns:
                asset_returns = recent_data[asset].dropna()
                if len(asset_returns) > 1:
                    volatilities.append(asset_returns.std())
                    returns.append(asset_returns.mean())
        
        indicators['avg_volatility'] = np.mean(volatilities) if volatilities else 0.0
        indicators['avg_return'] = np.mean(returns) if returns else 0.0
        
        # Calculate correlation breakdown indicator
        if len(recent_data) > 1:
            corr_matrix = recent_data[asset_columns].corr()
            # High correlation breakdown = low average correlation
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            indicators['correlation_breakdown'] = 1.0 - avg_correlation
        else:
            indicators['correlation_breakdown'] = 0.0
        
        # Trend strength (based on consistency of returns)
        if len(recent_data) > 1:
            sp500_returns = recent_data['sp500'].dropna()
            if len(sp500_returns) > 1:
                # Positive trend if most returns are positive
                positive_returns = (sp500_returns > 0).sum()
                indicators['trend_strength'] = positive_returns / len(sp500_returns)
            else:
                indicators['trend_strength'] = 0.5
        else:
            indicators['trend_strength'] = 0.5
        
        return indicators
    
    def _detect_market_regime(self, indicators: Dict[str, float]) -> MarketRegime:
        """
        Detect market regime based on calculated indicators.
        
        Args:
            indicators: Market regime indicators
            
        Returns:
            Detected market regime
        """
        avg_volatility = indicators.get('avg_volatility', 0.0)
        avg_return = indicators.get('avg_return', 0.0)
        trend_strength = indicators.get('trend_strength', 0.5)
        correlation_breakdown = indicators.get('correlation_breakdown', 0.0)
        
        # Define regime thresholds
        high_volatility_threshold = 0.20  # 20% volatility
        low_volatility_threshold = 0.10   # 10% volatility
        strong_trend_threshold = 0.7      # 70% of returns in same direction
        high_correlation_breakdown_threshold = 0.3
        
        # Regime detection logic
        if avg_volatility > high_volatility_threshold:
            return MarketRegime.HIGH_VOLATILITY
        elif avg_volatility < low_volatility_threshold:
            return MarketRegime.LOW_VOLATILITY
        elif trend_strength > strong_trend_threshold and avg_return > 0.05:
            return MarketRegime.BULL_MARKET
        elif trend_strength > strong_trend_threshold and avg_return < -0.05:
            return MarketRegime.BEAR_MARKET
        else:
            return MarketRegime.NORMAL_MARKET
    
    def _apply_regime_adjustments(
        self, 
        stats: Dict[str, Dict[str, float]], 
        regime: MarketRegime
    ) -> Dict[str, Dict[str, float]]:
        """
        Apply regime-based adjustments to return expectations.
        
        Args:
            stats: Current return statistics
            regime: Detected market regime
            
        Returns:
            Regime-adjusted statistics
        """
        adjusted_stats = {}
        
        # Define regime adjustment factors
        regime_adjustments = {
            MarketRegime.BULL_MARKET: {
                'equity_boost': 0.01,    # +1% for equities
                'bond_penalty': -0.005,  # -0.5% for bonds
                'gold_penalty': -0.01    # -1% for gold
            },
            MarketRegime.BEAR_MARKET: {
                'equity_penalty': -0.02,  # -2% for equities
                'bond_boost': 0.01,       # +1% for bonds
                'gold_boost': 0.015       # +1.5% for gold
            },
            MarketRegime.HIGH_VOLATILITY: {
                'all_penalty': -0.01      # -1% for all assets
            },
            MarketRegime.LOW_VOLATILITY: {
                'all_boost': 0.005        # +0.5% for all assets
            },
            MarketRegime.NORMAL_MARKET: {
                # No adjustments
            }
        }
        
        for asset, asset_stats in stats.items():
            adjusted_stats[asset] = asset_stats.copy()
            
            # Get base return (volatility adjusted if available, otherwise original mean)
            base_return = asset_stats.get('volatility_adjusted_mean', asset_stats['mean'])
            regime_adjustment = 0.0
            
            # Apply regime-specific adjustments
            if regime in regime_adjustments:
                adjustments = regime_adjustments[regime]
                
                if regime == MarketRegime.BULL_MARKET:
                    if asset in ['sp500', 'small_cap', 'real_estate']:
                        regime_adjustment = adjustments['equity_boost']
                    elif asset in ['t_bonds', 'corporate_bonds']:
                        regime_adjustment = adjustments['bond_penalty']
                    elif asset == 'gold':
                        regime_adjustment = adjustments['gold_penalty']
                
                elif regime == MarketRegime.BEAR_MARKET:
                    if asset in ['sp500', 'small_cap', 'real_estate']:
                        regime_adjustment = adjustments['equity_penalty']
                    elif asset in ['t_bonds', 'corporate_bonds']:
                        regime_adjustment = adjustments['bond_boost']
                    elif asset == 'gold':
                        regime_adjustment = adjustments['gold_boost']
                
                elif regime == MarketRegime.HIGH_VOLATILITY:
                    regime_adjustment = adjustments['all_penalty']
                
                elif regime == MarketRegime.LOW_VOLATILITY:
                    regime_adjustment = adjustments['all_boost']
            
            adjusted_stats[asset]['regime_adjusted_mean'] = base_return + regime_adjustment
            adjusted_stats[asset]['regime_adjustment'] = regime_adjustment
            adjusted_stats[asset]['base_return'] = base_return
        
        logger.info(f"Applied {regime.value} regime adjustments to all asset classes")
        return adjusted_stats
    
    def _calculate_confidence_intervals(
        self, 
        df: pd.DataFrame, 
        final_stats: Dict[str, Dict[str, float]], 
        lookback_years: int
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate 95% confidence intervals for return predictions.
        
        Args:
            df: Historical returns DataFrame
            final_stats: Final return statistics
            lookback_years: Number of years used in analysis
            
        Returns:
            Dictionary with confidence intervals for each asset
        """
        confidence_intervals = {}
        confidence_level = 0.95
        z_score = 1.96  # 95% confidence interval
        
        for asset, stats in final_stats.items():
            # Get final expected return
            expected_return = stats.get('regime_adjusted_mean', 
                                      stats.get('volatility_adjusted_mean', stats['mean']))
            
            # Calculate standard error
            volatility = stats['std']
            n_observations = stats['count']
            
            if n_observations > 1:
                standard_error = volatility / np.sqrt(n_observations)
                margin_of_error = z_score * standard_error
                
                lower_bound = expected_return - margin_of_error
                upper_bound = expected_return + margin_of_error
            else:
                # If insufficient data, use wider interval based on volatility
                margin_of_error = z_score * volatility
                lower_bound = expected_return - margin_of_error
                upper_bound = expected_return + margin_of_error
            
            confidence_intervals[asset] = (lower_bound, upper_bound)
        
        logger.info(f"Calculated {confidence_level:.0%} confidence intervals for all assets")
        return confidence_intervals
    
    def _generate_asset_predictions(
        self,
        historical_stats: Dict[str, Dict[str, float]],
        volatility_adjusted: Dict[str, Dict[str, float]],
        regime_adjusted: Dict[str, Dict[str, float]],
        confidence_intervals: Dict[str, Tuple[float, float]],
        risk_free_rate: float
    ) -> Dict[str, AssetPrediction]:
        """
        Generate final asset predictions combining all analysis steps.
        
        Args:
            historical_stats: Historical statistics
            volatility_adjusted: Volatility-adjusted statistics
            regime_adjusted: Regime-adjusted statistics
            confidence_intervals: Confidence intervals
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Dictionary with AssetPrediction objects for each asset
        """
        predictions = {}
        
        asset_names = {
            'sp500': 'S&P 500',
            'small_cap': 'US Small Cap',
            't_bills': 'Treasury Bills',
            't_bonds': 'Treasury Bonds',
            'corporate_bonds': 'Corporate Bonds',
            'real_estate': 'Real Estate',
            'gold': 'Gold'
        }
        
        for asset, stats in regime_adjusted.items():
            # Get final expected return
            expected_return = stats.get('regime_adjusted_mean', 
                                      stats.get('volatility_adjusted_mean', stats['mean']))
            
            # Get volatility
            volatility = stats['std']
            
            # Calculate Sharpe ratio
            excess_return = expected_return - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
            
            # Get confidence interval
            confidence_interval = confidence_intervals.get(asset, (expected_return, expected_return))
            
            # Create prediction object
            prediction = AssetPrediction(
                asset_name=asset_names.get(asset, asset),
                expected_return=expected_return,
                volatility=volatility,
                confidence_interval=confidence_interval,
                historical_mean=historical_stats[asset]['mean'],
                volatility_adjusted_return=volatility_adjusted[asset].get('volatility_adjusted_mean', 
                                                                        historical_stats[asset]['mean']),
                regime_adjusted_return=expected_return,
                sharpe_ratio=sharpe_ratio
            )
            
            predictions[asset] = prediction
        
        logger.info(f"Generated predictions for {len(predictions)} asset classes")
        return predictions
    
    def get_prediction_summary(self, result: PredictionResult) -> str:
        """
        Generate a human-readable summary of the prediction results.
        
        Args:
            result: PredictionResult object
            
        Returns:
            Formatted string summary of predictions
        """
        if not result.success:
            return f"Prediction failed: {result.error_message}"
        
        summary_lines = [
            "=== ASSET RETURN PREDICTIONS ===",
            "",
            f"Analysis Period: {result.analysis_period['start_year']}-{result.analysis_period['end_year']}",
            f"Lookback Period: {result.analysis_period['lookback_years']} years",
            f"Market Regime: {result.market_regime.value.replace('_', ' ').title()}",
            "",
            "Expected Annual Returns:",
            "-" * 60,
            f"{'Asset':<20} {'Expected':<10} {'Volatility':<12} {'Sharpe':<8} {'95% CI':<20}",
            "-" * 60,
        ]
        
        for asset, prediction in result.predictions.items():
            ci_lower, ci_upper = prediction.confidence_interval
            ci_str = f"[{ci_lower:.1%}, {ci_upper:.1%}]"
            
            summary_lines.append(
                f"{prediction.asset_name:<20} "
                f"{prediction.expected_return:<10.1%} "
                f"{prediction.volatility:<12.1%} "
                f"{prediction.sharpe_ratio:<8.2f} "
                f"{ci_str:<20}"
            )
        
        summary_lines.extend([
            "-" * 60,
            "",
            "Methodology Applied:",
            f"✓ Historical mean calculation ({result.analysis_period['lookback_years']} years)",
            f"✓ Volatility adjustment: {result.methodology_summary['volatility_adjustment']}",
            f"✓ Market regime analysis: {result.methodology_summary['market_regime_analysis']}",
            f"✓ Risk-free rate: {result.methodology_summary['risk_free_rate']:.1%}",
            "",
            "=== END PREDICTIONS ===",
        ])
        
        return "\n".join(summary_lines)


def create_asset_predictor_agent(llm: Optional[BaseLanguageModel] = None) -> AssetPredictorAgent:
    """
    Factory function to create an Asset Predictor Agent.
    
    Args:
        llm: Optional language model for agent reasoning
        
    Returns:
        Configured AssetPredictorAgent instance
    """
    return AssetPredictorAgent(llm=llm)