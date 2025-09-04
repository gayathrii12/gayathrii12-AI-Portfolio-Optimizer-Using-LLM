"""
Risk Metrics Calculation Module for the Financial Returns Optimizer system.

This module provides functions to calculate various risk metrics including Alpha, Beta,
portfolio volatility, Sharpe ratio, and maximum drawdown. All calculations follow
standard financial formulas and are designed to work with the system's data models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

from models.data_models import AssetReturns, PortfolioAllocation, RiskMetrics, ErrorResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskMetricsCalculator:
    """
    Calculator class for portfolio risk metrics.
    
    This class provides methods to calculate various risk metrics used in portfolio
    analysis, including Alpha, Beta, volatility, Sharpe ratio, and maximum drawdown.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the risk metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"RiskMetricsCalculator initialized with risk-free rate: {risk_free_rate:.2%}")
    
    def calculate_portfolio_returns(
        self, 
        historical_data: List[AssetReturns], 
        allocation: PortfolioAllocation
    ) -> pd.Series:
        """
        Calculate portfolio returns based on allocation and historical data.
        
        Args:
            historical_data: List of historical asset returns
            allocation: Portfolio allocation percentages
            
        Returns:
            Series of portfolio returns by year
        """
        logger.info("Calculating portfolio returns from historical data and allocation")
        
        # Convert to DataFrame
        df = self._convert_to_dataframe(historical_data)
        
        # Calculate weighted portfolio returns
        portfolio_returns = (
            df['sp500'] * (allocation.sp500 / 100) +
            df['small_cap'] * (allocation.small_cap / 100) +
            (df['t_bonds'] + df['corporate_bonds']) / 2 * (allocation.bonds / 100) +
            df['gold'] * (allocation.gold / 100) +
            df['real_estate'] * (allocation.real_estate / 100)
        )
        
        # Convert percentages to decimals
        portfolio_returns = portfolio_returns / 100
        
        logger.info(f"Calculated portfolio returns for {len(portfolio_returns)} years")
        return portfolio_returns
    
    def calculate_alpha(
        self, 
        portfolio_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate Alpha (excess return relative to benchmark).
        
        Alpha = Portfolio Return - (Risk-free Rate + Beta * (Benchmark Return - Risk-free Rate))
        
        Args:
            portfolio_returns: Series of portfolio returns (decimal format)
            benchmark_returns: Series of benchmark returns (decimal format)
            
        Returns:
            Alpha as a percentage
        """
        logger.info("Calculating Alpha relative to benchmark")
        
        try:
            # Ensure series are aligned
            aligned_data = pd.DataFrame({
                'portfolio': portfolio_returns,
                'benchmark': benchmark_returns
            }).dropna()
            
            if len(aligned_data) < 2:
                logger.warning("Insufficient data for Alpha calculation")
                return 0.0
            
            portfolio_ret = aligned_data['portfolio']
            benchmark_ret = aligned_data['benchmark']
            
            # Calculate Beta first (needed for Alpha calculation)
            beta = self.calculate_beta(portfolio_ret, benchmark_ret)
            
            # Calculate average returns
            avg_portfolio_return = portfolio_ret.mean()
            avg_benchmark_return = benchmark_ret.mean()
            
            # Alpha = Portfolio Return - (Risk-free Rate + Beta * (Benchmark Return - Risk-free Rate))
            alpha = avg_portfolio_return - (self.risk_free_rate + beta * (avg_benchmark_return - self.risk_free_rate))
            
            # Convert to percentage
            alpha_percentage = alpha * 100
            
            logger.info(f"Alpha calculated: {alpha_percentage:.4f}%")
            return round(alpha_percentage, 4)
            
        except Exception as e:
            logger.error(f"Alpha calculation failed: {str(e)}")
            return 0.0
    
    def calculate_beta(
        self, 
        portfolio_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate Beta (volatility correlation with benchmark).
        
        Beta = Covariance(Portfolio, Benchmark) / Variance(Benchmark)
        
        Args:
            portfolio_returns: Series of portfolio returns (decimal format)
            benchmark_returns: Series of benchmark returns (decimal format)
            
        Returns:
            Beta coefficient
        """
        logger.info("Calculating Beta relative to benchmark")
        
        try:
            # Ensure series are aligned
            aligned_data = pd.DataFrame({
                'portfolio': portfolio_returns,
                'benchmark': benchmark_returns
            }).dropna()
            
            if len(aligned_data) < 2:
                logger.warning("Insufficient data for Beta calculation")
                return 1.0  # Default to market beta
            
            portfolio_ret = aligned_data['portfolio']
            benchmark_ret = aligned_data['benchmark']
            
            # Calculate covariance and variance
            covariance = np.cov(portfolio_ret, benchmark_ret)[0, 1]
            benchmark_variance = np.var(benchmark_ret, ddof=1)
            
            if benchmark_variance == 0:
                logger.warning("Benchmark variance is zero, returning Beta = 1.0")
                return 1.0
            
            beta = covariance / benchmark_variance
            
            logger.info(f"Beta calculated: {beta:.4f}")
            return round(beta, 4)
            
        except Exception as e:
            logger.error(f"Beta calculation failed: {str(e)}")
            return 1.0
    
    def calculate_volatility(self, returns: pd.Series) -> float:
        """
        Calculate portfolio volatility (standard deviation of returns).
        
        Args:
            returns: Series of returns (decimal format)
            
        Returns:
            Volatility as a percentage
        """
        logger.info("Calculating portfolio volatility")
        
        try:
            if len(returns) < 2:
                logger.warning("Insufficient data for volatility calculation")
                return 0.0
            
            # Calculate standard deviation
            volatility = returns.std(ddof=1)
            
            # Convert to percentage
            volatility_percentage = volatility * 100
            
            logger.info(f"Volatility calculated: {volatility_percentage:.4f}%")
            return round(volatility_percentage, 4)
            
        except Exception as e:
            logger.error(f"Volatility calculation failed: {str(e)}")
            return 0.0
    
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return).
        
        Sharpe Ratio = (Portfolio Return - Risk-free Rate) / Portfolio Volatility
        
        Args:
            returns: Series of returns (decimal format)
            
        Returns:
            Sharpe ratio
        """
        logger.info("Calculating Sharpe ratio")
        
        try:
            if len(returns) < 2:
                logger.warning("Insufficient data for Sharpe ratio calculation")
                return 0.0
            
            # Calculate average return and volatility
            avg_return = returns.mean()
            volatility = returns.std(ddof=1)
            
            if volatility == 0:
                logger.warning("Volatility is zero, cannot calculate Sharpe ratio")
                return 0.0
            
            # Sharpe ratio = (Return - Risk-free Rate) / Volatility
            sharpe_ratio = (avg_return - self.risk_free_rate) / volatility
            
            logger.info(f"Sharpe ratio calculated: {sharpe_ratio:.4f}")
            return round(sharpe_ratio, 4)
            
        except Exception as e:
            logger.error(f"Sharpe ratio calculation failed: {str(e)}")
            return 0.0
    
    def calculate_maximum_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown (largest peak-to-trough decline).
        
        Args:
            returns: Series of returns (decimal format)
            
        Returns:
            Maximum drawdown as a negative percentage
        """
        logger.info("Calculating maximum drawdown")
        
        try:
            if len(returns) < 2:
                logger.warning("Insufficient data for maximum drawdown calculation")
                return 0.0
            
            # Calculate cumulative returns (portfolio value over time)
            cumulative_returns = (1 + returns).cumprod()
            
            # Calculate running maximum (peak values)
            running_max = cumulative_returns.expanding().max()
            
            # Calculate drawdown at each point
            drawdown = (cumulative_returns - running_max) / running_max
            
            # Find maximum drawdown (most negative value)
            max_drawdown = drawdown.min()
            
            # Convert to percentage
            max_drawdown_percentage = max_drawdown * 100
            
            logger.info(f"Maximum drawdown calculated: {max_drawdown_percentage:.4f}%")
            return round(max_drawdown_percentage, 4)
            
        except Exception as e:
            logger.error(f"Maximum drawdown calculation failed: {str(e)}")
            return 0.0
    
    def calculate_all_metrics(
        self, 
        historical_data: List[AssetReturns], 
        allocation: PortfolioAllocation,
        benchmark_asset: str = 'sp500'
    ) -> RiskMetrics:
        """
        Calculate all risk metrics for a portfolio.
        
        Args:
            historical_data: List of historical asset returns
            allocation: Portfolio allocation percentages
            benchmark_asset: Asset to use as benchmark (default: 'sp500')
            
        Returns:
            RiskMetrics object with all calculated metrics
        """
        logger.info("Calculating all risk metrics for portfolio")
        
        try:
            # Convert to DataFrame
            df = self._convert_to_dataframe(historical_data)
            
            # Calculate portfolio returns
            portfolio_returns = self.calculate_portfolio_returns(historical_data, allocation)
            
            # Get benchmark returns
            benchmark_returns = df[benchmark_asset] / 100  # Convert to decimal
            
            # Calculate all metrics
            alpha = self.calculate_alpha(portfolio_returns, benchmark_returns)
            beta = self.calculate_beta(portfolio_returns, benchmark_returns)
            volatility = self.calculate_volatility(portfolio_returns)
            sharpe_ratio = self.calculate_sharpe_ratio(portfolio_returns)
            max_drawdown = self.calculate_maximum_drawdown(portfolio_returns)
            
            # Create RiskMetrics object
            risk_metrics = RiskMetrics(
                alpha=alpha,
                beta=beta,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown
            )
            
            logger.info("All risk metrics calculated successfully")
            return risk_metrics
            
        except Exception as e:
            error_msg = f"Risk metrics calculation failed: {str(e)}"
            logger.error(error_msg)
            
            # Return default metrics on error
            return RiskMetrics(
                alpha=0.0,
                beta=1.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0
            )
    
    def validate_metrics(self, metrics: RiskMetrics) -> Tuple[bool, List[str]]:
        """
        Validate calculated risk metrics for reasonableness.
        
        Args:
            metrics: Calculated risk metrics
            
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        is_valid = True
        
        # Check Alpha
        if abs(metrics.alpha) > 20:  # Alpha beyond ±20% is unusual
            warnings.append(f"Alpha ({metrics.alpha:.2f}%) is unusually high")
        
        # Check Beta
        if metrics.beta < 0:
            warnings.append(f"Beta ({metrics.beta:.2f}) is negative, indicating inverse correlation")
        elif metrics.beta > 3:
            warnings.append(f"Beta ({metrics.beta:.2f}) is very high, indicating high volatility")
        
        # Check Volatility
        if metrics.volatility > 50:
            warnings.append(f"Volatility ({metrics.volatility:.2f}%) is very high")
        elif metrics.volatility < 1:
            warnings.append(f"Volatility ({metrics.volatility:.2f}%) is unusually low")
        
        # Check Sharpe Ratio
        if metrics.sharpe_ratio > 3:
            warnings.append(f"Sharpe ratio ({metrics.sharpe_ratio:.2f}) is unusually high")
        elif metrics.sharpe_ratio < -2:
            warnings.append(f"Sharpe ratio ({metrics.sharpe_ratio:.2f}) is very poor")
        
        # Check Maximum Drawdown
        if metrics.max_drawdown < -80:
            warnings.append(f"Maximum drawdown ({metrics.max_drawdown:.2f}%) is extremely high")
        elif metrics.max_drawdown > 0:
            warnings.append(f"Maximum drawdown ({metrics.max_drawdown:.2f}%) should be negative or zero")
            is_valid = False
        
        if warnings:
            logger.warning(f"Risk metrics validation found {len(warnings)} warnings")
            for warning in warnings:
                logger.warning(warning)
        else:
            logger.info("Risk metrics validation passed without warnings")
        
        return is_valid, warnings
    
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
        
        return df


def calculate_portfolio_risk_metrics(
    historical_data: List[AssetReturns],
    allocation: PortfolioAllocation,
    risk_free_rate: float = 0.02,
    benchmark_asset: str = 'sp500'
) -> RiskMetrics:
    """
    Convenience function to calculate all risk metrics for a portfolio.
    
    Args:
        historical_data: List of historical asset returns
        allocation: Portfolio allocation percentages
        risk_free_rate: Annual risk-free rate (default 2%)
        benchmark_asset: Asset to use as benchmark (default: 'sp500')
        
    Returns:
        RiskMetrics object with all calculated metrics
    """
    calculator = RiskMetricsCalculator(risk_free_rate=risk_free_rate)
    return calculator.calculate_all_metrics(historical_data, allocation, benchmark_asset)


def validate_risk_metrics(metrics: RiskMetrics) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate risk metrics.
    
    Args:
        metrics: Calculated risk metrics
        
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    calculator = RiskMetricsCalculator()
    return calculator.validate_metrics(metrics)