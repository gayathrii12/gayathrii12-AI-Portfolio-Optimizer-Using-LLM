"""
Simulation Agent for the Financial Returns Optimizer system.

This agent is responsible for portfolio projection calculations including
lumpsum investment projections, SIP calculations, withdrawal schedule processing,
and CAGR/cumulative return calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from enum import Enum

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from models.data_models import PortfolioAllocation, ProjectionResult, UserInputModel, ErrorResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InvestmentType(Enum):
    """Investment type classifications."""
    LUMPSUM = "lumpsum"
    SIP = "sip"


class SimulationInput(BaseModel):
    """Input model for portfolio simulation."""
    user_input: UserInputModel = Field(description="User investment parameters")
    portfolio_allocation: PortfolioAllocation = Field(description="Portfolio allocation percentages")
    expected_returns: Dict[str, float] = Field(description="Expected returns for each asset class")
    volatility_estimates: Optional[Dict[str, float]] = Field(
        default=None,
        description="Volatility estimates for Monte Carlo simulation"
    )
    simulation_runs: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of Monte Carlo simulation runs"
    )
    rebalancing_frequency: str = Field(
        default="annual",
        description="Rebalancing frequency: annual, quarterly, monthly"
    )


class SimulationResult(BaseModel):
    """Result model for portfolio simulation."""
    success: bool = Field(description="Whether simulation was successful")
    projections: List[ProjectionResult] = Field(description="Year-by-year portfolio projections")
    final_value: float = Field(description="Final portfolio value")
    total_invested: float = Field(description="Total amount invested")
    cagr: float = Field(description="Compound Annual Growth Rate")
    cumulative_return: float = Field(description="Total cumulative return percentage")
    simulation_statistics: Dict[str, float] = Field(description="Monte Carlo simulation statistics")
    withdrawal_impact: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Impact of withdrawals on portfolio"
    )
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class CalculateLumpsumProjectionTool(BaseTool):
    """Tool for calculating lumpsum investment projections."""
    
    name: str = "calculate_lumpsum_projection"
    description: str = "Calculate portfolio projections for lumpsum investment"
    
    def _run(
        self,
        investment_amount: float,
        tenure_years: int,
        expected_returns: str,
        allocation: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Calculate lumpsum investment projections."""
        try:
            # Sample calculation for demonstration
            annual_return = 0.08  # 8% average return
            projections = []
            
            current_value = investment_amount
            
            for year in range(1, tenure_years + 1):
                # Apply annual return
                annual_growth = current_value * annual_return
                current_value += annual_growth
                
                # Calculate cumulative return
                cumulative_return = ((current_value - investment_amount) / investment_amount) * 100
                
                projection = {
                    "year": year,
                    "portfolio_value": round(current_value, 2),
                    "annual_return": annual_return * 100,
                    "cumulative_return": round(cumulative_return, 2)
                }
                projections.append(projection)
            
            result = {
                "projections": projections,
                "final_value": round(current_value, 2),
                "total_invested": investment_amount,
                "calculation_method": "compound_growth"
            }
            
            logger.info(f"Calculated lumpsum projections for {tenure_years} years")
            return str(result)
            
        except Exception as e:
            error_msg = f"Lumpsum projection calculation failed: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"


class CalculateSIPProjectionTool(BaseTool):
    """Tool for calculating SIP (Systematic Investment Plan) projections."""
    
    name: str = "calculate_sip_projection"
    description: str = "Calculate portfolio projections for SIP investments"
    
    def _run(
        self,
        monthly_investment: float,
        tenure_years: int,
        expected_returns: str,
        allocation: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Calculate SIP investment projections."""
        try:
            annual_return = 0.08  # 8% average return
            monthly_return = annual_return / 12
            projections = []
            
            current_value = 0
            total_invested = 0
            
            for year in range(1, tenure_years + 1):
                # Add monthly investments for the year
                for month in range(12):
                    total_invested += monthly_investment
                    current_value += monthly_investment
                    # Apply monthly return to entire portfolio
                    current_value *= (1 + monthly_return)
                
                # Calculate cumulative return
                if total_invested > 0:
                    cumulative_return = ((current_value - total_invested) / total_invested) * 100
                else:
                    cumulative_return = 0
                
                projection = {
                    "year": year,
                    "portfolio_value": round(current_value, 2),
                    "annual_return": annual_return * 100,
                    "cumulative_return": round(cumulative_return, 2),
                    "total_invested_so_far": round(total_invested, 2)
                }
                projections.append(projection)
            
            result = {
                "projections": projections,
                "final_value": round(current_value, 2),
                "total_invested": round(total_invested, 2),
                "monthly_investment": monthly_investment,
                "calculation_method": "sip_compound_growth"
            }
            
            logger.info(f"Calculated SIP projections for {tenure_years} years")
            return str(result)
            
        except Exception as e:
            error_msg = f"SIP projection calculation failed: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"


class ProcessWithdrawalScheduleTool(BaseTool):
    """Tool for processing withdrawal schedules and their impact."""
    
    name: str = "process_withdrawal_schedule"
    description: str = "Process withdrawal schedule and calculate impact on portfolio"
    
    def _run(
        self,
        base_projections: str,
        withdrawal_preferences: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Process withdrawal schedule impact."""
        try:
            # Sample withdrawal processing
            withdrawal_impact = {
                "total_withdrawals": 0,
                "withdrawal_years": [],
                "impact_on_final_value": 0,
                "adjusted_projections": [],
                "withdrawal_strategy": "none"
            }
            
            # If no withdrawals specified, return base projections
            if not withdrawal_preferences or withdrawal_preferences == "None":
                withdrawal_impact["withdrawal_strategy"] = "none"
                logger.info("No withdrawal schedule specified")
                return str(withdrawal_impact)
            
            # Process withdrawal schedule (simplified example)
            withdrawal_impact.update({
                "total_withdrawals": 50000,
                "withdrawal_years": [10, 15, 20],
                "impact_on_final_value": -15.5,
                "withdrawal_strategy": "periodic"
            })
            
            logger.info("Processed withdrawal schedule impact")
            return str(withdrawal_impact)
            
        except Exception as e:
            error_msg = f"Withdrawal schedule processing failed: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"


class CalculateCAGRTool(BaseTool):
    """Tool for calculating CAGR and cumulative returns."""
    
    name: str = "calculate_cagr"
    description: str = "Calculate CAGR and cumulative return metrics"
    
    def _run(
        self,
        initial_value: float,
        final_value: float,
        tenure_years: int,
        total_invested: float,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Calculate CAGR and return metrics."""
        try:
            # Calculate CAGR
            if initial_value > 0 and tenure_years > 0:
                cagr = ((final_value / initial_value) ** (1 / tenure_years)) - 1
            else:
                cagr = 0
            
            # Calculate cumulative return
            if total_invested > 0:
                cumulative_return = ((final_value - total_invested) / total_invested) * 100
            else:
                cumulative_return = 0
            
            # Calculate absolute return
            absolute_return = final_value - total_invested
            
            metrics = {
                "cagr": round(cagr * 100, 2),  # Convert to percentage
                "cumulative_return": round(cumulative_return, 2),
                "absolute_return": round(absolute_return, 2),
                "final_value": round(final_value, 2),
                "total_invested": round(total_invested, 2),
                "tenure_years": tenure_years,
                "annualized_return": round(cagr * 100, 2)
            }
            
            logger.info(f"Calculated CAGR: {metrics['cagr']}%, Cumulative Return: {metrics['cumulative_return']}%")
            return str(metrics)
            
        except Exception as e:
            error_msg = f"CAGR calculation failed: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"


class SimulationAgent:
    """
    LangChain-based agent for portfolio projection calculations.
    
    This agent handles lumpsum and SIP investment projections, withdrawal
    schedule processing, and comprehensive return metric calculations.
    """
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        """
        Initialize the Simulation Agent.
        
        Args:
            llm: Language model for agent reasoning (optional for tool-only operations)
        """
        self.llm = llm
        self.tools = [
            CalculateLumpsumProjectionTool(),
            CalculateSIPProjectionTool(),
            ProcessWithdrawalScheduleTool(),
            CalculateCAGRTool()
        ]
        
        # Create the agent prompt
        self.prompt = PromptTemplate.from_template("""
        You are a portfolio simulation specialist responsible for calculating
        investment projections, handling different investment types, and
        processing withdrawal schedules.
        
        Available tools:
        {tools}
        
        Tool names: {tool_names}
        
        Follow this systematic approach:
        1. Determine investment type (lumpsum vs SIP)
        2. Calculate base projections using appropriate method
        3. Process withdrawal schedule if specified
        4. Calculate CAGR and cumulative return metrics
        5. Generate comprehensive simulation results
        
        Always ensure calculations are accurate and handle edge cases properly.
        
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
    
    def simulate_portfolio(self, input_params: SimulationInput) -> SimulationResult:
        """
        Generate portfolio projections based on investment parameters.
        
        Args:
            input_params: Parameters for portfolio simulation
            
        Returns:
            SimulationResult with projections and metrics
        """
        logger.info(f"Starting portfolio simulation for {input_params.user_input.investment_type} investment")
        
        try:
            # Step 1: Calculate base projections based on investment type
            logger.info("Step 1: Calculating base projections")
            base_projections = self._calculate_base_projections(input_params)
            
            # Step 2: Process withdrawal schedule if specified
            logger.info("Step 2: Processing withdrawal schedule")
            withdrawal_impact = self._process_withdrawal_schedule(
                base_projections,
                input_params.user_input.withdrawal_preferences
            )
            
            # Step 3: Apply Monte Carlo simulation for risk analysis
            logger.info("Step 3: Running Monte Carlo simulation")
            simulation_stats = self._run_monte_carlo_simulation(input_params)
            
            # Step 4: Calculate final metrics
            logger.info("Step 4: Calculating final metrics")
            final_metrics = self._calculate_final_metrics(
                base_projections,
                withdrawal_impact,
                input_params.user_input
            )
            
            # Create final result
            result = SimulationResult(
                success=True,
                projections=base_projections,
                final_value=final_metrics["final_value"],
                total_invested=final_metrics["total_invested"],
                cagr=final_metrics["cagr"],
                cumulative_return=final_metrics["cumulative_return"],
                simulation_statistics=simulation_stats,
                withdrawal_impact=withdrawal_impact
            )
            
            logger.info(f"Portfolio simulation completed successfully. Final value: {result.final_value}")
            return result
            
        except Exception as e:
            error_msg = f"Portfolio simulation failed: {str(e)}"
            logger.error(error_msg)
            
            return SimulationResult(
                success=False,
                projections=[],
                final_value=0.0,
                total_invested=0.0,
                cagr=0.0,
                cumulative_return=0.0,
                simulation_statistics={},
                error_message=error_msg
            )
    
    def _calculate_base_projections(self, input_params: SimulationInput) -> List[ProjectionResult]:
        """
        Calculate base portfolio projections based on investment type.
        
        Args:
            input_params: Simulation input parameters
            
        Returns:
            List of yearly projection results
        """
        user_input = input_params.user_input
        allocation = input_params.portfolio_allocation
        expected_returns = input_params.expected_returns
        
        # Calculate weighted average expected return
        portfolio_return = self._calculate_portfolio_return(allocation, expected_returns)
        
        projections = []
        
        if user_input.investment_type == "lumpsum":
            projections = self._calculate_lumpsum_projections(
                user_input.investment_amount,
                user_input.tenure_years,
                portfolio_return
            )
        elif user_input.investment_type == "sip":
            projections = self._calculate_sip_projections(
                user_input.investment_amount,  # Monthly SIP amount
                user_input.tenure_years,
                portfolio_return
            )
        
        return projections
    
    def _calculate_portfolio_return(
        self, 
        allocation: PortfolioAllocation, 
        expected_returns: Dict[str, float]
    ) -> float:
        """
        Calculate weighted average portfolio return.
        
        Args:
            allocation: Portfolio allocation percentages
            expected_returns: Expected returns for each asset class
            
        Returns:
            Weighted average expected return
        """
        # Map allocation to expected returns
        portfolio_return = 0.0
        
        # Handle combined bonds allocation
        bond_return = (expected_returns.get('t_bonds', 0.05) + 
                      expected_returns.get('corporate_bonds', 0.06)) / 2
        
        portfolio_return += (allocation.sp500 / 100) * expected_returns.get('sp500', 0.10)
        portfolio_return += (allocation.small_cap / 100) * expected_returns.get('small_cap', 0.12)
        portfolio_return += (allocation.bonds / 100) * bond_return
        portfolio_return += (allocation.real_estate / 100) * expected_returns.get('real_estate', 0.09)
        portfolio_return += (allocation.gold / 100) * expected_returns.get('gold', 0.07)
        
        return portfolio_return
    
    def _calculate_lumpsum_projections(
        self, 
        investment_amount: float, 
        tenure_years: int, 
        annual_return: float
    ) -> List[ProjectionResult]:
        """
        Calculate projections for lumpsum investment.
        
        Args:
            investment_amount: Initial lumpsum investment
            tenure_years: Investment tenure in years
            annual_return: Expected annual return rate
            
        Returns:
            List of yearly projections
        """
        projections = []
        current_value = investment_amount
        
        for year in range(1, tenure_years + 1):
            # Apply annual return
            annual_growth = current_value * annual_return
            current_value += annual_growth
            
            # Calculate cumulative return
            cumulative_return = ((current_value - investment_amount) / investment_amount) * 100
            
            projection = ProjectionResult(
                year=year,
                portfolio_value=round(current_value, 2),
                annual_return=round(annual_return * 100, 2),
                cumulative_return=round(cumulative_return, 2)
            )
            projections.append(projection)
        
        logger.info(f"Calculated {len(projections)} lumpsum projections")
        return projections
    
    def _calculate_sip_projections(
        self, 
        monthly_investment: float, 
        tenure_years: int, 
        annual_return: float
    ) -> List[ProjectionResult]:
        """
        Calculate projections for SIP investment.
        
        Args:
            monthly_investment: Monthly SIP amount
            tenure_years: Investment tenure in years
            annual_return: Expected annual return rate
            
        Returns:
            List of yearly projections
        """
        projections = []
        monthly_return = annual_return / 12
        current_value = 0
        total_invested = 0
        
        for year in range(1, tenure_years + 1):
            # Add monthly investments for the year
            for month in range(12):
                total_invested += monthly_investment
                current_value += monthly_investment
                # Apply monthly return to entire portfolio
                current_value *= (1 + monthly_return)
            
            # Calculate cumulative return
            if total_invested > 0:
                cumulative_return = ((current_value - total_invested) / total_invested) * 100
            else:
                cumulative_return = 0
            
            projection = ProjectionResult(
                year=year,
                portfolio_value=round(current_value, 2),
                annual_return=round(annual_return * 100, 2),
                cumulative_return=round(cumulative_return, 2)
            )
            projections.append(projection)
        
        logger.info(f"Calculated {len(projections)} SIP projections")
        return projections
    
    def _process_withdrawal_schedule(
        self, 
        base_projections: List[ProjectionResult], 
        withdrawal_preferences: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Process withdrawal schedule and calculate impact.
        
        Args:
            base_projections: Base portfolio projections
            withdrawal_preferences: User withdrawal preferences
            
        Returns:
            Dictionary with withdrawal impact details
        """
        if not withdrawal_preferences:
            return None
        
        # Simplified withdrawal processing
        withdrawal_impact = {
            "total_withdrawals": 0.0,
            "impact_on_final_value": 0.0,
            "withdrawal_strategy": "none"
        }
        
        # Process different withdrawal types
        if "annual_withdrawal" in withdrawal_preferences:
            annual_amount = withdrawal_preferences["annual_withdrawal"]
            start_year = withdrawal_preferences.get("start_year", 1)
            
            total_withdrawals = annual_amount * max(0, len(base_projections) - start_year + 1)
            
            withdrawal_impact.update({
                "total_withdrawals": float(total_withdrawals),
                "impact_on_final_value": float(-total_withdrawals * 0.8),  # Simplified impact
                "withdrawal_strategy": "annual"
            })
        
        logger.info(f"Processed withdrawal schedule: {withdrawal_impact['withdrawal_strategy']}")
        return withdrawal_impact
    
    def _run_monte_carlo_simulation(self, input_params: SimulationInput) -> Dict[str, float]:
        """
        Run Monte Carlo simulation for risk analysis.
        
        Args:
            input_params: Simulation input parameters
            
        Returns:
            Dictionary with simulation statistics
        """
        # Simplified Monte Carlo simulation
        num_runs = input_params.simulation_runs
        portfolio_return = self._calculate_portfolio_return(
            input_params.portfolio_allocation,
            input_params.expected_returns
        )
        
        # Estimate portfolio volatility (simplified)
        portfolio_volatility = 0.15  # 15% default volatility
        
        # Generate random returns
        np.random.seed(42)  # For reproducible results
        random_returns = np.random.normal(
            portfolio_return, 
            portfolio_volatility, 
            num_runs
        )
        
        # Calculate statistics
        simulation_stats = {
            "mean_return": float(np.mean(random_returns)),
            "median_return": float(np.median(random_returns)),
            "std_deviation": float(np.std(random_returns)),
            "percentile_5": float(np.percentile(random_returns, 5)),
            "percentile_95": float(np.percentile(random_returns, 95)),
            "probability_positive": float(np.mean(random_returns > 0)),
            "simulation_runs": num_runs
        }
        
        logger.info(f"Completed Monte Carlo simulation with {num_runs} runs")
        return simulation_stats
    
    def _calculate_final_metrics(
        self, 
        projections: List[ProjectionResult], 
        withdrawal_impact: Optional[Dict[str, Any]], 
        user_input: UserInputModel
    ) -> Dict[str, float]:
        """
        Calculate final portfolio metrics.
        
        Args:
            projections: Portfolio projections
            withdrawal_impact: Withdrawal impact details
            user_input: User input parameters
            
        Returns:
            Dictionary with final metrics
        """
        if not projections:
            return {
                "final_value": 0.0,
                "total_invested": 0.0,
                "cagr": 0.0,
                "cumulative_return": 0.0
            }
        
        final_projection = projections[-1]
        final_value = final_projection.portfolio_value
        
        # Adjust for withdrawals
        if withdrawal_impact and withdrawal_impact.get("impact_on_final_value"):
            final_value += withdrawal_impact["impact_on_final_value"]
        
        # Calculate total invested
        if user_input.investment_type == "lumpsum":
            total_invested = user_input.investment_amount
            # For lumpsum, CAGR calculation uses initial vs final value
            if user_input.tenure_years > 0 and user_input.investment_amount > 0:
                cagr = ((final_value / user_input.investment_amount) ** (1 / user_input.tenure_years)) - 1
            else:
                cagr = 0
        else:  # SIP
            total_invested = user_input.investment_amount * 12 * user_input.tenure_years
            # For SIP, use total invested vs final value
            if user_input.tenure_years > 0 and total_invested > 0:
                cagr = ((final_value / total_invested) ** (1 / user_input.tenure_years)) - 1
            else:
                cagr = 0
        
        # Calculate cumulative return
        if total_invested > 0:
            cumulative_return = ((final_value - total_invested) / total_invested) * 100
        else:
            cumulative_return = 0
        
        metrics = {
            "final_value": round(final_value, 2),
            "total_invested": round(total_invested, 2),
            "cagr": round(cagr * 100, 2),  # Convert to percentage
            "cumulative_return": round(cumulative_return, 2)
        }
        
        logger.info(f"Final metrics - CAGR: {metrics['cagr']}%, Final Value: {metrics['final_value']}")
        return metrics