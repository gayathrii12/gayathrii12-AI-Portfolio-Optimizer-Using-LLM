"""
Langgraph Workflow for Financial Returns Optimizer

This module implements the main workflow that coordinates all agents
using Langgraph framework for intelligent portfolio management.
"""

from typing import Dict, List, Optional, Any, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import logging
from datetime import datetime

from agents.return_prediction_agent import ReturnPredictionAgent
from agents.portfolio_allocation_agent import PortfolioAllocationAgent
from agents.rebalancing_agent import RebalancingAgent
from agents.model_manager import ModelManager, get_model_manager
from models.asset_return_models import AssetReturnModels

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """
    State schema for the Langgraph workflow
    """
    # Input parameters
    investment_amount: float
    investment_horizon: int
    risk_profile: str
    investment_type: str
    monthly_amount: Optional[float]
    withdrawal_amount: Optional[float]
    
    # Asset classes to analyze
    asset_classes: List[str]
    
    # Agent outputs
    predicted_returns: Optional[Dict[str, float]]
    confidence_scores: Optional[Dict[str, float]]
    prediction_rationale: Optional[str]
    
    portfolio_allocation: Optional[Dict[str, float]]
    expected_portfolio_return: Optional[float]
    allocation_rationale: Optional[str]
    
    rebalancing_schedule: Optional[List[Dict[str, Any]]]
    final_allocation: Optional[Dict[str, float]]
    rebalancing_rationale: Optional[str]
    
    # Workflow control
    agent_status: Optional[str]
    error: Optional[str]
    workflow_complete: bool
    
    # Metadata
    workflow_id: Optional[str]
    created_at: Optional[str]


class FinancialPlanningWorkflow:
    """
    Main workflow class that coordinates all agents using Langgraph
    """
    
    def __init__(self, asset_models: Optional[AssetReturnModels] = None):
        # Initialize or get models
        if asset_models is None:
            logger.info("No asset models provided, initializing from model manager...")
            model_manager = get_model_manager()
            
            # Try to initialize models
            if not model_manager.initialize_models():
                logger.error("Failed to initialize ML models")
                raise RuntimeError("Cannot create workflow without ML models")
            
            self.asset_models = model_manager.get_asset_models()
            if self.asset_models is None:
                raise RuntimeError("Failed to get initialized models from model manager")
        else:
            self.asset_models = asset_models
        
        # Initialize agents with ML models
        self.return_prediction_agent = ReturnPredictionAgent(self.asset_models)
        self.portfolio_allocation_agent = PortfolioAllocationAgent()
        self.rebalancing_agent = RebalancingAgent()
        
        # Initialize memory for checkpointing
        self.memory = MemorySaver()
        
        # Create workflow graph
        self.workflow = self._create_workflow()
        
        logger.info("Financial Planning Workflow initialized with ML models")
    
    def _create_workflow(self) -> StateGraph:
        """
        Create the Langgraph workflow with agent coordination
        """
        try:
            # Create state graph
            workflow = StateGraph(WorkflowState)
            
            # Add nodes for each agent
            workflow.add_node("return_prediction", self._return_prediction_node)
            workflow.add_node("portfolio_allocation", self._portfolio_allocation_node)
            workflow.add_node("rebalancing", self._rebalancing_node)
            workflow.add_node("error_handler", self._error_handler_node)
            workflow.add_node("completion", self._completion_node)
            
            # Define workflow edges
            workflow.set_entry_point("return_prediction")
            
            # Return prediction -> Portfolio allocation or error
            workflow.add_conditional_edges(
                "return_prediction",
                self._check_return_prediction_status,
                {
                    "success": "portfolio_allocation",
                    "error": "error_handler"
                }
            )
            
            # Portfolio allocation -> Rebalancing or error
            workflow.add_conditional_edges(
                "portfolio_allocation",
                self._check_allocation_status,
                {
                    "success": "rebalancing",
                    "error": "error_handler"
                }
            )
            
            # Rebalancing -> Completion or error
            workflow.add_conditional_edges(
                "rebalancing",
                self._check_rebalancing_status,
                {
                    "success": "completion",
                    "error": "error_handler"
                }
            )
            
            # Error handler and completion end the workflow
            workflow.add_edge("error_handler", END)
            workflow.add_edge("completion", END)
            
            # Compile workflow with memory
            compiled_workflow = workflow.compile(checkpointer=self.memory)
            
            logger.info("Langgraph workflow created successfully")
            return compiled_workflow
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise
    
    def _return_prediction_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node for return prediction agent
        """
        try:
            logger.info("Executing return prediction node")
            
            # Add default asset classes if not provided
            if not state.get('asset_classes'):
                state['asset_classes'] = [
                    'sp500', 'small_cap', 't_bills', 't_bonds',
                    'corporate_bonds', 'real_estate', 'gold'
                ]
            
            # Execute return prediction agent
            updated_state = self.return_prediction_agent.predict_returns(state)
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Return prediction node failed: {e}")
            state['error'] = f"Return prediction failed: {str(e)}"
            state['agent_status'] = 'return_prediction_failed'
            return state
    
    def _portfolio_allocation_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node for portfolio allocation agent
        """
        try:
            logger.info("Executing portfolio allocation node")
            
            # Execute portfolio allocation agent
            updated_state = self.portfolio_allocation_agent.allocate_portfolio(state)
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Portfolio allocation node failed: {e}")
            state['error'] = f"Portfolio allocation failed: {str(e)}"
            state['agent_status'] = 'portfolio_allocation_failed'
            return state
    
    def _rebalancing_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node for rebalancing agent
        """
        try:
            logger.info("Executing rebalancing node")
            
            # Execute rebalancing agent
            updated_state = self.rebalancing_agent.create_rebalancing_strategy(state)
            
            # Also calculate rebalancing impact
            if updated_state.get('agent_status') == 'rebalancing_complete':
                updated_state = self.rebalancing_agent.calculate_rebalancing_impact(updated_state)
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Rebalancing node failed: {e}")
            state['error'] = f"Rebalancing failed: {str(e)}"
            state['agent_status'] = 'rebalancing_failed'
            return state
    
    def _error_handler_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node for handling errors and providing fallback responses
        """
        try:
            logger.warning(f"Handling workflow error: {state.get('error', 'Unknown error')}")
            
            # Provide fallback values based on what failed
            agent_status = state.get('agent_status', '')
            
            if 'return_prediction_failed' in agent_status:
                # Provide fallback returns
                state['predicted_returns'] = {
                    'sp500': 0.10, 'small_cap': 0.11, 't_bills': 0.03,
                    't_bonds': 0.05, 'corporate_bonds': 0.06,
                    'real_estate': 0.08, 'gold': 0.07
                }
                state['prediction_rationale'] = "Using historical average returns due to prediction failure"
                
                # Try to continue with portfolio allocation
                try:
                    state = self.portfolio_allocation_agent.allocate_portfolio(state)
                    if state.get('agent_status') == 'portfolio_allocation_complete':
                        state = self.rebalancing_agent.create_rebalancing_strategy(state)
                except Exception as e:
                    logger.error(f"Fallback processing failed: {e}")
            
            elif 'portfolio_allocation_failed' in agent_status:
                # Provide fallback allocation based on risk profile
                risk_profile = state.get('risk_profile', 'moderate')
                fallback_allocations = {
                    'low': {'sp500': 15, 'small_cap': 5, 't_bills': 30, 't_bonds': 25, 
                           'corporate_bonds': 15, 'real_estate': 7, 'gold': 3},
                    'moderate': {'sp500': 30, 'small_cap': 10, 't_bills': 15, 't_bonds': 20,
                                'corporate_bonds': 10, 'real_estate': 10, 'gold': 5},
                    'high': {'sp500': 45, 'small_cap': 20, 't_bills': 5, 't_bonds': 10,
                            'corporate_bonds': 5, 'real_estate': 10, 'gold': 5}
                }
                
                state['portfolio_allocation'] = fallback_allocations.get(risk_profile, fallback_allocations['moderate'])
                state['allocation_rationale'] = f"Using default {risk_profile} risk allocation due to allocation failure"
                
                # Try rebalancing with fallback allocation
                try:
                    state = self.rebalancing_agent.create_rebalancing_strategy(state)
                except Exception as e:
                    logger.error(f"Fallback rebalancing failed: {e}")
            
            state['workflow_complete'] = True
            state['agent_status'] = 'completed_with_errors'
            
            return state
            
        except Exception as e:
            logger.error(f"Error handler failed: {e}")
            state['error'] = f"Critical workflow failure: {str(e)}"
            state['workflow_complete'] = True
            state['agent_status'] = 'critical_failure'
            return state
    
    def _completion_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node for workflow completion and final validation
        """
        try:
            logger.info("Executing completion node")
            
            # Validate that all required outputs are present
            required_outputs = ['predicted_returns', 'portfolio_allocation', 'rebalancing_schedule']
            missing_outputs = [output for output in required_outputs if not state.get(output)]
            
            if missing_outputs:
                logger.warning(f"Missing outputs: {missing_outputs}")
                state['error'] = f"Incomplete workflow: missing {', '.join(missing_outputs)}"
                state['agent_status'] = 'incomplete'
            else:
                state['agent_status'] = 'completed_successfully'
            
            # Mark workflow as complete
            state['workflow_complete'] = True
            
            # Add completion timestamp
            state['completed_at'] = datetime.now().isoformat()
            
            logger.info(f"Workflow completed with status: {state['agent_status']}")
            return state
            
        except Exception as e:
            logger.error(f"Completion node failed: {e}")
            state['error'] = f"Completion failed: {str(e)}"
            state['workflow_complete'] = True
            state['agent_status'] = 'completion_failed'
            return state
    
    def _check_return_prediction_status(self, state: WorkflowState) -> str:
        """
        Check return prediction status for conditional routing
        """
        status = state.get('agent_status', '')
        if 'return_prediction_complete' in status:
            return "success"
        else:
            return "error"
    
    def _check_allocation_status(self, state: WorkflowState) -> str:
        """
        Check portfolio allocation status for conditional routing
        """
        status = state.get('agent_status', '')
        if 'portfolio_allocation_complete' in status:
            return "success"
        else:
            return "error"
    
    def _check_rebalancing_status(self, state: WorkflowState) -> str:
        """
        Check rebalancing status for conditional routing
        """
        status = state.get('agent_status', '')
        if 'rebalancing_complete' in status:
            return "success"
        else:
            return "error"
    
    def execute_workflow(self, input_data: Dict[str, Any], 
                        workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the complete workflow with given input data
        
        Args:
            input_data: Input parameters for the workflow
            workflow_id: Optional workflow ID for checkpointing
            
        Returns:
            Final workflow state with results
        """
        try:
            logger.info(f"Starting workflow execution with ID: {workflow_id}")
            
            # Create initial state
            initial_state = WorkflowState(
                investment_amount=input_data.get('investment_amount', 0),
                investment_horizon=input_data.get('investment_horizon', 10),
                risk_profile=input_data.get('risk_profile', 'moderate'),
                investment_type=input_data.get('investment_type', 'lump_sum'),
                monthly_amount=input_data.get('monthly_amount'),
                withdrawal_amount=input_data.get('withdrawal_amount'),
                asset_classes=input_data.get('asset_classes', []),
                workflow_complete=False,
                workflow_id=workflow_id,
                created_at=datetime.now().isoformat()
            )
            
            # Execute workflow
            config = {"configurable": {"thread_id": workflow_id or "default"}}
            final_state = self.workflow.invoke(initial_state, config)
            
            logger.info(f"Workflow execution completed with status: {final_state.get('agent_status', 'unknown')}")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                'error': f"Workflow execution failed: {str(e)}",
                'workflow_complete': True,
                'agent_status': 'execution_failed'
            }
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get the current status of a workflow execution
        
        Args:
            workflow_id: ID of the workflow to check
            
        Returns:
            Current workflow state
        """
        try:
            config = {"configurable": {"thread_id": workflow_id}}
            # Get current state from memory
            current_state = self.workflow.get_state(config)
            return current_state.values if current_state else {}
            
        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return {'error': f"Failed to get workflow status: {str(e)}"}
    
    def retry_failed_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Retry a failed workflow from the last successful checkpoint
        
        Args:
            workflow_id: ID of the workflow to retry
            
        Returns:
            Updated workflow state
        """
        try:
            logger.info(f"Retrying workflow: {workflow_id}")
            
            config = {"configurable": {"thread_id": workflow_id}}
            
            # Get current state
            current_state = self.workflow.get_state(config)
            if not current_state:
                raise ValueError(f"No workflow found with ID: {workflow_id}")
            
            # Continue execution from current state
            final_state = self.workflow.invoke(None, config)
            
            logger.info(f"Workflow retry completed with status: {final_state.get('agent_status', 'unknown')}")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow retry failed: {e}")
            return {
                'error': f"Workflow retry failed: {str(e)}",
                'workflow_complete': True,
                'agent_status': 'retry_failed'
            }