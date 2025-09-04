"""
Demo script for Langgraph Agent Workflow

This script demonstrates the complete workflow execution
with sample data to verify all components work together.
"""

import sys
import os
import logging
from unittest.mock import Mock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.langgraph_workflow import FinancialPlanningWorkflow
from models.asset_return_models import AssetReturnModels

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_asset_models():
    """Create mock asset models for demo"""
    mock_models = Mock(spec=AssetReturnModels)
    
    # Mock different returns for different assets
    def mock_predict_returns(asset_class, horizon):
        returns = {
            'sp500': 0.10,
            'small_cap': 0.11,
            't_bills': 0.03,
            't_bonds': 0.05,
            'corporate_bonds': 0.06,
            'real_estate': 0.08,
            'gold': 0.07
        }
        return returns.get(asset_class, 0.06)
    
    mock_models.predict_returns.side_effect = mock_predict_returns
    return mock_models


def demo_workflow_execution():
    """Demonstrate complete workflow execution"""
    logger.info("Starting Langgraph Workflow Demo")
    
    try:
        # Create mock asset models
        asset_models = create_mock_asset_models()
        
        # Create workflow
        logger.info("Creating workflow...")
        workflow = FinancialPlanningWorkflow(asset_models)
        logger.info("Workflow created successfully")
        
        # Sample input data
        input_data = {
            'investment_amount': 100000.0,
            'investment_horizon': 10,
            'risk_profile': 'moderate',
            'investment_type': 'lump_sum'
        }
        
        logger.info(f"Executing workflow with input: {input_data}")
        
        # Execute workflow
        result = workflow.execute_workflow(input_data, "demo_workflow")
        
        # Display results
        logger.info("Workflow execution completed!")
        logger.info(f"Status: {result.get('agent_status', 'unknown')}")
        logger.info(f"Workflow complete: {result.get('workflow_complete', False)}")
        
        if result.get('error'):
            logger.error(f"Error: {result['error']}")
        
        # Display predicted returns
        if 'predicted_returns' in result:
            logger.info("Predicted Returns:")
            for asset, return_rate in result['predicted_returns'].items():
                logger.info(f"  {asset}: {return_rate:.2%}")
        
        # Display portfolio allocation
        if 'portfolio_allocation' in result:
            logger.info("Portfolio Allocation:")
            for asset, allocation in result['portfolio_allocation'].items():
                logger.info(f"  {asset}: {allocation:.1f}%")
        
        # Display expected return
        if 'expected_portfolio_return' in result:
            logger.info(f"Expected Portfolio Return: {result['expected_portfolio_return']:.2%}")
        
        # Display rebalancing schedule
        if 'rebalancing_schedule' in result:
            schedule = result['rebalancing_schedule']
            logger.info(f"Rebalancing Schedule ({len(schedule)} events):")
            for event in schedule[:3]:  # Show first 3 events
                logger.info(f"  Year {event['year']}: {event.get('rationale', 'N/A')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


def demo_different_risk_profiles():
    """Demonstrate workflow with different risk profiles"""
    logger.info("Testing different risk profiles...")
    
    asset_models = create_mock_asset_models()
    workflow = FinancialPlanningWorkflow(asset_models)
    
    risk_profiles = ['low', 'moderate', 'high']
    
    for risk_profile in risk_profiles:
        logger.info(f"\nTesting {risk_profile} risk profile:")
        
        input_data = {
            'investment_amount': 50000.0,
            'investment_horizon': 15,
            'risk_profile': risk_profile,
            'investment_type': 'lump_sum'
        }
        
        result = workflow.execute_workflow(input_data, f"demo_{risk_profile}")
        
        if result.get('portfolio_allocation'):
            allocation = result['portfolio_allocation']
            equity_allocation = allocation.get('sp500', 0) + allocation.get('small_cap', 0)
            logger.info(f"  Total equity allocation: {equity_allocation:.1f}%")
            logger.info(f"  Expected return: {result.get('expected_portfolio_return', 0):.2%}")


def demo_error_handling():
    """Demonstrate error handling capabilities"""
    logger.info("Testing error handling...")
    
    # Create mock that fails
    failing_models = Mock(spec=AssetReturnModels)
    failing_models.predict_returns.side_effect = Exception("Mock ML model failure")
    
    workflow = FinancialPlanningWorkflow(failing_models)
    
    input_data = {
        'investment_amount': 75000.0,
        'investment_horizon': 8,
        'risk_profile': 'moderate',
        'investment_type': 'lump_sum'
    }
    
    result = workflow.execute_workflow(input_data, "demo_error")
    
    logger.info(f"Error handling result: {result.get('agent_status', 'unknown')}")
    logger.info(f"Workflow completed despite errors: {result.get('workflow_complete', False)}")
    
    # Should still have fallback results
    if 'predicted_returns' in result:
        logger.info("Fallback returns provided successfully")
    if 'portfolio_allocation' in result:
        logger.info("Fallback allocation provided successfully")


if __name__ == "__main__":
    try:
        # Run basic demo
        demo_workflow_execution()
        
        print("\n" + "="*50)
        
        # Test different risk profiles
        demo_different_risk_profiles()
        
        print("\n" + "="*50)
        
        # Test error handling
        demo_error_handling()
        
        print("\n" + "="*50)
        print("All demos completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo script failed: {e}")
        sys.exit(1)