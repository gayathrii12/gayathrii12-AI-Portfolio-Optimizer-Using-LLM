"""
ML Integration Demo for Financial Returns Optimizer

This demo shows the complete ML model integration with the agent workflow,
demonstrating how ML predictions are used for portfolio allocation.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.workflow_factory import WorkflowFactory, create_workflow
from agents.model_manager import get_model_manager
from models.asset_return_models import AssetReturnModels

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_ml_model_status():
    """Demo: Check ML model status"""
    print("\n" + "="*60)
    print("ML MODEL STATUS DEMO")
    print("="*60)
    
    try:
        # Get model status
        status = WorkflowFactory.get_model_status()
        
        print(f"Model Status: {status['status']}")
        
        if status['status'] == 'initialized':
            print(f"Total Models: {status['total_models']}")
            print(f"Valid Models: {status['valid_models']}")
            
            print("\nModel Validation Results:")
            for asset, result in status['validation_results'].items():
                status_icon = "✓" if result.get('status') == 'valid' else "✗"
                print(f"  {status_icon} {asset}: {result.get('status', 'unknown')}")
                
                if 'test_prediction' in result:
                    prediction = result['test_prediction']
                    print(f"    Test prediction: {prediction:.4f} ({prediction*100:.2f}%)")
        else:
            print(f"Status Message: {status.get('message', 'No additional info')}")
            
    except Exception as e:
        print(f"Error checking model status: {e}")


def demo_ml_predictions():
    """Demo: Get ML model predictions"""
    print("\n" + "="*60)
    print("ML PREDICTIONS DEMO")
    print("="*60)
    
    try:
        # Create workflow with ML models
        workflow = create_workflow()
        if workflow is None:
            print("❌ Failed to create workflow with ML models")
            return
        
        print("✓ Workflow created successfully with ML models")
        
        # Get ML predictions
        prediction_input = {
            'investment_horizon': 10,
            'asset_classes': ['sp500', 'small_cap', 't_bills', 't_bonds', 
                            'corporate_bonds', 'real_estate', 'gold']
        }
        
        print(f"\nGetting ML predictions for {prediction_input['investment_horizon']}-year horizon...")
        
        # Execute return prediction
        result = workflow.return_prediction_agent.predict_returns(prediction_input)
        
        if result.get('agent_status') == 'return_prediction_complete':
            print("✓ ML predictions completed successfully")
            
            predicted_returns = result.get('predicted_returns', {})
            confidence_scores = result.get('confidence_scores', {})
            
            print(f"\nML Model Predictions:")
            print("-" * 50)
            for asset, prediction in predicted_returns.items():
                confidence = confidence_scores.get(asset, 0.5)
                confidence_icon = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
                print(f"{asset:15}: {prediction*100:6.2f}% (confidence: {confidence:.2f} {confidence_icon})")
            
            print(f"\nPrediction Rationale:")
            print(result.get('prediction_rationale', 'No rationale provided'))
            
        else:
            print(f"❌ ML predictions failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error getting ML predictions: {e}")


def demo_ml_enhanced_portfolio():
    """Demo: Complete ML-enhanced portfolio generation"""
    print("\n" + "="*60)
    print("ML-ENHANCED PORTFOLIO DEMO")
    print("="*60)
    
    try:
        # Create workflow
        workflow = create_workflow()
        if workflow is None:
            print("❌ Failed to create workflow")
            return
        
        # Test different risk profiles
        test_scenarios = [
            {
                'name': 'Conservative Investor',
                'input': {
                    'investment_amount': 100000,
                    'investment_horizon': 15,
                    'risk_profile': 'low',
                    'investment_type': 'lump_sum'
                }
            },
            {
                'name': 'Moderate Investor',
                'input': {
                    'investment_amount': 150000,
                    'investment_horizon': 10,
                    'risk_profile': 'moderate',
                    'investment_type': 'lump_sum'
                }
            },
            {
                'name': 'Aggressive Investor',
                'input': {
                    'investment_amount': 200000,
                    'investment_horizon': 20,
                    'risk_profile': 'high',
                    'investment_type': 'lump_sum'
                }
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n{scenario['name']} Scenario:")
            print("-" * 40)
            
            # Execute complete workflow
            result = workflow.execute_workflow(
                scenario['input'], 
                workflow_id=f"demo_{scenario['name'].lower().replace(' ', '_')}"
            )
            
            if result.get('workflow_complete') and result.get('agent_status') == 'completed_successfully':
                print("✓ ML-enhanced portfolio generated successfully")
                
                # Display results
                predicted_returns = result.get('predicted_returns', {})
                portfolio_allocation = result.get('portfolio_allocation', {})
                expected_return = result.get('expected_portfolio_return', 0)
                
                print(f"Expected Portfolio Return: {expected_return*100:.2f}%")
                
                print("\nML-Optimized Allocation:")
                for asset, allocation in sorted(portfolio_allocation.items(), key=lambda x: x[1], reverse=True):
                    if allocation > 0.1:  # Only show allocations > 0.1%
                        ml_return = predicted_returns.get(asset, 0)
                        print(f"  {asset:15}: {allocation:5.1f}% (ML predicts: {ml_return*100:5.2f}%)")
                
                print(f"\nAllocation Rationale:")
                print(result.get('allocation_rationale', 'No rationale provided')[:200] + "...")
                
            else:
                print(f"❌ Portfolio generation failed: {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"Error in ML-enhanced portfolio demo: {e}")


def demo_ml_vs_static_comparison():
    """Demo: Compare ML-enhanced vs static allocation"""
    print("\n" + "="*60)
    print("ML vs STATIC ALLOCATION COMPARISON")
    print("="*60)
    
    try:
        # Create workflow
        workflow = create_workflow()
        if workflow is None:
            print("❌ Failed to create workflow")
            return
        
        # Test input
        test_input = {
            'investment_amount': 100000,
            'investment_horizon': 10,
            'risk_profile': 'moderate',
            'investment_type': 'lump_sum'
        }
        
        print("Comparing ML-enhanced vs static moderate allocation...")
        
        # Get ML-enhanced allocation
        ml_result = workflow.execute_workflow(test_input, workflow_id="ml_comparison")
        
        if ml_result.get('workflow_complete'):
            ml_allocation = ml_result.get('portfolio_allocation', {})
            ml_return = ml_result.get('expected_portfolio_return', 0)
            predicted_returns = ml_result.get('predicted_returns', {})
            
            # Static moderate allocation (from base template)
            static_allocation = {
                'sp500': 30.0, 'small_cap': 10.0, 't_bills': 15.0,
                't_bonds': 20.0, 'corporate_bonds': 10.0,
                'real_estate': 10.0, 'gold': 5.0
            }
            
            # Calculate static expected return
            static_return = sum(
                (static_allocation.get(asset, 0) / 100.0) * predicted_returns.get(asset, 0.06)
                for asset in static_allocation.keys()
            )
            
            print(f"\nResults Comparison:")
            print("-" * 50)
            print(f"ML-Enhanced Expected Return: {ml_return*100:.2f}%")
            print(f"Static Expected Return:      {static_return*100:.2f}%")
            print(f"ML Advantage:                {(ml_return - static_return)*100:+.2f}%")
            
            print(f"\nAllocation Differences:")
            print("-" * 30)
            for asset in set(list(ml_allocation.keys()) + list(static_allocation.keys())):
                ml_alloc = ml_allocation.get(asset, 0)
                static_alloc = static_allocation.get(asset, 0)
                diff = ml_alloc - static_alloc
                
                if abs(diff) > 0.5:  # Only show significant differences
                    direction = "↑" if diff > 0 else "↓"
                    print(f"{asset:15}: ML {ml_alloc:5.1f}% vs Static {static_alloc:5.1f}% ({direction} {abs(diff):4.1f}%)")
            
        else:
            print(f"❌ Comparison failed: {ml_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error in ML vs static comparison: {e}")


def main():
    """Run all ML integration demos"""
    print("🤖 ML INTEGRATION DEMO FOR FINANCIAL RETURNS OPTIMIZER")
    print("=" * 80)
    
    try:
        # Demo 1: Model Status
        demo_ml_model_status()
        
        # Demo 2: ML Predictions
        demo_ml_predictions()
        
        # Demo 3: ML-Enhanced Portfolio
        demo_ml_enhanced_portfolio()
        
        # Demo 4: ML vs Static Comparison
        demo_ml_vs_static_comparison()
        
        print("\n" + "="*60)
        print("✅ ALL ML INTEGRATION DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nKey Integration Features Demonstrated:")
        print("• ML model initialization and validation")
        print("• Real-time ML predictions for asset returns")
        print("• ML-optimized portfolio allocation")
        print("• Error handling for ML model failures")
        print("• Confidence scoring for predictions")
        print("• Comparison with static allocations")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()