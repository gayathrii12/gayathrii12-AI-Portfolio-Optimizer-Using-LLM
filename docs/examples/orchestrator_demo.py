"""
Demo script for the Financial Returns Orchestrator.

This script demonstrates how to use the orchestrator to coordinate
all agents in the financial returns optimization pipeline.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.orchestrator import FinancialReturnsOrchestrator, OrchestrationInput, create_orchestrator
from models.data_models import UserInputModel


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/orchestrator_demo.log')
        ]
    )


def create_sample_user_input() -> UserInputModel:
    """Create sample user input for demonstration."""
    return UserInputModel(
        investment_amount=100000.0,
        investment_type="lumpsum",
        tenure_years=10,
        risk_profile="Moderate",
        return_expectation=8.0,
        rebalancing_preferences={
            "frequency": "annual",
            "threshold": 5.0
        },
        withdrawal_preferences=None
    )


def demonstrate_successful_orchestration():
    """Demonstrate successful orchestration pipeline."""
    print("\n" + "="*60)
    print("DEMONSTRATING SUCCESSFUL ORCHESTRATION PIPELINE")
    print("="*60)
    
    # Create user input
    user_input = create_sample_user_input()
    print(f"User Input:")
    print(f"  Investment Amount: ${user_input.investment_amount:,.2f}")
    print(f"  Investment Type: {user_input.investment_type}")
    print(f"  Tenure: {user_input.tenure_years} years")
    print(f"  Risk Profile: {user_input.risk_profile}")
    print(f"  Expected Return: {user_input.return_expectation}%")
    
    # Create orchestration input
    orchestration_input = OrchestrationInput(
        user_input=user_input,
        data_file_path="histretSP.xls",
        enable_retry=True,
        max_retries=3,
        timeout_seconds=300
    )
    
    # Create orchestrator
    print(f"\nCreating orchestrator...")
    orchestrator = create_orchestrator()
    
    # Check orchestrator status
    status = orchestrator.get_orchestration_status()
    print(f"Orchestrator Status:")
    print(f"  Initialized: {status['orchestrator_initialized']}")
    print(f"  Available Agents: {sum(status['agents_available'].values())}/5")
    print(f"  Tools Count: {status['tools_count']}")
    
    # Execute orchestration
    print(f"\nExecuting orchestration pipeline...")
    result = orchestrator.orchestrate(orchestration_input)
    
    # Display results
    print(f"\nOrchestration Results:")
    print(f"  Success: {result.success}")
    print(f"  Final Stage: {result.final_stage.value}")
    print(f"  Total Duration: {result.total_duration_seconds:.2f} seconds")
    print(f"  Stages Completed: {len(result.stage_results)}")
    
    # Display stage results
    print(f"\nStage Results:")
    for i, stage_result in enumerate(result.stage_results, 1):
        status_icon = "✓" if stage_result.success else "✗"
        print(f"  {i}. {status_icon} {stage_result.stage.value.replace('_', ' ').title()}")
        print(f"     Duration: {stage_result.duration_seconds:.2f}s")
        if stage_result.retry_count > 0:
            print(f"     Retries: {stage_result.retry_count}")
        if stage_result.error_message:
            print(f"     Error: {stage_result.error_message}")
    
    # Display final outputs if successful
    if result.success:
        print(f"\nFinal Outputs:")
        
        if result.cleaned_data:
            print(f"  Cleaned Data: {len(result.cleaned_data)} years of historical data")
        
        if result.expected_returns:
            print(f"  Expected Returns:")
            for asset, return_rate in result.expected_returns.items():
                print(f"    {asset}: {return_rate*100:.2f}%")
        
        if result.portfolio_allocation:
            print(f"  Portfolio Allocation:")
            print(f"    S&P 500: {result.portfolio_allocation.sp500:.1f}%")
            print(f"    Small Cap: {result.portfolio_allocation.small_cap:.1f}%")
            print(f"    Bonds: {result.portfolio_allocation.bonds:.1f}%")
            print(f"    Real Estate: {result.portfolio_allocation.real_estate:.1f}%")
            print(f"    Gold: {result.portfolio_allocation.gold:.1f}%")
        
        if result.projections:
            print(f"  Projections: {len(result.projections)} years")
            if result.projections:
                final_projection = result.projections[-1]
                print(f"    Final Value: ${final_projection.portfolio_value:,.2f}")
                print(f"    CAGR: {final_projection.annual_return:.2f}%")
    
    else:
        print(f"\nOrchestration Failed:")
        print(f"  Error: {result.error_message}")
    
    return result


def demonstrate_error_handling():
    """Demonstrate error handling and retry logic."""
    print("\n" + "="*60)
    print("DEMONSTRATING ERROR HANDLING AND RETRY LOGIC")
    print("="*60)
    
    # Create user input with invalid data file
    user_input = create_sample_user_input()
    
    orchestration_input = OrchestrationInput(
        user_input=user_input,
        data_file_path="nonexistent_file.xls",  # This will cause failure
        enable_retry=True,
        max_retries=2,
        timeout_seconds=60
    )
    
    orchestrator = create_orchestrator()
    
    print(f"Attempting orchestration with invalid data file...")
    result = orchestrator.orchestrate(orchestration_input)
    
    print(f"\nError Handling Results:")
    print(f"  Success: {result.success}")
    print(f"  Failed at Stage: {result.final_stage.value}")
    print(f"  Error Message: {result.error_message}")
    
    # Show retry attempts
    if result.stage_results:
        failed_stage = result.stage_results[0]
        print(f"  Retry Attempts: {failed_stage.retry_count}")
        print(f"  Stage Duration: {failed_stage.duration_seconds:.2f}s")
    
    return result


def demonstrate_different_risk_profiles():
    """Demonstrate orchestration with different risk profiles."""
    print("\n" + "="*60)
    print("DEMONSTRATING DIFFERENT RISK PROFILES")
    print("="*60)
    
    risk_profiles = ["Low", "Moderate", "High"]
    orchestrator = create_orchestrator()
    
    for risk_profile in risk_profiles:
        print(f"\nTesting {risk_profile} Risk Profile:")
        
        user_input = UserInputModel(
            investment_amount=100000.0,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile=risk_profile,
            return_expectation=8.0
        )
        
        orchestration_input = OrchestrationInput(
            user_input=user_input,
            data_file_path="histretSP.xls",
            enable_retry=False,
            max_retries=1
        )
        
        result = orchestrator.orchestrate(orchestration_input)
        
        print(f"  Success: {result.success}")
        print(f"  Stages Completed: {len(result.stage_results)}")
        
        if result.success and result.portfolio_allocation:
            print(f"  Allocation - Stocks: {result.portfolio_allocation.sp500 + result.portfolio_allocation.small_cap:.1f}%")
            print(f"  Allocation - Bonds: {result.portfolio_allocation.bonds:.1f}%")
            print(f"  Allocation - Alternatives: {result.portfolio_allocation.real_estate + result.portfolio_allocation.gold:.1f}%")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n" + "="*60)
    print("DEMONSTRATING PERFORMANCE MONITORING")
    print("="*60)
    
    user_input = create_sample_user_input()
    orchestration_input = OrchestrationInput(
        user_input=user_input,
        data_file_path="histretSP.xls",
        enable_retry=False,
        max_retries=1
    )
    
    orchestrator = create_orchestrator()
    
    # Monitor multiple runs
    run_times = []
    success_count = 0
    
    for i in range(3):
        print(f"\nRun {i+1}:")
        result = orchestrator.orchestrate(orchestration_input)
        
        run_times.append(result.total_duration_seconds)
        if result.success:
            success_count += 1
        
        print(f"  Duration: {result.total_duration_seconds:.2f}s")
        print(f"  Success: {result.success}")
        
        # Show stage breakdown
        for stage_result in result.stage_results:
            print(f"    {stage_result.stage.value}: {stage_result.duration_seconds:.2f}s")
    
    # Performance summary
    print(f"\nPerformance Summary:")
    print(f"  Average Duration: {sum(run_times)/len(run_times):.2f}s")
    print(f"  Min Duration: {min(run_times):.2f}s")
    print(f"  Max Duration: {max(run_times):.2f}s")
    print(f"  Success Rate: {success_count}/{len(run_times)} ({success_count/len(run_times)*100:.1f}%)")


def main():
    """Main demo function."""
    print("Financial Returns Orchestrator Demo")
    print("===================================")
    
    # Setup logging
    setup_logging()
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Demonstrate successful orchestration
        demonstrate_successful_orchestration()
        
        # Demonstrate error handling
        demonstrate_error_handling()
        
        # Demonstrate different risk profiles
        demonstrate_different_risk_profiles()
        
        # Demonstrate performance monitoring
        demonstrate_performance_monitoring()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nCheck 'logs/orchestrator_demo.log' for detailed logs.")
        
    except Exception as e:
        print(f"\nDemo failed with error: {str(e)}")
        logging.error(f"Demo failed: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)