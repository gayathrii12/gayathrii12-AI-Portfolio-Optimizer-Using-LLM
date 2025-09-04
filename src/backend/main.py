"""
Main entry point for Financial Returns Optimizer.

This module provides the main application class that coordinates all components,
processes user input, manages the complete data flow, and provides both
programmatic and command-line interfaces.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    LOG_LEVEL, LOG_FORMAT, LOGS_DIR, HISTORICAL_DATA_FILE, 
    OUTPUT_DIR, PROJECT_ROOT, RISK_PROFILES
)
from models.data_models import (
    UserInputModel, PortfolioAllocation, ProjectionResult, 
    RiskMetrics, ErrorResponse
)
from agents.orchestrator import (
    FinancialReturnsOrchestrator, OrchestrationInput, OrchestrationResult
)
from utils.terminal_output import TerminalOutputGenerator
from utils.json_output import JSONOutputGenerator
from utils.error_handling import ErrorHandler, ErrorType, ErrorCode


class ApplicationConfig:
    """Configuration management for system parameters."""
    
    def __init__(self):
        """Initialize application configuration."""
        self.data_file_path = str(PROJECT_ROOT / HISTORICAL_DATA_FILE)
        self.output_directory = OUTPUT_DIR
        self.log_directory = LOGS_DIR
        self.enable_retry = True
        self.max_retries = 3
        self.timeout_seconds = 300
        self.risk_profiles = RISK_PROFILES
        
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.
        
        Args:
            config_dict: Configuration parameters to update
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return {
            'data_file_path': self.data_file_path,
            'output_directory': str(self.output_directory),
            'log_directory': str(self.log_directory),
            'enable_retry': self.enable_retry,
            'max_retries': self.max_retries,
            'timeout_seconds': self.timeout_seconds,
            'risk_profiles': self.risk_profiles
        }


class FinancialReturnsOptimizer:
    """
    Main application class that coordinates all components of the Financial Returns Optimizer.
    
    This class manages the complete data flow from user input processing through
    orchestration to output generation, providing both programmatic and CLI interfaces.
    """
    
    def __init__(self, config: Optional[ApplicationConfig] = None):
        """
        Initialize the Financial Returns Optimizer application.
        
        Args:
            config: Optional application configuration
        """
        self.config = config or ApplicationConfig()
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler()
        
        # Initialize output generators
        self.terminal_output = TerminalOutputGenerator()
        self.json_output = JSONOutputGenerator()
        
        # Initialize orchestrator
        self.orchestrator = FinancialReturnsOrchestrator()
        
        self.logger.info("Financial Returns Optimizer initialized")
    
    def validate_user_input(self, user_input: Dict[str, Any]) -> Tuple[bool, Optional[UserInputModel], Optional[str]]:
        """
        Validate and process user input parameters.
        
        Args:
            user_input: Raw user input dictionary
            
        Returns:
            Tuple of (is_valid, validated_input, error_message)
        """
        try:
            # Create UserInputModel with validation
            validated_input = UserInputModel(**user_input)
            
            # Additional business logic validation
            if validated_input.risk_profile not in self.config.risk_profiles:
                return False, None, f"Invalid risk profile. Must be one of: {list(self.config.risk_profiles.keys())}"
            
            # Validate data file exists
            if not Path(self.config.data_file_path).exists():
                return False, None, f"Historical data file not found: {self.config.data_file_path}"
            
            self.logger.info(f"User input validated successfully for {validated_input.risk_profile} risk profile")
            return True, validated_input, None
            
        except Exception as e:
            error_msg = f"Input validation failed: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
    
    def process_investment_request(self, user_input: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Process a complete investment analysis request.
        
        Args:
            user_input: User investment parameters
            
        Returns:
            Tuple of (success, result_data)
        """
        self.logger.info("Processing investment request")
        
        try:
            # Validate user input
            is_valid, validated_input, error_msg = self.validate_user_input(user_input)
            if not is_valid:
                return False, {
                    'error': error_msg,
                    'error_type': 'validation_error'
                }
            
            # Create orchestration input
            orchestration_input = OrchestrationInput(
                user_input=validated_input,
                data_file_path=self.config.data_file_path,
                enable_retry=self.config.enable_retry,
                max_retries=self.config.max_retries,
                timeout_seconds=self.config.timeout_seconds
            )
            
            # Execute orchestration
            self.logger.info("Starting orchestration pipeline")
            orchestration_result = self.orchestrator.orchestrate(orchestration_input)
            
            if not orchestration_result.success:
                self.logger.error(f"Orchestration failed: {orchestration_result.error_message}")
                return False, {
                    'error': orchestration_result.error_message,
                    'error_type': 'orchestration_error',
                    'stage_results': [result.model_dump() for result in orchestration_result.stage_results]
                }
            
            # Generate outputs
            terminal_output = self.generate_terminal_output(orchestration_result, validated_input)
            json_output = self.generate_json_output(orchestration_result, validated_input)
            
            # Save outputs to files
            output_files = self.save_outputs(terminal_output, json_output, validated_input)
            
            self.logger.info("Investment request processed successfully")
            
            return True, {
                'orchestration_result': orchestration_result.model_dump(),
                'terminal_output': terminal_output,
                'json_output': json_output,
                'output_files': output_files,
                'processing_time': orchestration_result.total_duration_seconds
            }
            
        except Exception as e:
            error_response = self.error_handler.handle_exception(
                exception=e,
                component="FinancialReturnsOptimizer",
                operation="process_investment_request",
                user_input=user_input
            )
            
            self.logger.error(f"Request processing failed: {error_response.error_message}")
            return False, {
                'error': error_response.error_message,
                'error_type': 'system_error',
                'error_code': error_response.error_code
            }
    
    def generate_terminal_output(self, orchestration_result: OrchestrationResult, 
                               user_input: UserInputModel) -> str:
        """
        Generate human-readable terminal output.
        
        Args:
            orchestration_result: Results from orchestration
            user_input: User investment parameters
            
        Returns:
            str: Formatted terminal output
        """
        if not all([
            orchestration_result.portfolio_allocation,
            orchestration_result.projections,
            orchestration_result.risk_metrics
        ]):
            return "Error: Incomplete orchestration results for terminal output generation"
        
        return self.terminal_output.generate_complete_report(
            allocation=orchestration_result.portfolio_allocation,
            projections=orchestration_result.projections,
            risk_metrics=orchestration_result.risk_metrics,
            user_input=user_input
        )
    
    def generate_json_output(self, orchestration_result: OrchestrationResult,
                           user_input: UserInputModel) -> Dict[str, Any]:
        """
        Generate JSON output for React frontend.
        
        Args:
            orchestration_result: Results from orchestration
            user_input: User investment parameters
            
        Returns:
            Dict[str, Any]: JSON output structure
        """
        if not all([
            orchestration_result.portfolio_allocation,
            orchestration_result.projections,
            orchestration_result.risk_metrics
        ]):
            return {'error': 'Incomplete orchestration results for JSON output generation'}
        
        complete_json = self.json_output.generate_complete_json(
            allocation=orchestration_result.portfolio_allocation,
            projections=orchestration_result.projections,
            risk_metrics=orchestration_result.risk_metrics,
            user_input=user_input
        )
        
        return self.json_output.export_to_json_dict(complete_json)
    
    def save_outputs(self, terminal_output: str, json_output: Dict[str, Any],
                    user_input: UserInputModel) -> Dict[str, str]:
        """
        Save outputs to files in the output directory.
        
        Args:
            terminal_output: Terminal output string
            json_output: JSON output dictionary
            user_input: User investment parameters
            
        Returns:
            Dict[str, str]: Paths to saved output files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        risk_profile = user_input.risk_profile.lower()
        
        # Create output filenames
        terminal_filename = f"portfolio_analysis_{risk_profile}_{timestamp}.txt"
        json_filename = f"portfolio_data_{risk_profile}_{timestamp}.json"
        
        terminal_path = self.config.output_directory / terminal_filename
        json_path = self.config.output_directory / json_filename
        
        try:
            # Save terminal output
            with open(terminal_path, 'w', encoding='utf-8') as f:
                f.write(terminal_output)
            
            # Save JSON output
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, indent=2, default=str)
            
            self.logger.info(f"Outputs saved to {terminal_path} and {json_path}")
            
            return {
                'terminal_output_file': str(terminal_path),
                'json_output_file': str(json_path)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to save outputs: {str(e)}")
            return {
                'error': f"Failed to save outputs: {str(e)}"
            }
    
    def get_sample_input(self, risk_profile: str = "Moderate") -> Dict[str, Any]:
        """
        Get sample user input for testing and development.
        
        Args:
            risk_profile: Risk profile for sample input
            
        Returns:
            Dict[str, Any]: Sample user input
        """
        return {
            "investment_amount": 100000.0,
            "investment_type": "lumpsum",
            "tenure_years": 10,
            "risk_profile": risk_profile,
            "return_expectation": 12.0,
            "rebalancing_preferences": {
                "frequency": "annual",
                "threshold": 5.0
            },
            "withdrawal_preferences": None
        }


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper()),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOGS_DIR / "app.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_cli_parser() -> argparse.ArgumentParser:
    """
    Create command-line interface parser.
    
    Returns:
        argparse.ArgumentParser: CLI argument parser
    """
    parser = argparse.ArgumentParser(
        description="Financial Returns Optimizer - Portfolio Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with sample data
  python main.py --sample --risk-profile Moderate
  
  # Run with custom parameters
  python main.py --amount 50000 --type sip --years 15 --risk-profile High --return-expectation 15
  
  # Run interactive mode
  python main.py --interactive
        """
    )
    
    parser.add_argument(
        '--sample', 
        action='store_true',
        help='Use sample input data for testing'
    )
    
    parser.add_argument(
        '--interactive', 
        action='store_true',
        help='Run in interactive mode with prompts'
    )
    
    parser.add_argument(
        '--amount', 
        type=float,
        help='Investment amount'
    )
    
    parser.add_argument(
        '--type', 
        choices=['lumpsum', 'sip'],
        help='Investment type'
    )
    
    parser.add_argument(
        '--years', 
        type=int,
        help='Investment tenure in years'
    )
    
    parser.add_argument(
        '--risk-profile', 
        choices=['Low', 'Moderate', 'High'],
        help='Risk tolerance level'
    )
    
    parser.add_argument(
        '--return-expectation', 
        type=float,
        help='Expected annual return percentage'
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def get_interactive_input() -> Dict[str, Any]:
    """
    Get user input through interactive prompts.
    
    Returns:
        Dict[str, Any]: User input parameters
    """
    print("\n" + "="*60)
    print("Financial Returns Optimizer - Interactive Mode")
    print("="*60)
    
    try:
        amount = float(input("Investment amount ($): "))
        
        print("\nInvestment type:")
        print("1. Lumpsum")
        print("2. SIP (Systematic Investment Plan)")
        type_choice = input("Choose (1 or 2): ").strip()
        investment_type = "lumpsum" if type_choice == "1" else "sip"
        
        years = int(input("Investment tenure (years): "))
        
        print("\nRisk profile:")
        print("1. Low (Conservative)")
        print("2. Moderate (Balanced)")
        print("3. High (Aggressive)")
        risk_choice = input("Choose (1, 2, or 3): ").strip()
        risk_profiles = {"1": "Low", "2": "Moderate", "3": "High"}
        risk_profile = risk_profiles.get(risk_choice, "Moderate")
        
        return_expectation = float(input("Expected annual return (%): "))
        
        return {
            "investment_amount": amount,
            "investment_type": investment_type,
            "tenure_years": years,
            "risk_profile": risk_profile,
            "return_expectation": return_expectation,
            "rebalancing_preferences": {"frequency": "annual"},
            "withdrawal_preferences": None
        }
        
    except (ValueError, KeyboardInterrupt) as e:
        print(f"\nInput error: {e}")
        print("Using default sample input...")
        return {
            "investment_amount": 100000.0,
            "investment_type": "lumpsum",
            "tenure_years": 10,
            "risk_profile": "Moderate",
            "return_expectation": 12.0,
            "rebalancing_preferences": {"frequency": "annual"},
            "withdrawal_preferences": None
        }


def main():
    """Main application entry point with CLI support."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Financial Returns Optimizer starting...")
    
    try:
        # Initialize configuration
        config = ApplicationConfig()
        
        # Update config from command line arguments
        if args.output_dir:
            config.output_directory = Path(args.output_dir)
            config.output_directory.mkdir(exist_ok=True)
        
        # Load configuration file if provided
        if args.config:
            try:
                with open(args.config, 'r') as f:
                    config_data = json.load(f)
                config.update_from_dict(config_data)
                logger.info(f"Configuration loaded from {args.config}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # Initialize application
        app = FinancialReturnsOptimizer(config)
        
        # Determine input method
        user_input = None
        
        if args.sample:
            # Use sample input
            risk_profile = args.risk_profile or "Moderate"
            user_input = app.get_sample_input(risk_profile)
            logger.info(f"Using sample input with {risk_profile} risk profile")
            
        elif args.interactive:
            # Interactive input
            user_input = get_interactive_input()
            
        elif all([args.amount, args.type, args.years, args.risk_profile, args.return_expectation]):
            # Command line parameters
            user_input = {
                "investment_amount": args.amount,
                "investment_type": args.type,
                "tenure_years": args.years,
                "risk_profile": args.risk_profile,
                "return_expectation": args.return_expectation,
                "rebalancing_preferences": {"frequency": "annual"},
                "withdrawal_preferences": None
            }
            
        else:
            # Default to interactive mode
            print("No input method specified. Starting interactive mode...")
            user_input = get_interactive_input()
        
        # Process the investment request
        print("\nProcessing investment analysis...")
        success, result = app.process_investment_request(user_input)
        
        if success:
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            
            # Display terminal output
            if 'terminal_output' in result:
                print(result['terminal_output'])
            
            # Show output file locations
            if 'output_files' in result:
                print(f"\nOutput files saved:")
                for file_type, file_path in result['output_files'].items():
                    print(f"  {file_type}: {file_path}")
            
            # Show processing time
            if 'processing_time' in result:
                print(f"\nProcessing time: {result['processing_time']:.2f} seconds")
                
        else:
            print("\n" + "="*60)
            print("ANALYSIS FAILED")
            print("="*60)
            print(f"Error: {result.get('error', 'Unknown error')}")
            
            if 'stage_results' in result:
                print("\nStage Results:")
                for stage_result in result['stage_results']:
                    stage_name = stage_result.get('stage', 'Unknown')
                    success = stage_result.get('success', False)
                    status = "✓" if success else "✗"
                    print(f"  {status} {stage_name}")
        
        logger.info("Application completed")
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        logger.info("Application interrupted by user")
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}", exc_info=True)
        print(f"\nApplication error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()