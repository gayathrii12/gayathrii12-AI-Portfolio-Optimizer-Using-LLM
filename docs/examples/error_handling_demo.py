"""
Demonstration of the comprehensive error handling system.

This example shows how the error handling and recovery system works
in practice with various error scenarios.
"""

import logging
import tempfile
import os
from pathlib import Path

from utils.error_handling import (
    ErrorHandler, ErrorType, ErrorCode, ErrorContext,
    InputSanitizer, BaseFinancialError, DataValidationError,
    CalculationError, SecurityError, handle_errors
)
from utils.error_recovery import (
    ErrorRecoveryManager, RecoveryAction, RecoveryStrategy,
    with_recovery
)
from models.data_models import UserInputModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_input_sanitization():
    """Demonstrate input sanitization and security validation."""
    print("\n=== INPUT SANITIZATION DEMO ===")
    
    # Test valid input
    try:
        clean_string = InputSanitizer.sanitize_string("  Valid input  ", "test_field")
        print(f"✓ Valid input sanitized: '{clean_string}'")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Test malicious input detection
    try:
        InputSanitizer.sanitize_string("<script>alert('xss')</script>", "test_field")
        print("✗ Should have detected malicious script")
    except SecurityError as e:
        print(f"✓ Detected malicious input: {e.message}")
    
    # Test file path validation
    try:
        # Create a temporary valid file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            tmp.write(b"test data")
            tmp_path = tmp.name
        
        validated_path = InputSanitizer.validate_file_path(tmp_path)
        print(f"✓ Valid file path: {validated_path}")
        
        # Clean up
        os.unlink(tmp_path)
        
    except Exception as e:
        print(f"✗ File validation error: {e}")
    
    # Test invalid file extension
    try:
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"test data")
            tmp_path = tmp.name
        
        InputSanitizer.validate_file_path(tmp_path)
        print("✗ Should have rejected .txt file")
        
    except SecurityError as e:
        print(f"✓ Rejected invalid file extension: {e.message}")
        os.unlink(tmp_path)
    
    # Test dictionary sanitization
    try:
        test_dict = {
            "valid_key": "valid_value",
            "nested": {"inner": "value"},
            "list_data": ["item1", "item2"]
        }
        
        sanitized = InputSanitizer.sanitize_dict(test_dict)
        print(f"✓ Dictionary sanitized successfully: {len(sanitized)} keys")
        
    except Exception as e:
        print(f"✗ Dictionary sanitization error: {e}")


def demo_error_classification():
    """Demonstrate error classification and handling."""
    print("\n=== ERROR CLASSIFICATION DEMO ===")
    
    error_handler = ErrorHandler()
    
    # Test different error types
    test_errors = [
        ValueError("Invalid investment amount"),
        FileNotFoundError("Data file not found"),
        PermissionError("Access denied"),
        ZeroDivisionError("Division by zero in calculation"),
        MemoryError("Out of memory"),
        TimeoutError("Operation timed out"),
    ]
    
    for error in test_errors:
        try:
            raise error
        except Exception as e:
            response = error_handler.handle_exception(
                exception=e,
                component="DemoComponent",
                operation="demo_operation",
                user_input={"test": "data"}
            )
            
            print(f"✓ {type(e).__name__} -> {response.error_type} (Code: {response.error_code})")
            print(f"  Message: {response.error_message}")
            print(f"  Suggestion: {response.suggested_action}")
    
    # Show error statistics
    stats = error_handler.get_error_statistics()
    print(f"\n✓ Error statistics: {stats['total_errors']} total errors")
    print(f"  Error types: {list(stats['error_types'].keys())}")


@handle_errors("DemoComponent", "risky_calculation")
def risky_calculation(x, y):
    """Example function that might fail."""
    if y == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    if x < 0:
        raise ValueError("Negative values not allowed")
    return x / y


def demo_error_decorator():
    """Demonstrate error handling decorator."""
    print("\n=== ERROR DECORATOR DEMO ===")
    
    # Test successful calculation
    try:
        result = risky_calculation(10, 2)
        print(f"✓ Successful calculation: 10 / 2 = {result}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Test division by zero
    try:
        result = risky_calculation(10, 0)
        print(f"✗ Should have failed: {result}")
    except BaseFinancialError as e:
        print(f"✓ Caught financial error: {e.message}")
        print(f"  Error type: {e.error_type}")
        print(f"  Error code: {e.error_code}")
    
    # Test negative value
    try:
        result = risky_calculation(-5, 2)
        print(f"✗ Should have failed: {result}")
    except BaseFinancialError as e:
        print(f"✓ Caught validation error: {e.message}")


def demo_error_recovery():
    """Demonstrate error recovery mechanisms."""
    print("\n=== ERROR RECOVERY DEMO ===")
    
    recovery_manager = ErrorRecoveryManager()
    
    # Add custom recovery action
    def fallback_calculation():
        return "fallback_result"
    
    recovery_action = RecoveryAction(
        error_types=[ErrorType.CALCULATION_ERROR],
        error_codes=[ErrorCode.DIVISION_BY_ZERO_ERROR],
        strategy=RecoveryStrategy.FALLBACK,
        fallback_function=fallback_calculation
    )
    recovery_manager.add_recovery_action(recovery_action)
    
    # Test function that fails then recovers
    def failing_function():
        raise CalculationError(
            error_type=ErrorType.CALCULATION_ERROR,
            error_code=ErrorCode.DIVISION_BY_ZERO_ERROR,
            message="Division by zero"
        )
    
    try:
        result = recovery_manager.execute_with_recovery(
            failing_function,
            component="DemoComponent",
            operation="demo_recovery"
        )
        print(f"✓ Recovery successful: {result}")
    except Exception as e:
        print(f"✗ Recovery failed: {e}")
    
    # Test retry mechanism
    attempt_count = 0
    
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise BaseFinancialError(
                error_type=ErrorType.AGENT_COMMUNICATION,
                error_code=ErrorCode.AGENT_EXECUTION_TIMEOUT,
                message="Timeout error"
            )
        return f"success_after_{attempt_count}_attempts"
    
    try:
        result = recovery_manager.execute_with_recovery(
            flaky_function,
            component="DemoComponent",
            operation="demo_retry"
        )
        print(f"✓ Retry successful: {result}")
    except Exception as e:
        print(f"✗ Retry failed: {e}")


@with_recovery("DemoComponent", "demo_with_decorator")
def function_with_recovery():
    """Function that uses recovery decorator."""
    raise BaseFinancialError(
        error_type=ErrorType.AGENT_COMMUNICATION,
        error_code=ErrorCode.AGENT_COMMUNICATION_FAILED,
        message="Communication failed"
    )


def demo_recovery_decorator():
    """Demonstrate recovery decorator."""
    print("\n=== RECOVERY DECORATOR DEMO ===")
    
    try:
        result = function_with_recovery()
        print(f"✓ Function succeeded: {result}")
    except BaseFinancialError as e:
        print(f"✓ Function failed after recovery attempts: {e.message}")


def demo_user_input_validation():
    """Demonstrate user input validation with error handling."""
    print("\n=== USER INPUT VALIDATION DEMO ===")
    
    # Test valid user input
    try:
        valid_input = UserInputModel(
            investment_amount=10000.0,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=8.0
        )
        print(f"✓ Valid user input created: {valid_input.investment_amount}")
    except Exception as e:
        print(f"✗ Validation error: {e}")
    
    # Test invalid investment amount
    try:
        invalid_input = UserInputModel(
            investment_amount=-1000.0,  # Invalid negative amount
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Moderate",
            return_expectation=8.0
        )
        print(f"✗ Should have failed validation: {invalid_input}")
    except Exception as e:
        print(f"✓ Caught validation error: {e}")
    
    # Test invalid risk profile
    try:
        invalid_input = UserInputModel(
            investment_amount=10000.0,
            investment_type="lumpsum",
            tenure_years=10,
            risk_profile="Invalid",  # Invalid risk profile
            return_expectation=8.0
        )
        print(f"✗ Should have failed validation: {invalid_input}")
    except Exception as e:
        print(f"✓ Caught validation error: {e}")


def demo_edge_cases():
    """Demonstrate edge case handling."""
    print("\n=== EDGE CASES DEMO ===")
    
    error_handler = ErrorHandler()
    
    # Test empty input
    try:
        InputSanitizer.sanitize_string("", "empty_field")
        print("✓ Empty string handled correctly")
    except Exception as e:
        print(f"✗ Empty string error: {e}")
    
    # Test very large input
    try:
        large_string = "a" * (InputSanitizer.MAX_STRING_LENGTH + 1)
        InputSanitizer.sanitize_string(large_string, "large_field")
        print("✗ Should have rejected large string")
    except SecurityError as e:
        print(f"✓ Large string rejected: {e.message}")
    
    # Test deeply nested dictionary
    try:
        nested_dict = {}
        current = nested_dict
        for i in range(InputSanitizer.MAX_DICT_DEPTH + 1):
            current["nested"] = {}
            current = current["nested"]
        
        InputSanitizer.sanitize_dict(nested_dict)
        print("✗ Should have rejected deeply nested dict")
    except SecurityError as e:
        print(f"✓ Deep nesting rejected: {e.message}")
    
    # Test error statistics after multiple errors
    for i in range(5):
        try:
            raise ValueError(f"Test error {i}")
        except Exception as e:
            error_handler.handle_exception(
                exception=e,
                component="EdgeCaseDemo",
                operation="test_error"
            )
    
    stats = error_handler.get_error_statistics()
    print(f"✓ Generated {stats['total_errors']} test errors")
    
    # Clear error log
    error_handler.clear_error_log()
    stats_after_clear = error_handler.get_error_statistics()
    print(f"✓ Error log cleared: {stats_after_clear['total_errors']} errors remaining")


def main():
    """Run all error handling demonstrations."""
    print("COMPREHENSIVE ERROR HANDLING SYSTEM DEMO")
    print("=" * 50)
    
    try:
        demo_input_sanitization()
        demo_error_classification()
        demo_error_decorator()
        demo_error_recovery()
        demo_recovery_decorator()
        demo_user_input_validation()
        demo_edge_cases()
        
        print("\n" + "=" * 50)
        print("✓ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("The error handling system is working correctly!")
        
    except Exception as e:
        print(f"\n✗ DEMO FAILED: {e}")
        logger.exception("Demo failed with exception")


if __name__ == "__main__":
    main()