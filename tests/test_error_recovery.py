"""
Unit tests for the error recovery system.

This module tests error recovery strategies, automatic retry mechanisms,
and fallback functionality.
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import Any

from utils.error_recovery import (
    RecoveryStrategy, RecoveryAction, ErrorRecoveryManager, 
    recovery_manager, with_recovery
)
from utils.error_handling import (
    ErrorType, ErrorCode, BaseFinancialError, AgentCommunicationError,
    CalculationError, ErrorContext
)


class TestRecoveryStrategy:
    """Test recovery strategy enum."""
    
    def test_recovery_strategy_values(self):
        """Test RecoveryStrategy enum values."""
        assert RecoveryStrategy.RETRY.value == "retry"
        assert RecoveryStrategy.FALLBACK.value == "fallback"
        assert RecoveryStrategy.SKIP.value == "skip"
        assert RecoveryStrategy.ABORT.value == "abort"
        assert RecoveryStrategy.ALTERNATIVE_METHOD.value == "alternative_method"


class TestRecoveryAction:
    """Test recovery action configuration."""
    
    def test_recovery_action_creation(self):
        """Test creating a recovery action."""
        action = RecoveryAction(
            error_types=[ErrorType.AGENT_COMMUNICATION],
            error_codes=[ErrorCode.AGENT_EXECUTION_TIMEOUT],
            strategy=RecoveryStrategy.RETRY,
            max_attempts=3,
            backoff_factor=2.0
        )
        
        assert action.error_types == [ErrorType.AGENT_COMMUNICATION]
        assert action.error_codes == [ErrorCode.AGENT_EXECUTION_TIMEOUT]
        assert action.strategy == RecoveryStrategy.RETRY
        assert action.max_attempts == 3
        assert action.backoff_factor == 2.0
        assert action.fallback_function is None
        assert action.recovery_data == {}
    
    def test_recovery_action_with_fallback(self):
        """Test recovery action with fallback function."""
        def fallback_func():
            return "fallback_result"
        
        action = RecoveryAction(
            error_types=[ErrorType.CALCULATION_ERROR],
            error_codes=[ErrorCode.DIVISION_BY_ZERO_ERROR],
            strategy=RecoveryStrategy.FALLBACK,
            fallback_function=fallback_func,
            recovery_data={"default_result": "default"}
        )
        
        assert action.fallback_function == fallback_func
        assert action.recovery_data == {"default_result": "default"}


class TestErrorRecoveryManager:
    """Test error recovery manager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.recovery_manager = ErrorRecoveryManager()
    
    def test_default_recovery_actions(self):
        """Test that default recovery actions are set up."""
        assert len(self.recovery_manager.recovery_actions) > 0
        
        # Check for communication error retry action
        comm_actions = [
            action for action in self.recovery_manager.recovery_actions
            if ErrorType.AGENT_COMMUNICATION in action.error_types
        ]
        assert len(comm_actions) > 0
        assert comm_actions[0].strategy == RecoveryStrategy.RETRY
    
    def test_add_recovery_action(self):
        """Test adding custom recovery action."""
        initial_count = len(self.recovery_manager.recovery_actions)
        
        action = RecoveryAction(
            error_types=[ErrorType.DATA_VALIDATION],
            error_codes=[ErrorCode.INVALID_INVESTMENT_AMOUNT],
            strategy=RecoveryStrategy.SKIP
        )
        
        self.recovery_manager.add_recovery_action(action)
        
        assert len(self.recovery_manager.recovery_actions) == initial_count + 1
        assert action in self.recovery_manager.recovery_actions
    
    def test_find_recovery_action(self):
        """Test finding recovery action for specific error."""
        # Add a test action
        action = RecoveryAction(
            error_types=[ErrorType.DATA_VALIDATION],
            error_codes=[ErrorCode.INVALID_INVESTMENT_AMOUNT],
            strategy=RecoveryStrategy.SKIP
        )
        self.recovery_manager.add_recovery_action(action)
        
        # Find the action
        found_action = self.recovery_manager._find_recovery_action(
            ErrorType.DATA_VALIDATION,
            ErrorCode.INVALID_INVESTMENT_AMOUNT
        )
        
        assert found_action == action
    
    def test_find_recovery_action_not_found(self):
        """Test finding recovery action when none exists."""
        found_action = self.recovery_manager._find_recovery_action(
            ErrorType.SECURITY_VIOLATION,
            ErrorCode.MALICIOUS_INPUT_DETECTED
        )
        
        assert found_action is None
    
    def test_execute_with_recovery_success(self):
        """Test successful execution without errors."""
        def successful_function(x, y):
            return x + y
        
        result = self.recovery_manager.execute_with_recovery(
            successful_function, 1, 2,
            component="TestComponent",
            operation="test_operation"
        )
        
        assert result == 3
    
    def test_execute_with_recovery_retry_success(self):
        """Test retry recovery strategy with eventual success."""
        call_count = 0
        
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise AgentCommunicationError(
                    error_type=ErrorType.AGENT_COMMUNICATION,
                    error_code=ErrorCode.AGENT_EXECUTION_TIMEOUT,
                    message="Timeout error"
                )
            return "success"
        
        # Mock time.sleep to speed up test
        with patch('time.sleep'):
            result = self.recovery_manager.execute_with_recovery(
                flaky_function,
                component="TestComponent",
                operation="test_operation"
            )
        
        assert result == "success"
        assert call_count == 3
    
    def test_execute_with_recovery_retry_failure(self):
        """Test retry recovery strategy with ultimate failure."""
        def always_failing_function():
            raise AgentCommunicationError(
                error_type=ErrorType.AGENT_COMMUNICATION,
                error_code=ErrorCode.AGENT_EXECUTION_TIMEOUT,
                message="Timeout error"
            )
        
        # Mock time.sleep to speed up test
        with patch('time.sleep'):
            with pytest.raises(AgentCommunicationError):
                self.recovery_manager.execute_with_recovery(
                    always_failing_function,
                    component="TestComponent",
                    operation="test_operation"
                )
    
    def test_execute_with_recovery_fallback(self):
        """Test fallback recovery strategy."""
        def fallback_function():
            return "fallback_result"
        
        # Add fallback action
        action = RecoveryAction(
            error_types=[ErrorType.CALCULATION_ERROR],
            error_codes=[ErrorCode.DIVISION_BY_ZERO_ERROR],
            strategy=RecoveryStrategy.FALLBACK,
            fallback_function=fallback_function
        )
        self.recovery_manager.add_recovery_action(action)
        
        def failing_function():
            raise CalculationError(
                error_type=ErrorType.CALCULATION_ERROR,
                error_code=ErrorCode.DIVISION_BY_ZERO_ERROR,
                message="Division by zero"
            )
        
        result = self.recovery_manager.execute_with_recovery(
            failing_function,
            component="TestComponent",
            operation="test_operation"
        )
        
        assert result == "fallback_result"
    
    def test_execute_with_recovery_skip(self):
        """Test skip recovery strategy."""
        # Add skip action
        action = RecoveryAction(
            error_types=[ErrorType.DATA_VALIDATION],
            error_codes=[ErrorCode.DATA_OUT_OF_RANGE],
            strategy=RecoveryStrategy.SKIP,
            recovery_data={"skip_result": "skipped"}
        )
        self.recovery_manager.add_recovery_action(action)
        
        def failing_function():
            raise BaseFinancialError(
                error_type=ErrorType.DATA_VALIDATION,
                error_code=ErrorCode.DATA_OUT_OF_RANGE,
                message="Data out of range"
            )
        
        result = self.recovery_manager.execute_with_recovery(
            failing_function,
            component="TestComponent",
            operation="test_operation"
        )
        
        assert result == "skipped"
    
    def test_execute_with_recovery_alternative_method(self):
        """Test alternative method recovery strategy."""
        def alternative_method():
            return "alternative_result"
        
        # Add alternative method action
        action = RecoveryAction(
            error_types=[ErrorType.CALCULATION_ERROR],
            error_codes=[ErrorCode.MATHEMATICAL_OVERFLOW],
            strategy=RecoveryStrategy.ALTERNATIVE_METHOD,
            recovery_data={"alternative_methods": [alternative_method]}
        )
        self.recovery_manager.add_recovery_action(action)
        
        def failing_function():
            raise CalculationError(
                error_type=ErrorType.CALCULATION_ERROR,
                error_code=ErrorCode.MATHEMATICAL_OVERFLOW,
                message="Mathematical overflow"
            )
        
        result = self.recovery_manager.execute_with_recovery(
            failing_function,
            component="TestComponent",
            operation="test_operation"
        )
        
        assert result == "alternative_result"
    
    def test_execute_with_recovery_abort(self):
        """Test abort recovery strategy."""
        # Add abort action
        action = RecoveryAction(
            error_types=[ErrorType.SECURITY_VIOLATION],
            error_codes=[ErrorCode.MALICIOUS_INPUT_DETECTED],
            strategy=RecoveryStrategy.ABORT
        )
        self.recovery_manager.add_recovery_action(action)
        
        def failing_function():
            raise BaseFinancialError(
                error_type=ErrorType.SECURITY_VIOLATION,
                error_code=ErrorCode.MALICIOUS_INPUT_DETECTED,
                message="Malicious input"
            )
        
        with pytest.raises(BaseFinancialError):
            self.recovery_manager.execute_with_recovery(
                failing_function,
                component="TestComponent",
                operation="test_operation"
            )
    
    def test_execute_with_recovery_no_action(self):
        """Test execution when no recovery action is found."""
        def failing_function():
            raise BaseFinancialError(
                error_type=ErrorType.SECURITY_VIOLATION,
                error_code=ErrorCode.MALICIOUS_INPUT_DETECTED,
                message="Malicious input"
            )
        
        with pytest.raises(BaseFinancialError):
            self.recovery_manager.execute_with_recovery(
                failing_function,
                component="TestComponent",
                operation="test_operation"
            )
    
    def test_execute_with_recovery_unknown_exception(self):
        """Test handling unknown exception types."""
        def failing_function():
            raise RuntimeError("Unknown error")
        
        with pytest.raises(BaseFinancialError) as exc_info:
            self.recovery_manager.execute_with_recovery(
                failing_function,
                component="TestComponent",
                operation="test_operation"
            )
        
        # Should be converted to BaseFinancialError
        assert "Unknown error" in str(exc_info.value)


class TestWithRecoveryDecorator:
    """Test the with_recovery decorator."""
    
    def test_with_recovery_decorator_success(self):
        """Test decorator with successful function."""
        @with_recovery("TestComponent", "test_operation")
        def successful_function(x, y):
            return x + y
        
        result = successful_function(1, 2)
        assert result == 3
    
    def test_with_recovery_decorator_with_retry(self):
        """Test decorator with retry recovery."""
        call_count = 0
        
        @with_recovery("TestComponent", "test_operation")
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise AgentCommunicationError(
                    error_type=ErrorType.AGENT_COMMUNICATION,
                    error_code=ErrorCode.AGENT_EXECUTION_TIMEOUT,
                    message="Timeout error"
                )
            return "success"
        
        # Mock time.sleep to speed up test
        with patch('time.sleep'):
            result = flaky_function()
        
        assert result == "success"
        assert call_count == 3
    
    def test_with_recovery_decorator_custom_actions(self):
        """Test decorator with custom recovery actions."""
        def fallback_function():
            return "custom_fallback"
        
        custom_action = RecoveryAction(
            error_types=[ErrorType.DATA_VALIDATION],
            error_codes=[ErrorCode.INVALID_INVESTMENT_AMOUNT],
            strategy=RecoveryStrategy.FALLBACK,
            fallback_function=fallback_function
        )
        
        @with_recovery("TestComponent", "test_operation", [custom_action])
        def failing_function():
            raise BaseFinancialError(
                error_type=ErrorType.DATA_VALIDATION,
                error_code=ErrorCode.INVALID_INVESTMENT_AMOUNT,
                message="Invalid amount"
            )
        
        result = failing_function()
        assert result == "custom_fallback"


class TestIntegration:
    """Integration tests for error recovery system."""
    
    def test_complex_recovery_scenario(self):
        """Test complex recovery scenario with multiple strategies."""
        recovery_manager = ErrorRecoveryManager()
        
        # Add multiple recovery actions
        retry_action = RecoveryAction(
            error_types=[ErrorType.AGENT_COMMUNICATION],
            error_codes=[ErrorCode.AGENT_COMMUNICATION_FAILED],
            strategy=RecoveryStrategy.RETRY,
            max_attempts=2
        )
        
        def fallback_method():
            return "fallback_success"
        
        fallback_action = RecoveryAction(
            error_types=[ErrorType.CALCULATION_ERROR],
            error_codes=[ErrorCode.DIVISION_BY_ZERO_ERROR],
            strategy=RecoveryStrategy.FALLBACK,
            fallback_function=fallback_method
        )
        
        recovery_manager.add_recovery_action(retry_action)
        recovery_manager.add_recovery_action(fallback_action)
        
        # Test retry scenario
        retry_count = 0
        def retry_function():
            nonlocal retry_count
            retry_count += 1
            if retry_count < 2:
                raise AgentCommunicationError(
                    error_type=ErrorType.AGENT_COMMUNICATION,
                    error_code=ErrorCode.AGENT_COMMUNICATION_FAILED,
                    message="Communication failed"
                )
            return "retry_success"
        
        with patch('time.sleep'):
            result = recovery_manager.execute_with_recovery(
                retry_function,
                component="TestComponent",
                operation="retry_test"
            )
        
        assert result == "retry_success"
        assert retry_count == 2
        
        # Test fallback scenario
        def fallback_function():
            raise CalculationError(
                error_type=ErrorType.CALCULATION_ERROR,
                error_code=ErrorCode.DIVISION_BY_ZERO_ERROR,
                message="Division by zero"
            )
        
        result = recovery_manager.execute_with_recovery(
            fallback_function,
            component="TestComponent",
            operation="fallback_test"
        )
        
        assert result == "fallback_success"
    
    def test_recovery_with_context(self):
        """Test recovery with error context preservation."""
        recovery_manager = ErrorRecoveryManager()
        
        def fallback_method(*args, **kwargs):
            # Fallback should receive the same arguments
            return f"fallback_{args[0]}_{kwargs.get('param', 'default')}"
        
        action = RecoveryAction(
            error_types=[ErrorType.DATA_VALIDATION],
            error_codes=[ErrorCode.INVALID_DATA_FORMAT],
            strategy=RecoveryStrategy.FALLBACK,
            fallback_function=fallback_method
        )
        recovery_manager.add_recovery_action(action)
        
        def failing_function(value, param=None):
            raise BaseFinancialError(
                error_type=ErrorType.DATA_VALIDATION,
                error_code=ErrorCode.INVALID_DATA_FORMAT,
                message="Invalid data format",
                context=ErrorContext(
                    component="TestComponent",
                    operation="failing_function",
                    user_input={"value": value, "param": param}
                )
            )
        
        result = recovery_manager.execute_with_recovery(
            failing_function, "test_value",
            component="TestComponent",
            operation="test_operation",
            param="test_param"
        )
        
        assert result == "fallback_test_value_test_param"


if __name__ == "__main__":
    pytest.main([__file__])