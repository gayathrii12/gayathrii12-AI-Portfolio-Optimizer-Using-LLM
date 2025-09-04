"""
Unit tests for the comprehensive error handling system.

This module tests error classification, input sanitization, security validation,
and error recovery mechanisms.
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from pydantic import ValidationError, BaseModel, Field

from utils.error_handling import (
    ErrorType, ErrorSeverity, ErrorCode, ErrorContext, DetailedErrorResponse,
    InputSanitizer, BaseFinancialError, DataValidationError, CalculationError,
    AgentCommunicationError, SecurityError, SystemError, ErrorHandler,
    handle_errors, error_handler
)


class TestErrorTypes:
    """Test error type classifications."""
    
    def test_error_type_enum(self):
        """Test ErrorType enum values."""
        assert ErrorType.DATA_VALIDATION.value == "data_validation"
        assert ErrorType.CALCULATION_ERROR.value == "calculation_error"
        assert ErrorType.AGENT_COMMUNICATION.value == "agent_communication"
        assert ErrorType.SECURITY_VIOLATION.value == "security_violation"
    
    def test_error_severity_enum(self):
        """Test ErrorSeverity enum values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"
    
    def test_error_code_enum(self):
        """Test ErrorCode enum values."""
        assert ErrorCode.INVALID_INVESTMENT_AMOUNT.value == 1001
        assert ErrorCode.PORTFOLIO_ALLOCATION_ERROR.value == 2001
        assert ErrorCode.AGENT_INITIALIZATION_FAILED.value == 3001
        assert ErrorCode.FILE_NOT_FOUND.value == 4001
        assert ErrorCode.INVALID_FILE_PATH.value == 5001


class TestErrorContext:
    """Test error context model."""
    
    def test_error_context_creation(self):
        """Test creating error context."""
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation",
            user_input={"test": "value"},
            system_state={"state": "active"}
        )
        
        assert context.component == "TestComponent"
        assert context.operation == "test_operation"
        assert context.user_input == {"test": "value"}
        assert context.system_state == {"state": "active"}
    
    def test_error_context_optional_fields(self):
        """Test error context with optional fields."""
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation"
        )
        
        assert context.component == "TestComponent"
        assert context.operation == "test_operation"
        assert context.user_input is None
        assert context.system_state is None
        assert context.stack_trace is None


class TestInputSanitizer:
    """Test input sanitization and security validation."""
    
    def test_sanitize_string_valid(self):
        """Test sanitizing valid string input."""
        result = InputSanitizer.sanitize_string("  valid input  ", "test_field")
        assert result == "valid input"
    
    def test_sanitize_string_malicious_script(self):
        """Test detecting malicious script tags."""
        with pytest.raises(SecurityError) as exc_info:
            InputSanitizer.sanitize_string("<script>alert('xss')</script>", "test_field")
        
        assert exc_info.value.error_type == ErrorType.SECURITY_VIOLATION
        assert exc_info.value.error_code == ErrorCode.MALICIOUS_INPUT_DETECTED
    
    def test_sanitize_string_javascript_url(self):
        """Test detecting JavaScript URLs."""
        with pytest.raises(SecurityError) as exc_info:
            InputSanitizer.sanitize_string("javascript:alert('xss')", "test_field")
        
        assert exc_info.value.error_type == ErrorType.SECURITY_VIOLATION
        assert exc_info.value.error_code == ErrorCode.MALICIOUS_INPUT_DETECTED
    
    def test_sanitize_string_too_long(self):
        """Test string length validation."""
        long_string = "a" * (InputSanitizer.MAX_STRING_LENGTH + 1)
        
        with pytest.raises(SecurityError) as exc_info:
            InputSanitizer.sanitize_string(long_string, "test_field")
        
        assert exc_info.value.error_type == ErrorType.INPUT_SANITIZATION
        assert exc_info.value.error_code == ErrorCode.INPUT_TOO_LARGE
    
    def test_sanitize_string_non_string(self):
        """Test non-string input."""
        with pytest.raises(SecurityError) as exc_info:
            InputSanitizer.sanitize_string(123, "test_field")
        
        assert exc_info.value.error_type == ErrorType.INPUT_SANITIZATION
        assert exc_info.value.error_code == ErrorCode.MALICIOUS_INPUT_DETECTED
    
    def test_sanitize_string_null_bytes(self):
        """Test removing null bytes."""
        result = InputSanitizer.sanitize_string("test\x00string", "test_field")
        assert result == "teststring"
    
    def test_validate_file_path_valid(self):
        """Test validating valid file path."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            tmp.write(b"test data")
            tmp_path = tmp.name
        
        try:
            result = InputSanitizer.validate_file_path(tmp_path)
            assert Path(result).exists()
        finally:
            os.unlink(tmp_path)
    
    def test_validate_file_path_not_found(self):
        """Test file not found error."""
        with pytest.raises(DataValidationError) as exc_info:
            InputSanitizer.validate_file_path("/nonexistent/file.xlsx")
        
        assert exc_info.value.error_type == ErrorType.FILE_VALIDATION
        assert exc_info.value.error_code == ErrorCode.FILE_NOT_FOUND
    
    def test_validate_file_path_invalid_extension(self):
        """Test invalid file extension."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"test data")
            tmp_path = tmp.name
        
        try:
            with pytest.raises(SecurityError) as exc_info:
                InputSanitizer.validate_file_path(tmp_path)
            
            assert exc_info.value.error_type == ErrorType.FILE_VALIDATION
            assert exc_info.value.error_code == ErrorCode.INVALID_FILE_PATH
        finally:
            os.unlink(tmp_path)
    
    def test_validate_file_path_non_string(self):
        """Test non-string file path."""
        with pytest.raises(SecurityError) as exc_info:
            InputSanitizer.validate_file_path(123)
        
        assert exc_info.value.error_type == ErrorType.FILE_VALIDATION
        assert exc_info.value.error_code == ErrorCode.INVALID_FILE_PATH
    
    def test_sanitize_dict_valid(self):
        """Test sanitizing valid dictionary."""
        data = {
            "key1": "value1",
            "key2": {"nested": "value2"},
            "key3": ["item1", "item2"]
        }
        
        result = InputSanitizer.sanitize_dict(data)
        assert result["key1"] == "value1"
        assert result["key2"]["nested"] == "value2"
        assert result["key3"] == ["item1", "item2"]
    
    def test_sanitize_dict_too_deep(self):
        """Test dictionary nesting too deep."""
        # Create deeply nested dict
        data = {}
        current = data
        for i in range(InputSanitizer.MAX_DICT_DEPTH + 1):
            current["nested"] = {}
            current = current["nested"]
        
        with pytest.raises(SecurityError) as exc_info:
            InputSanitizer.sanitize_dict(data)
        
        assert exc_info.value.error_type == ErrorType.INPUT_SANITIZATION
        assert exc_info.value.error_code == ErrorCode.INPUT_TOO_LARGE
    
    def test_sanitize_dict_too_large(self):
        """Test dictionary too large."""
        data = {f"key_{i}": f"value_{i}" for i in range(InputSanitizer.MAX_LIST_LENGTH + 1)}
        
        with pytest.raises(SecurityError) as exc_info:
            InputSanitizer.sanitize_dict(data)
        
        assert exc_info.value.error_type == ErrorType.INPUT_SANITIZATION
        assert exc_info.value.error_code == ErrorCode.INPUT_TOO_LARGE
    
    def test_sanitize_dict_non_dict(self):
        """Test non-dictionary input."""
        with pytest.raises(SecurityError) as exc_info:
            InputSanitizer.sanitize_dict("not a dict")
        
        assert exc_info.value.error_type == ErrorType.INPUT_SANITIZATION
        assert exc_info.value.error_code == ErrorCode.MALICIOUS_INPUT_DETECTED
    
    def test_sanitize_list_valid(self):
        """Test sanitizing valid list."""
        data = ["item1", {"key": "value"}, ["nested", "list"]]
        
        result = InputSanitizer.sanitize_list(data)
        assert result[0] == "item1"
        assert result[1]["key"] == "value"
        assert result[2] == ["nested", "list"]
    
    def test_sanitize_list_too_large(self):
        """Test list too large."""
        data = [f"item_{i}" for i in range(InputSanitizer.MAX_LIST_LENGTH + 1)]
        
        with pytest.raises(SecurityError) as exc_info:
            InputSanitizer.sanitize_list(data)
        
        assert exc_info.value.error_type == ErrorType.INPUT_SANITIZATION
        assert exc_info.value.error_code == ErrorCode.INPUT_TOO_LARGE
    
    def test_sanitize_list_non_list(self):
        """Test non-list input."""
        with pytest.raises(SecurityError) as exc_info:
            InputSanitizer.sanitize_list("not a list")
        
        assert exc_info.value.error_type == ErrorType.INPUT_SANITIZATION
        assert exc_info.value.error_code == ErrorCode.MALICIOUS_INPUT_DETECTED


class TestBaseFinancialError:
    """Test base financial error class."""
    
    def test_base_financial_error_creation(self):
        """Test creating base financial error."""
        context = ErrorContext(component="TestComponent", operation="test_op")
        
        error = BaseFinancialError(
            error_type=ErrorType.DATA_VALIDATION,
            error_code=ErrorCode.INVALID_INVESTMENT_AMOUNT,
            message="Test error message",
            context=context,
            severity=ErrorSeverity.HIGH
        )
        
        assert error.error_type == ErrorType.DATA_VALIDATION
        assert error.error_code == ErrorCode.INVALID_INVESTMENT_AMOUNT
        assert error.message == "Test error message"
        assert error.context == context
        assert error.severity == ErrorSeverity.HIGH
        assert isinstance(error.timestamp, datetime)
    
    def test_to_error_response(self):
        """Test converting error to response."""
        context = ErrorContext(component="TestComponent", operation="test_op")
        
        error = BaseFinancialError(
            error_type=ErrorType.DATA_VALIDATION,
            error_code=ErrorCode.INVALID_INVESTMENT_AMOUNT,
            message="Test error message",
            context=context
        )
        
        response = error.to_error_response()
        
        assert isinstance(response, DetailedErrorResponse)
        assert response.error_type == ErrorType.DATA_VALIDATION.value
        assert response.error_code == ErrorCode.INVALID_INVESTMENT_AMOUNT.value
        assert response.error_message == "Test error message"
        assert response.error_context == context
    
    def test_suggested_actions(self):
        """Test suggested actions for different error codes."""
        error = BaseFinancialError(
            error_type=ErrorType.DATA_VALIDATION,
            error_code=ErrorCode.INVALID_INVESTMENT_AMOUNT,
            message="Invalid amount"
        )
        
        response = error.to_error_response()
        assert "valid investment amount" in response.suggested_action.lower()


class TestSpecificErrors:
    """Test specific error types."""
    
    def test_data_validation_error(self):
        """Test DataValidationError."""
        error = DataValidationError(
            error_type=ErrorType.DATA_VALIDATION,
            error_code=ErrorCode.INVALID_DATA_FORMAT,
            message="Invalid data format"
        )
        
        assert error.severity == ErrorSeverity.MEDIUM
        assert len(error.recovery_suggestions) > 0
    
    def test_calculation_error(self):
        """Test CalculationError."""
        error = CalculationError(
            error_type=ErrorType.CALCULATION_ERROR,
            error_code=ErrorCode.DIVISION_BY_ZERO_ERROR,
            message="Division by zero"
        )
        
        assert error.severity == ErrorSeverity.HIGH
        assert len(error.recovery_suggestions) > 0
    
    def test_agent_communication_error(self):
        """Test AgentCommunicationError."""
        error = AgentCommunicationError(
            error_type=ErrorType.AGENT_COMMUNICATION,
            error_code=ErrorCode.AGENT_EXECUTION_TIMEOUT,
            message="Agent timeout"
        )
        
        assert error.severity == ErrorSeverity.HIGH
        assert len(error.recovery_suggestions) > 0
    
    def test_security_error(self):
        """Test SecurityError."""
        error = SecurityError(
            error_type=ErrorType.SECURITY_VIOLATION,
            error_code=ErrorCode.MALICIOUS_INPUT_DETECTED,
            message="Malicious input"
        )
        
        assert error.severity == ErrorSeverity.CRITICAL
        assert len(error.recovery_suggestions) > 0
    
    def test_system_error(self):
        """Test SystemError."""
        error = SystemError(
            error_type=ErrorType.SYSTEM_ERROR,
            error_code=ErrorCode.MEMORY_ERROR,
            message="Out of memory"
        )
        
        assert error.severity == ErrorSeverity.CRITICAL
        assert len(error.recovery_suggestions) > 0


class TestErrorHandler:
    """Test error handler functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.error_handler = ErrorHandler()
    
    def test_handle_base_financial_error(self):
        """Test handling BaseFinancialError."""
        original_error = DataValidationError(
            error_type=ErrorType.DATA_VALIDATION,
            error_code=ErrorCode.INVALID_INVESTMENT_AMOUNT,
            message="Invalid amount"
        )
        
        response = self.error_handler.handle_exception(
            exception=original_error,
            component="TestComponent",
            operation="test_operation"
        )
        
        assert isinstance(response, DetailedErrorResponse)
        assert response.error_type == ErrorType.DATA_VALIDATION.value
        assert response.error_code == ErrorCode.INVALID_INVESTMENT_AMOUNT.value
    
    def test_handle_validation_error(self):
        """Test handling Pydantic ValidationError."""
        # Create a simple model to trigger validation error
        class TestModel(BaseModel):
            amount: float = Field(gt=0)
        
        try:
            TestModel(amount=-100)
        except ValidationError as e:
            response = self.error_handler.handle_exception(
                exception=e,
                component="TestComponent",
                operation="test_operation"
            )
            
            assert response.error_type == ErrorType.DATA_VALIDATION.value
            assert response.error_code == ErrorCode.INVALID_DATA_FORMAT.value
            assert "amount" in response.error_message
    
    def test_handle_value_error(self):
        """Test handling ValueError."""
        error = ValueError("Invalid value provided")
        
        response = self.error_handler.handle_exception(
            exception=error,
            component="TestComponent",
            operation="test_operation"
        )
        
        assert response.error_type == ErrorType.DATA_VALIDATION.value
        assert response.error_code == ErrorCode.DATA_OUT_OF_RANGE.value
        assert "Invalid value" in response.error_message
    
    def test_handle_file_not_found_error(self):
        """Test handling FileNotFoundError."""
        error = FileNotFoundError("File not found")
        
        response = self.error_handler.handle_exception(
            exception=error,
            component="TestComponent",
            operation="test_operation"
        )
        
        assert response.error_type == ErrorType.FILE_VALIDATION.value
        assert response.error_code == ErrorCode.FILE_NOT_FOUND.value
    
    def test_handle_permission_error(self):
        """Test handling PermissionError."""
        error = PermissionError("Permission denied")
        
        response = self.error_handler.handle_exception(
            exception=error,
            component="TestComponent",
            operation="test_operation"
        )
        
        assert response.error_type == ErrorType.SYSTEM_ERROR.value
        assert response.error_code == ErrorCode.ACCESS_DENIED.value
    
    def test_handle_memory_error(self):
        """Test handling MemoryError."""
        error = MemoryError("Out of memory")
        
        response = self.error_handler.handle_exception(
            exception=error,
            component="TestComponent",
            operation="test_operation"
        )
        
        assert response.error_type == ErrorType.SYSTEM_ERROR.value
        assert response.error_code == ErrorCode.MEMORY_ERROR.value
        assert response.error_severity == ErrorSeverity.CRITICAL
    
    def test_handle_timeout_error(self):
        """Test handling TimeoutError."""
        error = TimeoutError("Operation timed out")
        
        response = self.error_handler.handle_exception(
            exception=error,
            component="TestComponent",
            operation="test_operation"
        )
        
        assert response.error_type == ErrorType.AGENT_COMMUNICATION.value
        assert response.error_code == ErrorCode.AGENT_EXECUTION_TIMEOUT.value
    
    def test_handle_zero_division_error(self):
        """Test handling ZeroDivisionError."""
        error = ZeroDivisionError("Division by zero")
        
        response = self.error_handler.handle_exception(
            exception=error,
            component="TestComponent",
            operation="test_operation"
        )
        
        assert response.error_type == ErrorType.CALCULATION_ERROR.value
        assert response.error_code == ErrorCode.DIVISION_BY_ZERO_ERROR.value
    
    def test_handle_overflow_error(self):
        """Test handling OverflowError."""
        error = OverflowError("Numerical overflow")
        
        response = self.error_handler.handle_exception(
            exception=error,
            component="TestComponent",
            operation="test_operation"
        )
        
        assert response.error_type == ErrorType.CALCULATION_ERROR.value
        assert response.error_code == ErrorCode.MATHEMATICAL_OVERFLOW.value
    
    def test_handle_unknown_error(self):
        """Test handling unknown error types."""
        error = RuntimeError("Unknown error")
        
        response = self.error_handler.handle_exception(
            exception=error,
            component="TestComponent",
            operation="test_operation"
        )
        
        assert response.error_type == ErrorType.SYSTEM_ERROR.value
        assert response.error_code == ErrorCode.SYSTEM_RESOURCE_ERROR.value
        assert response.error_severity == ErrorSeverity.CRITICAL
    
    def test_error_statistics(self):
        """Test error statistics collection."""
        # Generate some errors
        errors = [
            ValueError("Error 1"),
            FileNotFoundError("Error 2"),
            ValueError("Error 3"),
        ]
        
        for error in errors:
            self.error_handler.handle_exception(
                exception=error,
                component="TestComponent",
                operation="test_operation"
            )
        
        stats = self.error_handler.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert ErrorType.DATA_VALIDATION.value in stats["error_types"]
        assert ErrorType.FILE_VALIDATION.value in stats["error_types"]
        assert stats["error_types"][ErrorType.DATA_VALIDATION.value] == 2
        assert stats["error_types"][ErrorType.FILE_VALIDATION.value] == 1
    
    def test_clear_error_log(self):
        """Test clearing error log."""
        # Generate an error
        error = ValueError("Test error")
        self.error_handler.handle_exception(
            exception=error,
            component="TestComponent",
            operation="test_operation"
        )
        
        assert len(self.error_handler.error_log) == 1
        
        # Clear the log
        self.error_handler.clear_error_log()
        
        assert len(self.error_handler.error_log) == 0
        assert len(self.error_handler.error_counts) == 0


class TestErrorDecorator:
    """Test error handling decorator."""
    
    def test_handle_errors_decorator_success(self):
        """Test decorator with successful function."""
        @handle_errors("TestComponent", "test_operation")
        def successful_function(x, y):
            return x + y
        
        result = successful_function(1, 2)
        assert result == 3
    
    def test_handle_errors_decorator_exception(self):
        """Test decorator with exception."""
        @handle_errors("TestComponent", "test_operation")
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(BaseFinancialError) as exc_info:
            failing_function()
        
        assert exc_info.value.error_type == ErrorType.DATA_VALIDATION
        assert exc_info.value.error_code == ErrorCode.DATA_OUT_OF_RANGE
        assert "Test error" in exc_info.value.message


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_string_sanitization(self):
        """Test sanitizing empty string."""
        result = InputSanitizer.sanitize_string("", "test_field")
        assert result == ""
    
    def test_whitespace_only_string(self):
        """Test sanitizing whitespace-only string."""
        result = InputSanitizer.sanitize_string("   \t\n   ", "test_field")
        assert result == ""
    
    def test_empty_dict_sanitization(self):
        """Test sanitizing empty dictionary."""
        result = InputSanitizer.sanitize_dict({})
        assert result == {}
    
    def test_empty_list_sanitization(self):
        """Test sanitizing empty list."""
        result = InputSanitizer.sanitize_list([])
        assert result == []
    
    def test_error_context_with_none_values(self):
        """Test error context with None values."""
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation",
            user_input=None,
            system_state=None,
            stack_trace=None
        )
        
        assert context.component == "TestComponent"
        assert context.operation == "test_operation"
        assert context.user_input is None
        assert context.system_state is None
        assert context.stack_trace is None
    
    def test_error_response_serialization(self):
        """Test error response can be serialized."""
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation"
        )
        
        response = DetailedErrorResponse(
            error_type=ErrorType.DATA_VALIDATION.value,
            error_message="Test error",
            error_code=ErrorCode.INVALID_INVESTMENT_AMOUNT.value,
            suggested_action="Test action",
            error_severity=ErrorSeverity.MEDIUM,
            error_context=context
        )
        
        # Should be able to convert to dict
        response_dict = response.model_dump()
        assert isinstance(response_dict, dict)
        assert response_dict["error_type"] == ErrorType.DATA_VALIDATION.value
        assert response_dict["error_code"] == ErrorCode.INVALID_INVESTMENT_AMOUNT.value


class TestIntegration:
    """Integration tests for error handling system."""
    
    def test_full_error_handling_flow(self):
        """Test complete error handling flow."""
        handler = ErrorHandler()
        
        # Simulate a complex error scenario
        try:
            # This would be user input
            user_input = {"investment_amount": -1000}
            
            # This would trigger validation
            if user_input["investment_amount"] <= 0:
                raise ValueError("Investment amount must be positive")
                
        except Exception as e:
            response = handler.handle_exception(
                exception=e,
                component="PortfolioOptimizer",
                operation="validate_input",
                user_input=user_input,
                system_state={"stage": "input_validation"}
            )
            
            # Verify complete error response
            assert isinstance(response, DetailedErrorResponse)
            assert response.error_type == ErrorType.DATA_VALIDATION.value
            assert response.error_code == ErrorCode.DATA_OUT_OF_RANGE.value
            assert response.error_context.component == "PortfolioOptimizer"
            assert response.error_context.operation == "validate_input"
            assert response.error_context.user_input == user_input
            assert len(response.recovery_suggestions) > 0
    
    def test_cascading_error_handling(self):
        """Test handling cascading errors."""
        handler = ErrorHandler()
        
        # Simulate multiple related errors
        errors = [
            FileNotFoundError("Data file not found"),
            ValueError("Invalid data format"),
            ZeroDivisionError("Division by zero in calculations")
        ]
        
        responses = []
        for error in errors:
            response = handler.handle_exception(
                exception=error,
                component="DataProcessor",
                operation="process_data"
            )
            responses.append(response)
        
        # Verify all errors were handled
        assert len(responses) == 3
        assert all(isinstance(r, DetailedErrorResponse) for r in responses)
        
        # Verify error statistics
        stats = handler.get_error_statistics()
        assert stats["total_errors"] == 3
        assert len(stats["error_types"]) >= 2  # At least file and calculation errors


if __name__ == "__main__":
    pytest.main([__file__])