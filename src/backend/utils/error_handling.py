"""
Comprehensive error handling system for the Financial Returns Optimizer.

This module provides error classification, user-friendly error messages,
input sanitization, and security validation for the multi-agent system.
"""

import logging
import traceback
import re
from typing import Dict, Any, List, Optional, Union, Type
from enum import Enum
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from models.data_models import ErrorResponse

# Configure logging
logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of error types in the system."""
    
    # Data validation errors
    DATA_VALIDATION = "data_validation"
    INPUT_VALIDATION = "input_validation"
    FILE_VALIDATION = "file_validation"
    
    # Calculation errors
    CALCULATION_ERROR = "calculation_error"
    MATHEMATICAL_ERROR = "mathematical_error"
    NUMERICAL_OVERFLOW = "numerical_overflow"
    DIVISION_BY_ZERO = "division_by_zero"
    
    # Agent communication errors
    AGENT_COMMUNICATION = "agent_communication"
    AGENT_TIMEOUT = "agent_timeout"
    AGENT_INITIALIZATION = "agent_initialization"
    PIPELINE_ERROR = "pipeline_error"
    
    # System errors
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_ERROR = "dependency_error"
    
    # Security errors
    SECURITY_VIOLATION = "security_violation"
    INPUT_SANITIZATION = "input_sanitization"
    FILE_ACCESS_DENIED = "file_access_denied"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCode(Enum):
    """Standardized error codes for the system."""
    
    # Data validation errors (1000-1999)
    INVALID_INVESTMENT_AMOUNT = 1001
    INVALID_TENURE_YEARS = 1002
    INVALID_RISK_PROFILE = 1003
    INVALID_RETURN_EXPECTATION = 1004
    MISSING_REQUIRED_FIELD = 1005
    INVALID_DATA_FORMAT = 1006
    DATA_OUT_OF_RANGE = 1007
    CORRUPTED_DATA_FILE = 1008
    INSUFFICIENT_DATA = 1009
    
    # Calculation errors (2000-2999)
    PORTFOLIO_ALLOCATION_ERROR = 2001
    PROJECTION_CALCULATION_ERROR = 2002
    RISK_METRICS_ERROR = 2003
    RETURN_PREDICTION_ERROR = 2004
    REBALANCING_ERROR = 2005
    NEGATIVE_PORTFOLIO_VALUE = 2006
    ALLOCATION_CONSTRAINT_VIOLATION = 2007
    MATHEMATICAL_OVERFLOW = 2008
    DIVISION_BY_ZERO_ERROR = 2009
    
    # Agent communication errors (3000-3999)
    AGENT_INITIALIZATION_FAILED = 3001
    AGENT_EXECUTION_TIMEOUT = 3002
    AGENT_COMMUNICATION_FAILED = 3003
    PIPELINE_STAGE_FAILED = 3004
    DATA_FLOW_ERROR = 3005
    ORCHESTRATION_ERROR = 3006
    
    # System errors (4000-4999)
    FILE_NOT_FOUND = 4001
    FILE_ACCESS_ERROR = 4002
    CONFIGURATION_ERROR = 4003
    DEPENDENCY_MISSING = 4004
    MEMORY_ERROR = 4005
    SYSTEM_RESOURCE_ERROR = 4006
    
    # Security errors (5000-5999)
    INVALID_FILE_PATH = 5001
    MALICIOUS_INPUT_DETECTED = 5002
    ACCESS_DENIED = 5003
    INPUT_TOO_LARGE = 5004
    UNSAFE_OPERATION = 5005


class ErrorContext(BaseModel):
    """Context information for errors."""
    
    component: str = Field(description="Component where error occurred")
    operation: str = Field(description="Operation being performed")
    user_input: Optional[Dict[str, Any]] = Field(default=None, description="User input that caused error")
    system_state: Optional[Dict[str, Any]] = Field(default=None, description="System state when error occurred")
    stack_trace: Optional[str] = Field(default=None, description="Stack trace for debugging")


class DetailedErrorResponse(ErrorResponse):
    """Extended error response with additional context."""
    
    error_severity: ErrorSeverity = Field(description="Severity level of the error")
    error_context: Optional[ErrorContext] = Field(default=None, description="Additional error context")
    recovery_suggestions: List[str] = Field(default_factory=list, description="Specific recovery suggestions")
    related_errors: List[str] = Field(default_factory=list, description="Related error messages")


class InputSanitizer:
    """Input sanitization and security validation."""
    
    # Security patterns to detect
    MALICIOUS_PATTERNS = [
        r'<script.*?>.*?</script>',  # Script tags
        r'javascript:',              # JavaScript URLs
        r'on\w+\s*=',               # Event handlers
        r'eval\s*\(',               # eval() calls
        r'exec\s*\(',               # exec() calls
        r'import\s+',               # import statements
        r'__.*__',                  # Python dunder methods
        r'\.\./',                   # Path traversal
        r'[;&|`$]',                 # Shell injection characters
    ]
    
    # File path validation
    ALLOWED_FILE_EXTENSIONS = {'.xls', '.xlsx', '.csv'}
    MAX_FILE_SIZE_MB = 100
    
    # Input size limits
    MAX_STRING_LENGTH = 1000
    MAX_DICT_DEPTH = 10
    MAX_LIST_LENGTH = 1000
    
    @classmethod
    def sanitize_string(cls, value: str, field_name: str = "input") -> str:
        """
        Sanitize string input for security.
        
        Args:
            value: String to sanitize
            field_name: Name of the field being sanitized
            
        Returns:
            Sanitized string
            
        Raises:
            SecurityError: If malicious content is detected
        """
        if not isinstance(value, str):
            raise SecurityError(
                error_type=ErrorType.INPUT_SANITIZATION,
                error_code=ErrorCode.MALICIOUS_INPUT_DETECTED,
                message=f"Expected string for {field_name}, got {type(value).__name__}",
                context=ErrorContext(
                    component="InputSanitizer",
                    operation="sanitize_string",
                    user_input={"field": field_name, "type": type(value).__name__}
                )
            )
        
        # Check length
        if len(value) > cls.MAX_STRING_LENGTH:
            raise SecurityError(
                error_type=ErrorType.INPUT_SANITIZATION,
                error_code=ErrorCode.INPUT_TOO_LARGE,
                message=f"Input too long for {field_name}: {len(value)} > {cls.MAX_STRING_LENGTH}",
                context=ErrorContext(
                    component="InputSanitizer",
                    operation="sanitize_string",
                    user_input={"field": field_name, "length": len(value)}
                )
            )
        
        # Check for malicious patterns
        for pattern in cls.MALICIOUS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise SecurityError(
                    error_type=ErrorType.SECURITY_VIOLATION,
                    error_code=ErrorCode.MALICIOUS_INPUT_DETECTED,
                    message=f"Potentially malicious content detected in {field_name}",
                    context=ErrorContext(
                        component="InputSanitizer",
                        operation="sanitize_string",
                        user_input={"field": field_name, "pattern": pattern}
                    )
                )
        
        # Basic sanitization
        sanitized = value.strip()
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        return sanitized
    
    @classmethod
    def validate_file_path(cls, file_path: str) -> str:
        """
        Validate and sanitize file path.
        
        Args:
            file_path: File path to validate
            
        Returns:
            Validated file path
            
        Raises:
            SecurityError: If file path is invalid or unsafe
        """
        if not isinstance(file_path, str):
            raise SecurityError(
                error_type=ErrorType.FILE_VALIDATION,
                error_code=ErrorCode.INVALID_FILE_PATH,
                message=f"File path must be string, got {type(file_path).__name__}",
                context=ErrorContext(
                    component="InputSanitizer",
                    operation="validate_file_path",
                    user_input={"type": type(file_path).__name__}
                )
            )
        
        # Sanitize the path string first
        sanitized_path = cls.sanitize_string(file_path, "file_path")
        
        try:
            path = Path(sanitized_path).resolve()
        except Exception as e:
            raise SecurityError(
                error_type=ErrorType.FILE_VALIDATION,
                error_code=ErrorCode.INVALID_FILE_PATH,
                message=f"Invalid file path format: {str(e)}",
                context=ErrorContext(
                    component="InputSanitizer",
                    operation="validate_file_path",
                    user_input={"file_path": sanitized_path}
                )
            )
        
        # Check if file exists
        if not path.exists():
            raise DataValidationError(
                error_type=ErrorType.FILE_VALIDATION,
                error_code=ErrorCode.FILE_NOT_FOUND,
                message=f"File not found: {path}",
                context=ErrorContext(
                    component="InputSanitizer",
                    operation="validate_file_path",
                    user_input={"file_path": str(path)}
                )
            )
        
        # Check file extension
        if path.suffix.lower() not in cls.ALLOWED_FILE_EXTENSIONS:
            raise SecurityError(
                error_type=ErrorType.FILE_VALIDATION,
                error_code=ErrorCode.INVALID_FILE_PATH,
                message=f"File extension not allowed: {path.suffix}. Allowed: {cls.ALLOWED_FILE_EXTENSIONS}",
                context=ErrorContext(
                    component="InputSanitizer",
                    operation="validate_file_path",
                    user_input={"file_path": str(path), "extension": path.suffix}
                )
            )
        
        # Check file size
        try:
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > cls.MAX_FILE_SIZE_MB:
                raise SecurityError(
                    error_type=ErrorType.FILE_VALIDATION,
                    error_code=ErrorCode.INPUT_TOO_LARGE,
                    message=f"File too large: {file_size_mb:.1f}MB > {cls.MAX_FILE_SIZE_MB}MB",
                    context=ErrorContext(
                        component="InputSanitizer",
                        operation="validate_file_path",
                        user_input={"file_path": str(path), "size_mb": file_size_mb}
                    )
                )
        except OSError as e:
            raise SystemError(
                error_type=ErrorType.FILE_VALIDATION,
                error_code=ErrorCode.FILE_ACCESS_ERROR,
                message=f"Cannot access file: {str(e)}",
                context=ErrorContext(
                    component="InputSanitizer",
                    operation="validate_file_path",
                    user_input={"file_path": str(path)}
                )
            )
        
        return str(path)
    
    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any], max_depth: int = None) -> Dict[str, Any]:
        """
        Sanitize dictionary input recursively.
        
        Args:
            data: Dictionary to sanitize
            max_depth: Maximum recursion depth
            
        Returns:
            Sanitized dictionary
            
        Raises:
            SecurityError: If input is unsafe
        """
        if max_depth is None:
            max_depth = cls.MAX_DICT_DEPTH
        
        if max_depth <= 0:
            raise SecurityError(
                error_type=ErrorType.INPUT_SANITIZATION,
                error_code=ErrorCode.INPUT_TOO_LARGE,
                message="Dictionary nesting too deep",
                context=ErrorContext(
                    component="InputSanitizer",
                    operation="sanitize_dict"
                )
            )
        
        if not isinstance(data, dict):
            raise SecurityError(
                error_type=ErrorType.INPUT_SANITIZATION,
                error_code=ErrorCode.MALICIOUS_INPUT_DETECTED,
                message=f"Expected dict, got {type(data).__name__}",
                context=ErrorContext(
                    component="InputSanitizer",
                    operation="sanitize_dict",
                    user_input={"type": type(data).__name__}
                )
            )
        
        if len(data) > cls.MAX_LIST_LENGTH:
            raise SecurityError(
                error_type=ErrorType.INPUT_SANITIZATION,
                error_code=ErrorCode.INPUT_TOO_LARGE,
                message=f"Dictionary too large: {len(data)} > {cls.MAX_LIST_LENGTH}",
                context=ErrorContext(
                    component="InputSanitizer",
                    operation="sanitize_dict",
                    user_input={"size": len(data)}
                )
            )
        
        sanitized = {}
        for key, value in data.items():
            # Sanitize key
            if isinstance(key, str):
                sanitized_key = cls.sanitize_string(key, f"dict_key_{key}")
            else:
                sanitized_key = key
            
            # Sanitize value
            if isinstance(value, str):
                sanitized_value = cls.sanitize_string(value, f"dict_value_{key}")
            elif isinstance(value, dict):
                sanitized_value = cls.sanitize_dict(value, max_depth - 1)
            elif isinstance(value, list):
                sanitized_value = cls.sanitize_list(value, max_depth - 1)
            else:
                sanitized_value = value
            
            sanitized[sanitized_key] = sanitized_value
        
        return sanitized
    
    @classmethod
    def sanitize_list(cls, data: List[Any], max_depth: int = None) -> List[Any]:
        """
        Sanitize list input recursively.
        
        Args:
            data: List to sanitize
            max_depth: Maximum recursion depth
            
        Returns:
            Sanitized list
            
        Raises:
            SecurityError: If input is unsafe
        """
        if max_depth is None:
            max_depth = cls.MAX_DICT_DEPTH
        
        if max_depth <= 0:
            raise SecurityError(
                error_type=ErrorType.INPUT_SANITIZATION,
                error_code=ErrorCode.INPUT_TOO_LARGE,
                message="List nesting too deep",
                context=ErrorContext(
                    component="InputSanitizer",
                    operation="sanitize_list"
                )
            )
        
        if not isinstance(data, list):
            raise SecurityError(
                error_type=ErrorType.INPUT_SANITIZATION,
                error_code=ErrorCode.MALICIOUS_INPUT_DETECTED,
                message=f"Expected list, got {type(data).__name__}",
                context=ErrorContext(
                    component="InputSanitizer",
                    operation="sanitize_list",
                    user_input={"type": type(data).__name__}
                )
            )
        
        if len(data) > cls.MAX_LIST_LENGTH:
            raise SecurityError(
                error_type=ErrorType.INPUT_SANITIZATION,
                error_code=ErrorCode.INPUT_TOO_LARGE,
                message=f"List too large: {len(data)} > {cls.MAX_LIST_LENGTH}",
                context=ErrorContext(
                    component="InputSanitizer",
                    operation="sanitize_list",
                    user_input={"size": len(data)}
                )
            )
        
        sanitized = []
        for i, item in enumerate(data):
            if isinstance(item, str):
                sanitized_item = cls.sanitize_string(item, f"list_item_{i}")
            elif isinstance(item, dict):
                sanitized_item = cls.sanitize_dict(item, max_depth - 1)
            elif isinstance(item, list):
                sanitized_item = cls.sanitize_list(item, max_depth - 1)
            else:
                sanitized_item = item
            
            sanitized.append(sanitized_item)
        
        return sanitized


class BaseFinancialError(Exception):
    """Base exception class for financial system errors."""
    
    def __init__(
        self,
        error_type: ErrorType,
        error_code: ErrorCode,
        message: str,
        context: Optional[ErrorContext] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_suggestions: Optional[List[str]] = None
    ):
        self.error_type = error_type
        self.error_code = error_code
        self.message = message
        self.context = context
        self.severity = severity
        self.recovery_suggestions = recovery_suggestions or []
        self.timestamp = datetime.now()
        
        super().__init__(self.message)
    
    def to_error_response(self) -> DetailedErrorResponse:
        """Convert exception to structured error response."""
        return DetailedErrorResponse(
            error_type=self.error_type.value,
            error_message=self.message,
            error_code=self.error_code.value,
            suggested_action=self._get_suggested_action(),
            timestamp=self.timestamp,
            error_severity=self.severity,
            error_context=self.context,
            recovery_suggestions=self.recovery_suggestions
        )
    
    def _get_suggested_action(self) -> str:
        """Get suggested action based on error type and code."""
        suggestions = {
            ErrorCode.INVALID_INVESTMENT_AMOUNT: "Please enter a valid investment amount between $1 and $1,000,000,000",
            ErrorCode.INVALID_TENURE_YEARS: "Please enter a tenure between 1 and 50 years",
            ErrorCode.INVALID_RISK_PROFILE: "Please select a valid risk profile: Low, Moderate, or High",
            ErrorCode.INVALID_RETURN_EXPECTATION: "Please enter a return expectation between 0% and 50%",
            ErrorCode.FILE_NOT_FOUND: "Please check that the data file exists and is accessible",
            ErrorCode.CORRUPTED_DATA_FILE: "Please verify the data file is not corrupted and try again",
            ErrorCode.PORTFOLIO_ALLOCATION_ERROR: "Please check your risk profile and try again",
            ErrorCode.AGENT_INITIALIZATION_FAILED: "Please restart the application and try again",
            ErrorCode.MALICIOUS_INPUT_DETECTED: "Please remove any special characters or scripts from your input",
        }
        
        return suggestions.get(self.error_code, "Please check your input and try again")


class DataValidationError(BaseFinancialError):
    """Exception for data validation errors."""
    
    def __init__(self, error_type: ErrorType, error_code: ErrorCode, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            error_type=error_type,
            error_code=error_code,
            message=message,
            context=context,
            severity=ErrorSeverity.MEDIUM,
            recovery_suggestions=[
                "Verify your input data is correct",
                "Check data format and ranges",
                "Ensure all required fields are provided"
            ]
        )


class CalculationError(BaseFinancialError):
    """Exception for calculation and mathematical errors."""
    
    def __init__(self, error_type: ErrorType, error_code: ErrorCode, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            error_type=error_type,
            error_code=error_code,
            message=message,
            context=context,
            severity=ErrorSeverity.HIGH,
            recovery_suggestions=[
                "Check input parameters for edge cases",
                "Verify data quality and completeness",
                "Try with different input values"
            ]
        )


class AgentCommunicationError(BaseFinancialError):
    """Exception for agent communication and pipeline errors."""
    
    def __init__(self, error_type: ErrorType, error_code: ErrorCode, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            error_type=error_type,
            error_code=error_code,
            message=message,
            context=context,
            severity=ErrorSeverity.HIGH,
            recovery_suggestions=[
                "Retry the operation",
                "Check system resources",
                "Restart the application if problem persists"
            ]
        )


class SecurityError(BaseFinancialError):
    """Exception for security violations and input sanitization errors."""
    
    def __init__(self, error_type: ErrorType, error_code: ErrorCode, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            error_type=error_type,
            error_code=error_code,
            message=message,
            context=context,
            severity=ErrorSeverity.CRITICAL,
            recovery_suggestions=[
                "Remove any special characters from input",
                "Use only allowed file types",
                "Ensure input is within size limits"
            ]
        )


class SystemError(BaseFinancialError):
    """Exception for system-level errors."""
    
    def __init__(self, error_type: ErrorType, error_code: ErrorCode, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            error_type=error_type,
            error_code=error_code,
            message=message,
            context=context,
            severity=ErrorSeverity.CRITICAL,
            recovery_suggestions=[
                "Check system resources",
                "Verify file permissions",
                "Contact system administrator if problem persists"
            ]
        )


class ErrorHandler:
    """Central error handling and recovery system."""
    
    def __init__(self):
        self.error_log: List[DetailedErrorResponse] = []
        self.error_counts: Dict[ErrorCode, int] = {}
    
    def handle_exception(
        self,
        exception: Exception,
        component: str,
        operation: str,
        user_input: Optional[Dict[str, Any]] = None,
        system_state: Optional[Dict[str, Any]] = None
    ) -> DetailedErrorResponse:
        """
        Handle any exception and convert to structured error response.
        
        Args:
            exception: Exception that occurred
            component: Component where error occurred
            operation: Operation being performed
            user_input: User input that caused error
            system_state: System state when error occurred
            
        Returns:
            Structured error response
        """
        # Create error context
        context = ErrorContext(
            component=component,
            operation=operation,
            user_input=user_input,
            system_state=system_state,
            stack_trace=traceback.format_exc()
        )
        
        # Handle known error types
        if isinstance(exception, BaseFinancialError):
            error_response = exception.to_error_response()
            error_response.error_context = context
        elif isinstance(exception, ValidationError):
            error_response = self._handle_pydantic_error(exception, context)
        elif isinstance(exception, ValueError):
            error_response = self._handle_value_error(exception, context)
        elif isinstance(exception, FileNotFoundError):
            error_response = self._handle_file_error(exception, context)
        elif isinstance(exception, PermissionError):
            error_response = self._handle_permission_error(exception, context)
        elif isinstance(exception, MemoryError):
            error_response = self._handle_memory_error(exception, context)
        elif isinstance(exception, TimeoutError):
            error_response = self._handle_timeout_error(exception, context)
        elif isinstance(exception, ZeroDivisionError):
            error_response = self._handle_division_error(exception, context)
        elif isinstance(exception, OverflowError):
            error_response = self._handle_overflow_error(exception, context)
        else:
            error_response = self._handle_unknown_error(exception, context)
        
        # Log the error
        logger.error(f"Error in {component}.{operation}: {error_response.error_message}")
        
        # Track error statistics
        error_code = ErrorCode(error_response.error_code)
        self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1
        
        # Store error for analysis
        self.error_log.append(error_response)
        
        return error_response
    
    def _handle_pydantic_error(self, exception: ValidationError, context: ErrorContext) -> DetailedErrorResponse:
        """Handle Pydantic validation errors."""
        error_details = []
        for error in exception.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            error_details.append(f"{field}: {message}")
        
        return DetailedErrorResponse(
            error_type=ErrorType.DATA_VALIDATION.value,
            error_message=f"Data validation failed: {'; '.join(error_details)}",
            error_code=ErrorCode.INVALID_DATA_FORMAT.value,
            suggested_action="Please check your input data and correct the validation errors",
            error_severity=ErrorSeverity.MEDIUM,
            error_context=context,
            recovery_suggestions=[
                "Check all required fields are provided",
                "Verify data types match expected formats",
                "Ensure numeric values are within valid ranges"
            ]
        )
    
    def _handle_value_error(self, exception: ValueError, context: ErrorContext) -> DetailedErrorResponse:
        """Handle ValueError exceptions."""
        return DetailedErrorResponse(
            error_type=ErrorType.DATA_VALIDATION.value,
            error_message=f"Invalid value: {str(exception)}",
            error_code=ErrorCode.DATA_OUT_OF_RANGE.value,
            suggested_action="Please check your input values are within valid ranges",
            error_severity=ErrorSeverity.MEDIUM,
            error_context=context,
            recovery_suggestions=[
                "Verify numeric values are positive where required",
                "Check percentage values are between 0 and 100",
                "Ensure dates are in valid format"
            ]
        )
    
    def _handle_file_error(self, exception: FileNotFoundError, context: ErrorContext) -> DetailedErrorResponse:
        """Handle file not found errors."""
        return DetailedErrorResponse(
            error_type=ErrorType.FILE_VALIDATION.value,
            error_message=f"File not found: {str(exception)}",
            error_code=ErrorCode.FILE_NOT_FOUND.value,
            suggested_action="Please check that the file exists and the path is correct",
            error_severity=ErrorSeverity.HIGH,
            error_context=context,
            recovery_suggestions=[
                "Verify the file path is correct",
                "Check file permissions",
                "Ensure the file has not been moved or deleted"
            ]
        )
    
    def _handle_permission_error(self, exception: PermissionError, context: ErrorContext) -> DetailedErrorResponse:
        """Handle permission errors."""
        return DetailedErrorResponse(
            error_type=ErrorType.SYSTEM_ERROR.value,
            error_message=f"Permission denied: {str(exception)}",
            error_code=ErrorCode.ACCESS_DENIED.value,
            suggested_action="Please check file permissions and access rights",
            error_severity=ErrorSeverity.HIGH,
            error_context=context,
            recovery_suggestions=[
                "Check file and directory permissions",
                "Run with appropriate user privileges",
                "Contact system administrator"
            ]
        )
    
    def _handle_memory_error(self, exception: MemoryError, context: ErrorContext) -> DetailedErrorResponse:
        """Handle memory errors."""
        return DetailedErrorResponse(
            error_type=ErrorType.SYSTEM_ERROR.value,
            error_message=f"Out of memory: {str(exception)}",
            error_code=ErrorCode.MEMORY_ERROR.value,
            suggested_action="Please reduce data size or increase available memory",
            error_severity=ErrorSeverity.CRITICAL,
            error_context=context,
            recovery_suggestions=[
                "Process smaller datasets",
                "Close other applications to free memory",
                "Increase system memory if possible"
            ]
        )
    
    def _handle_timeout_error(self, exception: TimeoutError, context: ErrorContext) -> DetailedErrorResponse:
        """Handle timeout errors."""
        return DetailedErrorResponse(
            error_type=ErrorType.AGENT_COMMUNICATION.value,
            error_message=f"Operation timed out: {str(exception)}",
            error_code=ErrorCode.AGENT_EXECUTION_TIMEOUT.value,
            suggested_action="Please try again or increase timeout settings",
            error_severity=ErrorSeverity.HIGH,
            error_context=context,
            recovery_suggestions=[
                "Retry the operation",
                "Reduce data complexity",
                "Check system performance"
            ]
        )
    
    def _handle_division_error(self, exception: ZeroDivisionError, context: ErrorContext) -> DetailedErrorResponse:
        """Handle division by zero errors."""
        return DetailedErrorResponse(
            error_type=ErrorType.CALCULATION_ERROR.value,
            error_message=f"Division by zero: {str(exception)}",
            error_code=ErrorCode.DIVISION_BY_ZERO_ERROR.value,
            suggested_action="Please check input data for zero values in denominators",
            error_severity=ErrorSeverity.HIGH,
            error_context=context,
            recovery_suggestions=[
                "Check for zero values in return calculations",
                "Verify portfolio values are not zero",
                "Use alternative calculation methods"
            ]
        )
    
    def _handle_overflow_error(self, exception: OverflowError, context: ErrorContext) -> DetailedErrorResponse:
        """Handle numerical overflow errors."""
        return DetailedErrorResponse(
            error_type=ErrorType.CALCULATION_ERROR.value,
            error_message=f"Numerical overflow: {str(exception)}",
            error_code=ErrorCode.MATHEMATICAL_OVERFLOW.value,
            suggested_action="Please reduce input values to prevent numerical overflow",
            error_severity=ErrorSeverity.HIGH,
            error_context=context,
            recovery_suggestions=[
                "Use smaller investment amounts",
                "Reduce projection timeframes",
                "Check for unrealistic return expectations"
            ]
        )
    
    def _handle_unknown_error(self, exception: Exception, context: ErrorContext) -> DetailedErrorResponse:
        """Handle unknown/unexpected errors."""
        return DetailedErrorResponse(
            error_type=ErrorType.SYSTEM_ERROR.value,
            error_message=f"Unexpected error: {str(exception)}",
            error_code=ErrorCode.SYSTEM_RESOURCE_ERROR.value,
            suggested_action="Please try again or contact support if the problem persists",
            error_severity=ErrorSeverity.CRITICAL,
            error_context=context,
            recovery_suggestions=[
                "Retry the operation",
                "Check system logs for more details",
                "Contact technical support"
            ]
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        total_errors = len(self.error_log)
        if total_errors == 0:
            return {"total_errors": 0}
        
        # Count by error type
        type_counts = {}
        severity_counts = {}
        recent_errors = []
        
        for error in self.error_log[-100:]:  # Last 100 errors
            error_type = error.error_type
            type_counts[error_type] = type_counts.get(error_type, 0) + 1
            
            severity = error.error_severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            if (datetime.now() - error.timestamp).total_seconds() < 3600:  # Last hour
                recent_errors.append({
                    "type": error_type,
                    "message": error.error_message,
                    "timestamp": error.timestamp.isoformat()
                })
        
        return {
            "total_errors": total_errors,
            "error_types": type_counts,
            "error_severities": severity_counts,
            "recent_errors": recent_errors,
            "most_common_errors": dict(sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        }
    
    def clear_error_log(self):
        """Clear the error log (for testing or maintenance)."""
        self.error_log.clear()
        self.error_counts.clear()


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(component: str, operation: str):
    """
    Decorator for automatic error handling.
    
    Args:
        component: Component name
        operation: Operation name
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_response = error_handler.handle_exception(
                    exception=e,
                    component=component,
                    operation=operation,
                    user_input=kwargs.get('user_input'),
                    system_state=kwargs.get('system_state')
                )
                raise BaseFinancialError(
                    error_type=ErrorType(error_response.error_type),
                    error_code=ErrorCode(error_response.error_code),
                    message=error_response.error_message,
                    context=error_response.error_context,
                    severity=error_response.error_severity,
                    recovery_suggestions=error_response.recovery_suggestions
                )
        return wrapper
    return decorator