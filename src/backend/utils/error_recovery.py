"""
Error recovery and resilience mechanisms for the Financial Returns Optimizer.

This module provides automatic error recovery, fallback strategies,
and system resilience features.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from functools import wraps

from utils.error_handling import (
    ErrorHandler, ErrorType, ErrorCode, BaseFinancialError,
    AgentCommunicationError, CalculationError, DataValidationError
)

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    ALTERNATIVE_METHOD = "alternative_method"


class RecoveryAction:
    """Defines a recovery action for specific error conditions."""
    
    def __init__(
        self,
        error_types: List[ErrorType],
        error_codes: List[ErrorCode],
        strategy: RecoveryStrategy,
        max_attempts: int = 3,
        backoff_factor: float = 2.0,
        fallback_function: Optional[Callable] = None,
        recovery_data: Optional[Dict[str, Any]] = None
    ):
        self.error_types = error_types
        self.error_codes = error_codes
        self.strategy = strategy
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.fallback_function = fallback_function
        self.recovery_data = recovery_data or {}


class ErrorRecoveryManager:
    """Manages error recovery strategies and execution."""
    
    def __init__(self):
        self.recovery_actions: List[RecoveryAction] = []
        self.error_handler = ErrorHandler()
        self._setup_default_recovery_actions()
    
    def _setup_default_recovery_actions(self):
        """Set up default recovery actions for common error scenarios."""
        
        # Retry for temporary communication errors
        self.add_recovery_action(RecoveryAction(
            error_types=[ErrorType.AGENT_COMMUNICATION],
            error_codes=[ErrorCode.AGENT_EXECUTION_TIMEOUT, ErrorCode.AGENT_COMMUNICATION_FAILED],
            strategy=RecoveryStrategy.RETRY,
            max_attempts=3,
            backoff_factor=2.0
        ))
        
        # Fallback for calculation errors
        self.add_recovery_action(RecoveryAction(
            error_types=[ErrorType.CALCULATION_ERROR],
            error_codes=[ErrorCode.DIVISION_BY_ZERO_ERROR, ErrorCode.MATHEMATICAL_OVERFLOW],
            strategy=RecoveryStrategy.ALTERNATIVE_METHOD,
            max_attempts=2
        ))
        
        # Skip for non-critical data validation errors
        self.add_recovery_action(RecoveryAction(
            error_types=[ErrorType.DATA_VALIDATION],
            error_codes=[ErrorCode.DATA_OUT_OF_RANGE],
            strategy=RecoveryStrategy.FALLBACK,
            max_attempts=1
        ))
    
    def add_recovery_action(self, action: RecoveryAction):
        """Add a recovery action to the manager."""
        self.recovery_actions.append(action)
    
    def execute_with_recovery(
        self,
        function: Callable,
        *args,
        component: str = "Unknown",
        operation: str = "unknown",
        **kwargs
    ) -> Any:
        """
        Execute a function with automatic error recovery.
        
        Args:
            function: Function to execute
            *args: Function arguments
            component: Component name for error context
            operation: Operation name for error context
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or recovery result
            
        Raises:
            BaseFinancialError: If all recovery attempts fail
        """
        last_error = None
        
        for attempt in range(1, 4):  # Default max attempts
            try:
                return function(*args, **kwargs)
                
            except BaseFinancialError as e:
                last_error = e
                recovery_action = self._find_recovery_action(e.error_type, e.error_code)
                
                if recovery_action is None:
                    # No recovery action found, re-raise the error
                    logger.debug(f"No recovery action found for {e.error_type} - {e.error_code}")
                    raise e
                
                logger.debug(f"Found recovery action: {recovery_action.strategy} for {e.error_type} - {e.error_code}")
                
                if attempt >= recovery_action.max_attempts:
                    # Max attempts reached
                    logger.error(f"Recovery failed after {attempt} attempts for {component}.{operation}")
                    raise e
                
                # Execute recovery strategy
                if recovery_action.strategy == RecoveryStrategy.RETRY:
                    if attempt >= recovery_action.max_attempts:
                        # Max attempts reached for retry
                        logger.error(f"Recovery failed after {attempt} attempts for {component}.{operation}")
                        raise e
                    
                    # Wait before retry
                    wait_time = recovery_action.backoff_factor ** (attempt - 1)
                    logger.info(f"Retrying {component}.{operation} in {wait_time} seconds (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    continue
                else:
                    # Execute non-retry recovery strategy
                    recovery_result = self._execute_recovery_strategy(
                        recovery_action, e, function, args, kwargs, component, operation, attempt
                    )
                    
                    if recovery_result is not None:
                        return recovery_result
                    else:
                        # Recovery strategy didn't return a result, re-raise error
                        raise e
            
            except Exception as e:
                # Handle non-financial errors
                last_error = e
                error_response = self.error_handler.handle_exception(
                    exception=e,
                    component=component,
                    operation=operation,
                    user_input=kwargs.get('user_input'),
                    system_state=kwargs.get('system_state')
                )
                
                # Convert to financial error and retry
                financial_error = BaseFinancialError(
                    error_type=ErrorType(error_response.error_type),
                    error_code=ErrorCode(error_response.error_code),
                    message=error_response.error_message,
                    context=error_response.error_context,
                    severity=error_response.error_severity
                )
                
                recovery_action = self._find_recovery_action(financial_error.error_type, financial_error.error_code)
                
                if recovery_action is None or attempt >= recovery_action.max_attempts:
                    raise financial_error
                
                if recovery_action.strategy == RecoveryStrategy.RETRY:
                    wait_time = recovery_action.backoff_factor ** (attempt - 1)
                    logger.info(f"Retrying {component}.{operation} in {wait_time} seconds (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    continue
        
        # If we get here, all recovery attempts failed
        if last_error:
            raise last_error
        else:
            raise BaseFinancialError(
                error_type=ErrorType.SYSTEM_ERROR,
                error_code=ErrorCode.SYSTEM_RESOURCE_ERROR,
                message="Unknown error occurred during recovery"
            )
    
    def _find_recovery_action(self, error_type: ErrorType, error_code: ErrorCode) -> Optional[RecoveryAction]:
        """Find the appropriate recovery action for an error."""
        # Search in reverse order to prioritize most recently added actions
        for action in reversed(self.recovery_actions):
            if error_type in action.error_types and error_code in action.error_codes:
                return action
        return None
    
    def _execute_recovery_strategy(
        self,
        action: RecoveryAction,
        error: BaseFinancialError,
        function: Callable,
        args: tuple,
        kwargs: dict,
        component: str,
        operation: str,
        attempt: int
    ) -> Optional[Any]:
        """Execute the recovery strategy."""
        
        if action.strategy == RecoveryStrategy.FALLBACK:
            if action.fallback_function:
                logger.info(f"Using fallback function for {component}.{operation}")
                try:
                    return action.fallback_function(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback function failed: {fallback_error}")
                    raise error
            else:
                # Return default fallback data
                return action.recovery_data.get('default_result')
        
        elif action.strategy == RecoveryStrategy.SKIP:
            logger.info(f"Skipping operation {component}.{operation} due to error")
            return action.recovery_data.get('skip_result')
        
        elif action.strategy == RecoveryStrategy.ABORT:
            logger.error(f"Aborting operation {component}.{operation}")
            raise error
        
        elif action.strategy == RecoveryStrategy.ALTERNATIVE_METHOD:
            # Try alternative methods if available
            alternative_methods = action.recovery_data.get('alternative_methods', [])
            
            for alt_method in alternative_methods:
                try:
                    logger.info(f"Trying alternative method {alt_method.__name__} for {component}.{operation}")
                    return alt_method(*args, **kwargs)
                except Exception as alt_error:
                    logger.warning(f"Alternative method {alt_method.__name__} failed: {alt_error}")
                    continue
            
            # All alternative methods failed
            logger.error(f"All alternative methods failed for {component}.{operation}")
            raise error
        
        return None


# Global recovery manager instance
recovery_manager = ErrorRecoveryManager()


def with_recovery(component: str, operation: str, recovery_actions: Optional[List[RecoveryAction]] = None):
    """
    Decorator for automatic error recovery.
    
    Args:
        component: Component name
        operation: Operation name
        recovery_actions: Custom recovery actions for this function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Add custom recovery actions temporarily
            original_actions = recovery_manager.recovery_actions.copy()
            
            if recovery_actions:
                for action in recovery_actions:
                    recovery_manager.add_recovery_action(action)
            
            try:
                return recovery_manager.execute_with_recovery(
                    func, *args, component=component, operation=operation, **kwargs
                )
            finally:
                # Restore original recovery actions
                recovery_manager.recovery_actions = original_actions
        
        return wrapper
    return decorator