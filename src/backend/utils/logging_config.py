"""
Comprehensive logging and monitoring configuration for Financial Returns Optimizer.

This module provides structured logging, performance monitoring, data quality monitoring,
and error tracking capabilities throughout the system.
"""

import logging
import logging.handlers
import json
import time
import functools
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from contextlib import contextmanager

from config import LOGS_DIR, LOG_LEVEL, LOG_FORMAT


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ComponentType(Enum):
    """System component types for structured logging."""
    DATA_LOADER = "data_loader"
    DATA_CLEANING_AGENT = "data_cleaning_agent"
    ASSET_PREDICTOR_AGENT = "asset_predictor_agent"
    PORTFOLIO_ALLOCATOR_AGENT = "portfolio_allocator_agent"
    SIMULATION_AGENT = "simulation_agent"
    REBALANCING_AGENT = "rebalancing_agent"
    ORCHESTRATOR = "orchestrator"
    OUTPUT_GENERATOR = "output_generator"
    RISK_METRICS = "risk_metrics"
    VALIDATION = "validation"


@dataclass
class PerformanceMetric:
    """Performance monitoring data structure."""
    component: str
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    data_size: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class DataQualityMetric:
    """Data quality monitoring data structure."""
    component: str
    dataset_name: str
    total_records: int
    missing_values: int
    outliers_detected: int
    validation_errors: int
    data_completeness_percent: float
    timestamp: datetime
    quality_score: float  # 0-100 scale


@dataclass
class ErrorEvent:
    """Error tracking data structure."""
    component: str
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: datetime
    severity: str
    context: Dict[str, Any]
    user_input: Optional[Dict[str, Any]] = None


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record):
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "component": getattr(record, 'component', 'unknown'),
            "operation": getattr(record, 'operation', 'unknown'),
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add performance metrics if available
        if hasattr(record, 'performance_metric') and record.performance_metric is not None:
            try:
                log_entry['performance'] = asdict(record.performance_metric)
            except (TypeError, AttributeError):
                log_entry['performance'] = str(record.performance_metric)
        
        # Add data quality metrics if available
        if hasattr(record, 'data_quality_metric') and record.data_quality_metric is not None:
            try:
                log_entry['data_quality'] = asdict(record.data_quality_metric)
            except (TypeError, AttributeError):
                log_entry['data_quality'] = str(record.data_quality_metric)
        
        # Add error details if available
        if hasattr(record, 'error_event') and record.error_event is not None:
            try:
                log_entry['error'] = asdict(record.error_event)
            except (TypeError, AttributeError):
                log_entry['error'] = str(record.error_event)
        
        # Add custom context if available
        if hasattr(record, 'context'):
            log_entry['context'] = record.context
        
        return json.dumps(log_entry, default=str)


class LoggingManager:
    """Centralized logging manager for the Financial Returns Optimizer."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize logging manager."""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.performance_metrics: List[PerformanceMetric] = []
        self.data_quality_metrics: List[DataQualityMetric] = []
        self.error_events: List[ErrorEvent] = []
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Set up all loggers with appropriate handlers and formatters."""
        # Ensure logs directory exists
        LOGS_DIR.mkdir(exist_ok=True)
        
        # Main application logger
        self.app_logger = self._create_logger(
            'financial_optimizer',
            LOGS_DIR / 'app.log',
            use_structured=True
        )
        
        # Performance monitoring logger
        self.performance_logger = self._create_logger(
            'performance',
            LOGS_DIR / 'performance.log',
            use_structured=True
        )
        
        # Data quality logger
        self.data_quality_logger = self._create_logger(
            'data_quality',
            LOGS_DIR / 'data_quality.log',
            use_structured=True
        )
        
        # Error tracking logger
        self.error_logger = self._create_logger(
            'errors',
            LOGS_DIR / 'errors.log',
            use_structured=True
        )
        
        # Agent-specific loggers
        self.agent_loggers = {}
        for component in ComponentType:
            self.agent_loggers[component.value] = self._create_logger(
                f'agent.{component.value}',
                LOGS_DIR / f'{component.value}.log'
            )
    
    def _create_logger(self, name: str, log_file: Path, use_structured: bool = False) -> logging.Logger:
        """Create a logger with file and console handlers."""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, LOG_LEVEL))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # Set formatters
        if use_structured:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(LOG_FORMAT)
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def get_logger(self, component: ComponentType) -> logging.Logger:
        """Get logger for specific component."""
        return self.agent_loggers.get(component.value, self.app_logger)
    
    def log_performance_metric(self, metric: PerformanceMetric):
        """Log performance metric."""
        self.performance_metrics.append(metric)
        
        # Create log record with performance data
        record = logging.LogRecord(
            name='performance',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg=f"Performance: {metric.component}.{metric.operation} took {metric.duration:.3f}s",
            args=(),
            exc_info=None
        )
        record.performance_metric = metric
        record.component = metric.component
        record.operation = metric.operation
        
        self.performance_logger.handle(record)
    
    def log_data_quality_metric(self, metric: DataQualityMetric):
        """Log data quality metric."""
        self.data_quality_metrics.append(metric)
        
        # Create log record with data quality data
        record = logging.LogRecord(
            name='data_quality',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg=f"Data Quality: {metric.dataset_name} - Score: {metric.quality_score:.1f}%",
            args=(),
            exc_info=None
        )
        record.data_quality_metric = metric
        record.component = metric.component
        record.operation = 'data_quality_check'
        
        self.data_quality_logger.handle(record)
    
    def log_error_event(self, error: ErrorEvent):
        """Log error event."""
        self.error_events.append(error)
        
        # Create log record with error data
        level = getattr(logging, error.severity.upper(), logging.ERROR)
        record = logging.LogRecord(
            name='errors',
            level=level,
            pathname='',
            lineno=0,
            msg=f"Error in {error.component}: {error.error_message}",
            args=(),
            exc_info=None
        )
        record.error_event = error
        record.component = error.component
        record.operation = 'error_handling'
        
        self.error_logger.handle(record)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if not self.performance_metrics:
            return {"message": "No performance metrics available"}
        
        metrics_by_component = {}
        for metric in self.performance_metrics:
            if metric.component not in metrics_by_component:
                metrics_by_component[metric.component] = []
            metrics_by_component[metric.component].append(metric)
        
        summary = {}
        for component, metrics in metrics_by_component.items():
            durations = [m.duration for m in metrics]
            summary[component] = {
                "total_operations": len(metrics),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "success_rate": sum(1 for m in metrics if m.success) / len(metrics) * 100
            }
        
        return summary
    
    def get_data_quality_summary(self) -> Dict[str, Any]:
        """Get data quality metrics summary."""
        if not self.data_quality_metrics:
            return {"message": "No data quality metrics available"}
        
        latest_metrics = {}
        for metric in self.data_quality_metrics:
            key = f"{metric.component}_{metric.dataset_name}"
            if key not in latest_metrics or metric.timestamp > latest_metrics[key].timestamp:
                latest_metrics[key] = metric
        
        summary = {
            "datasets_monitored": len(latest_metrics),
            "average_quality_score": sum(m.quality_score for m in latest_metrics.values()) / len(latest_metrics),
            "datasets": {
                key: {
                    "quality_score": metric.quality_score,
                    "completeness": metric.data_completeness_percent,
                    "total_records": metric.total_records,
                    "issues": {
                        "missing_values": metric.missing_values,
                        "outliers": metric.outliers_detected,
                        "validation_errors": metric.validation_errors
                    }
                }
                for key, metric in latest_metrics.items()
            }
        }
        
        return summary
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error events summary."""
        if not self.error_events:
            return {"message": "No error events recorded"}
        
        errors_by_component = {}
        errors_by_type = {}
        
        for error in self.error_events:
            # Group by component
            if error.component not in errors_by_component:
                errors_by_component[error.component] = 0
            errors_by_component[error.component] += 1
            
            # Group by error type
            if error.error_type not in errors_by_type:
                errors_by_type[error.error_type] = 0
            errors_by_type[error.error_type] += 1
        
        return {
            "total_errors": len(self.error_events),
            "errors_by_component": errors_by_component,
            "errors_by_type": errors_by_type,
            "recent_errors": [
                {
                    "component": e.component,
                    "type": e.error_type,
                    "message": e.error_message,
                    "timestamp": e.timestamp.isoformat()
                }
                for e in sorted(self.error_events, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }


# Global logging manager instance
logging_manager = LoggingManager()


def performance_monitor(component: ComponentType, operation: str = None):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                
                metric = PerformanceMetric(
                    component=component.value,
                    operation=op_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    success=True
                )
                
                logging_manager.log_performance_metric(metric)
                return result
                
            except Exception as e:
                end_time = time.time()
                
                metric = PerformanceMetric(
                    component=component.value,
                    operation=op_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    success=False,
                    error_message=str(e)
                )
                
                logging_manager.log_performance_metric(metric)
                raise
        
        return wrapper
    return decorator


def error_tracker(component: ComponentType, error_type: str = None):
    """Decorator for tracking errors."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_event = ErrorEvent(
                    component=component.value,
                    error_type=error_type or type(e).__name__,
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                    timestamp=datetime.now(),
                    severity="ERROR",
                    context={
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                )
                
                logging_manager.log_error_event(error_event)
                raise
        
        return wrapper
    return decorator


@contextmanager
def operation_context(component: ComponentType, operation: str, context: Dict[str, Any] = None):
    """Context manager for logging operations with structured context."""
    logger = logging_manager.get_logger(component)
    start_time = time.time()
    
    # Create log record with context
    record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname='',
        lineno=0,
        msg=f"Starting {operation}",
        args=(),
        exc_info=None
    )
    record.component = component.value
    record.operation = operation
    record.context = context or {}
    
    logger.handle(record)
    
    try:
        yield logger
        
        # Log successful completion
        duration = time.time() - start_time
        record = logging.LogRecord(
            name=logger.name,
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg=f"Completed {operation} in {duration:.3f}s",
            args=(),
            exc_info=None
        )
        record.component = component.value
        record.operation = operation
        record.context = context or {}
        
        logger.handle(record)
        
    except Exception as e:
        # Log error
        duration = time.time() - start_time
        record = logging.LogRecord(
            name=logger.name,
            level=logging.ERROR,
            pathname='',
            lineno=0,
            msg=f"Failed {operation} after {duration:.3f}s: {str(e)}",
            args=(),
            exc_info=None
        )
        record.component = component.value
        record.operation = operation
        record.context = context or {}
        
        logger.handle(record)
        raise


def log_data_quality(component: ComponentType, dataset_name: str, 
                    total_records: int, missing_values: int = 0, 
                    outliers_detected: int = 0, validation_errors: int = 0):
    """Log data quality metrics."""
    completeness = ((total_records - missing_values) / total_records * 100) if total_records > 0 else 0
    
    # Calculate quality score (0-100)
    quality_score = 100
    if total_records > 0:
        quality_score -= (missing_values / total_records) * 30  # Missing values penalty
        quality_score -= (outliers_detected / total_records) * 20  # Outliers penalty
        quality_score -= (validation_errors / total_records) * 50  # Validation errors penalty
        quality_score = max(0, quality_score)
    
    metric = DataQualityMetric(
        component=component.value,
        dataset_name=dataset_name,
        total_records=total_records,
        missing_values=missing_values,
        outliers_detected=outliers_detected,
        validation_errors=validation_errors,
        data_completeness_percent=completeness,
        timestamp=datetime.now(),
        quality_score=quality_score
    )
    
    logging_manager.log_data_quality_metric(metric)