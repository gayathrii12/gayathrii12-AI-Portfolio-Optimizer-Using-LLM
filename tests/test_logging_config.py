"""
Tests for comprehensive logging and monitoring system.

This module tests the logging configuration, performance monitoring,
data quality tracking, and error handling capabilities.
"""

import pytest
import json
import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from utils.logging_config import (
    LoggingManager,
    ComponentType,
    PerformanceMetric,
    DataQualityMetric,
    ErrorEvent,
    StructuredFormatter,
    performance_monitor,
    error_tracker,
    operation_context,
    log_data_quality,
    logging_manager
)


class TestLoggingManager:
    """Test cases for LoggingManager class."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        # Clear singleton instance
        LoggingManager._instance = None
        # Clear any existing metrics
        if hasattr(logging_manager, 'performance_metrics'):
            logging_manager.performance_metrics.clear()
        if hasattr(logging_manager, 'data_quality_metrics'):
            logging_manager.data_quality_metrics.clear()
        if hasattr(logging_manager, 'error_events'):
            logging_manager.error_events.clear()
    
    def test_singleton_pattern(self):
        """Test that LoggingManager follows singleton pattern."""
        manager1 = LoggingManager()
        manager2 = LoggingManager()
        assert manager1 is manager2
    
    def test_get_logger(self):
        """Test getting component-specific loggers."""
        manager = LoggingManager()
        logger = manager.get_logger(ComponentType.DATA_CLEANING_AGENT)
        assert logger.name == 'agent.data_cleaning_agent'
    
    def test_log_performance_metric(self):
        """Test logging performance metrics."""
        manager = LoggingManager()
        
        metric = PerformanceMetric(
            component="test_component",
            operation="test_operation",
            start_time=time.time(),
            end_time=time.time() + 1.5,
            duration=1.5,
            success=True
        )
        
        # Should not raise exception
        manager.log_performance_metric(metric)
        assert len(manager.performance_metrics) > 0
        assert manager.performance_metrics[-1].duration == 1.5
    
    def test_log_data_quality_metric(self):
        """Test logging data quality metrics."""
        manager = LoggingManager()
        
        metric = DataQualityMetric(
            component="test_component",
            dataset_name="test_dataset",
            total_records=1000,
            missing_values=50,
            outliers_detected=10,
            validation_errors=5,
            data_completeness_percent=95.0,
            timestamp=datetime.now(),
            quality_score=85.0
        )
        
        manager.log_data_quality_metric(metric)
        assert len(manager.data_quality_metrics) > 0
        assert manager.data_quality_metrics[-1].quality_score == 85.0
    
    def test_log_error_event(self):
        """Test logging error events."""
        manager = LoggingManager()
        
        error = ErrorEvent(
            component="test_component",
            error_type="TestError",
            error_message="Test error message",
            stack_trace="Test stack trace",
            timestamp=datetime.now(),
            severity="ERROR",
            context={"test": "context"}
        )
        
        manager.log_error_event(error)
        assert len(manager.error_events) > 0
        assert manager.error_events[-1].error_type == "TestError"
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        manager = LoggingManager()
        
        # Add some test metrics
        for i in range(3):
            metric = PerformanceMetric(
                component="test_component",
                operation="test_operation",
                start_time=time.time(),
                end_time=time.time() + (i + 1),
                duration=i + 1,
                success=True
            )
            manager.log_performance_metric(metric)
        
        summary = manager.get_performance_summary()
        assert "test_component" in summary
        assert summary["test_component"]["total_operations"] >= 3
        # Check that average is reasonable (should be around 2.0)
        assert 1.5 <= summary["test_component"]["avg_duration"] <= 2.5
    
    def test_get_data_quality_summary(self):
        """Test getting data quality summary."""
        manager = LoggingManager()
        
        metric = DataQualityMetric(
            component="test_component",
            dataset_name="test_dataset",
            total_records=1000,
            missing_values=50,
            outliers_detected=10,
            validation_errors=5,
            data_completeness_percent=95.0,
            timestamp=datetime.now(),
            quality_score=85.0
        )
        
        manager.log_data_quality_metric(metric)
        summary = manager.get_data_quality_summary()
        
        assert summary["datasets_monitored"] == 1
        assert summary["average_quality_score"] == 85.0
        assert "test_component_test_dataset" in summary["datasets"]
    
    def test_get_error_summary(self):
        """Test getting error summary."""
        manager = LoggingManager()
        
        error = ErrorEvent(
            component="test_component",
            error_type="TestError",
            error_message="Test error message",
            stack_trace="Test stack trace",
            timestamp=datetime.now(),
            severity="ERROR",
            context={"test": "context"}
        )
        
        manager.log_error_event(error)
        summary = manager.get_error_summary()
        
        assert summary["total_errors"] >= 1
        assert "test_component" in summary["errors_by_component"]
        assert "TestError" in summary["errors_by_type"]


class TestStructuredFormatter:
    """Test cases for StructuredFormatter class."""
    
    def test_format_basic_record(self):
        """Test formatting basic log record."""
        formatter = StructuredFormatter()
        
        # Create a mock log record
        record = MagicMock()
        record.created = time.time()
        record.levelname = "INFO"
        record.getMessage.return_value = "Test message"
        record.module = "test_module"
        record.funcName = "test_function"
        record.lineno = 123
        record.component = "test_component"
        record.operation = "test_operation"
        
        formatted = formatter.format(record)
        log_entry = json.loads(formatted)
        
        assert log_entry["level"] == "INFO"
        assert log_entry["message"] == "Test message"
        assert log_entry["component"] == "test_component"
        assert log_entry["operation"] == "test_operation"
    
    def test_format_with_performance_metric(self):
        """Test formatting log record with performance metric."""
        formatter = StructuredFormatter()
        
        record = MagicMock()
        record.created = time.time()
        record.levelname = "INFO"
        record.getMessage.return_value = "Performance test"
        record.module = "test_module"
        record.funcName = "test_function"
        record.lineno = 123
        record.component = "test_component"
        record.operation = "test_operation"
        
        # Add performance metric
        record.performance_metric = PerformanceMetric(
            component="test_component",
            operation="test_operation",
            start_time=time.time(),
            end_time=time.time() + 1,
            duration=1.0,
            success=True
        )
        
        formatted = formatter.format(record)
        log_entry = json.loads(formatted)
        
        assert "performance" in log_entry
        assert log_entry["performance"]["duration"] == 1.0


class TestDecorators:
    """Test cases for logging decorators."""
    
    def test_performance_monitor_decorator(self):
        """Test performance monitoring decorator."""
        @performance_monitor(ComponentType.DATA_CLEANING_AGENT, "test_operation")
        def test_function():
            time.sleep(0.1)
            return "success"
        
        result = test_function()
        assert result == "success"
        
        # Check that performance metric was logged
        manager = LoggingManager()
        assert len(manager.performance_metrics) > 0
        latest_metric = manager.performance_metrics[-1]
        assert latest_metric.component == ComponentType.DATA_CLEANING_AGENT.value
        assert latest_metric.operation == "test_operation"
        assert latest_metric.success is True
        assert latest_metric.duration >= 0.1
    
    def test_performance_monitor_decorator_with_exception(self):
        """Test performance monitoring decorator when function raises exception."""
        @performance_monitor(ComponentType.DATA_CLEANING_AGENT, "test_operation")
        def test_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_function()
        
        # Check that performance metric was logged with failure
        manager = LoggingManager()
        assert len(manager.performance_metrics) > 0
        latest_metric = manager.performance_metrics[-1]
        assert latest_metric.success is False
        assert latest_metric.error_message == "Test error"
    
    def test_error_tracker_decorator(self):
        """Test error tracking decorator."""
        @error_tracker(ComponentType.DATA_CLEANING_AGENT, "TestError")
        def test_function():
            raise ValueError("Test error message")
        
        with pytest.raises(ValueError):
            test_function()
        
        # Check that error event was logged
        manager = LoggingManager()
        assert len(manager.error_events) > 0
        latest_error = manager.error_events[-1]
        assert latest_error.component == ComponentType.DATA_CLEANING_AGENT.value
        assert latest_error.error_type == "TestError"
        assert latest_error.error_message == "Test error message"
    
    def test_operation_context(self):
        """Test operation context manager."""
        with operation_context(
            ComponentType.DATA_CLEANING_AGENT, 
            "test_operation", 
            {"test_key": "test_value"}
        ) as logger:
            assert logger is not None
            # Context manager should complete without error
    
    def test_log_data_quality_function(self):
        """Test log_data_quality function."""
        log_data_quality(
            ComponentType.DATA_CLEANING_AGENT,
            "test_dataset",
            total_records=1000,
            missing_values=50,
            outliers_detected=10,
            validation_errors=5
        )
        
        manager = LoggingManager()
        assert len(manager.data_quality_metrics) > 0
        latest_metric = manager.data_quality_metrics[-1]
        assert latest_metric.dataset_name == "test_dataset"
        assert latest_metric.total_records == 1000
        assert latest_metric.missing_values == 50


class TestIntegration:
    """Integration tests for logging system."""
    
    def test_full_logging_workflow(self):
        """Test complete logging workflow with all components."""
        # Simulate a complete operation with all logging types
        
        # 1. Performance monitoring
        @performance_monitor(ComponentType.DATA_CLEANING_AGENT, "full_workflow")
        @error_tracker(ComponentType.DATA_CLEANING_AGENT, "WorkflowError")
        def simulate_data_processing():
            # Simulate some processing time
            time.sleep(0.05)
            
            # Log data quality
            log_data_quality(
                ComponentType.DATA_CLEANING_AGENT,
                "test_dataset",
                total_records=1000,
                missing_values=25,
                outliers_detected=5,
                validation_errors=2
            )
            
            return {"status": "success", "records_processed": 1000}
        
        # Execute the workflow
        result = simulate_data_processing()
        assert result["status"] == "success"
        
        # Verify all logging components captured data
        manager = LoggingManager()
        
        # Check performance metrics
        assert len(manager.performance_metrics) > 0
        perf_metric = manager.performance_metrics[-1]
        assert perf_metric.operation == "full_workflow"
        assert perf_metric.success is True
        
        # Check data quality metrics
        assert len(manager.data_quality_metrics) > 0
        quality_metric = manager.data_quality_metrics[-1]
        assert quality_metric.dataset_name == "test_dataset"
        assert quality_metric.total_records == 1000
        
        # Get summaries
        perf_summary = manager.get_performance_summary()
        quality_summary = manager.get_data_quality_summary()
        
        assert len(perf_summary) > 0
        assert quality_summary["datasets_monitored"] > 0
    
    def test_logging_with_real_data_processing(self):
        """Test logging with actual data processing operations."""
        # Create temporary test data
        test_data = pd.DataFrame({
            'year': [2020, 2021, 2022],
            'sp500': [0.15, 0.12, -0.08],
            'bonds': [0.05, 0.03, 0.02]
        })
        
        @performance_monitor(ComponentType.DATA_LOADER, "process_test_data")
        def process_data(df):
            # Simulate data processing
            missing_count = df.isnull().sum().sum()
            
            # Log data quality
            log_data_quality(
                ComponentType.DATA_LOADER,
                "test_financial_data",
                total_records=len(df),
                missing_values=missing_count
            )
            
            return df.describe()
        
        result = process_data(test_data)
        assert result is not None
        
        # Verify logging captured the operation
        manager = LoggingManager()
        assert len(manager.performance_metrics) > 0
        assert len(manager.data_quality_metrics) > 0


class TestLogFileFormats:
    """Test log file formats and validation."""
    
    def test_structured_log_format_validation(self):
        """Test that structured logs produce valid JSON."""
        formatter = StructuredFormatter()
        
        record = MagicMock()
        record.created = time.time()
        record.levelname = "INFO"
        record.getMessage.return_value = "Test message"
        record.module = "test_module"
        record.funcName = "test_function"
        record.lineno = 123
        record.component = "test_component"
        record.operation = "test_operation"
        
        formatted = formatter.format(record)
        
        # Should be valid JSON
        try:
            log_entry = json.loads(formatted)
            assert isinstance(log_entry, dict)
            assert "timestamp" in log_entry
            assert "level" in log_entry
            assert "message" in log_entry
        except json.JSONDecodeError:
            pytest.fail("Formatted log is not valid JSON")
    
    def test_log_entry_required_fields(self):
        """Test that log entries contain all required fields."""
        formatter = StructuredFormatter()
        
        record = MagicMock()
        record.created = time.time()
        record.levelname = "ERROR"
        record.getMessage.return_value = "Error message"
        record.module = "error_module"
        record.funcName = "error_function"
        record.lineno = 456
        record.component = "error_component"
        record.operation = "error_operation"
        
        formatted = formatter.format(record)
        log_entry = json.loads(formatted)
        
        required_fields = [
            "timestamp", "level", "component", "operation", 
            "message", "module", "function", "line"
        ]
        
        for field in required_fields:
            assert field in log_entry, f"Required field '{field}' missing from log entry"


if __name__ == "__main__":
    pytest.main([__file__])