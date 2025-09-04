"""
Tests for log analysis utilities.

This module tests the log analysis functionality including report generation,
performance analysis, and optimization recommendations.
"""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from utils.log_analysis import (
    LogAnalyzer,
    LogAnalysisReport,
    create_log_analyzer,
    generate_daily_report,
    generate_weekly_report
)
from utils.logging_config import (
    LoggingManager,
    ComponentType,
    PerformanceMetric,
    DataQualityMetric,
    ErrorEvent
)


class TestLogAnalyzer:
    """Test cases for LogAnalyzer class."""
    
    @pytest.fixture
    def temp_logs_dir(self):
        """Create temporary logs directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            yield logs_dir
    
    @pytest.fixture
    def sample_log_data(self, temp_logs_dir):
        """Create sample log files for testing."""
        # Create sample app.log
        app_log = temp_logs_dir / 'app.log'
        with open(app_log, 'w') as f:
            f.write("2025-08-27 10:00:00,000 - test_component - INFO - Test info message\n")
            f.write("2025-08-27 10:01:00,000 - test_component - ERROR - Test error message\n")
            f.write("2025-08-27 10:02:00,000 - test_component - WARNING - Test warning message\n")
        
        # Create sample performance.log
        perf_log = temp_logs_dir / 'performance.log'
        perf_data = {
            "timestamp": "2025-08-27T10:00:00",
            "level": "INFO",
            "component": "data_cleaning_agent",
            "operation": "performance_test",
            "message": "Performance test",
            "performance": {
                "component": "data_cleaning_agent",
                "operation": "load_data",
                "duration": 2.5,
                "success": True
            }
        }
        with open(perf_log, 'w') as f:
            f.write(json.dumps(perf_data) + "\n")
        
        return temp_logs_dir    

    def test_log_analyzer_initialization(self, temp_logs_dir):
        """Test LogAnalyzer initialization."""
        analyzer = LogAnalyzer(temp_logs_dir)
        assert analyzer.logs_dir == temp_logs_dir
        assert analyzer.performance_threshold == 5.0
        assert analyzer.quality_threshold == 80.0
    
    def test_parse_log_file(self, sample_log_data):
        """Test parsing standard log file format."""
        analyzer = LogAnalyzer(sample_log_data)
        
        start_time = datetime(2025, 8, 27, 9, 0, 0)
        end_time = datetime(2025, 8, 27, 11, 0, 0)
        
        logs = analyzer._parse_log_file(
            sample_log_data / 'app.log', 
            start_time, 
            end_time
        )
        
        assert len(logs) == 3
        assert logs[0]['level'] == 'INFO'
        assert logs[1]['level'] == 'ERROR'
        assert logs[2]['level'] == 'WARNING'
    
    def test_parse_structured_log_file(self, sample_log_data):
        """Test parsing structured JSON log file format."""
        analyzer = LogAnalyzer(sample_log_data)
        
        start_time = datetime(2025, 8, 27, 9, 0, 0)
        end_time = datetime(2025, 8, 27, 11, 0, 0)
        
        logs = analyzer._parse_structured_log_file(
            sample_log_data / 'performance.log',
            start_time,
            end_time
        )
        
        assert len(logs) == 1
        assert logs[0]['component'] == 'data_cleaning_agent'
        assert 'performance' in logs[0]
    
    def test_identify_performance_issues(self, temp_logs_dir):
        """Test identification of performance issues."""
        analyzer = LogAnalyzer(temp_logs_dir)
        
        # Create performance logs with slow operations
        performance_logs = [
            {
                'performance': {
                    'component': 'test_component',
                    'operation': 'slow_operation',
                    'duration': 10.0,  # Exceeds threshold
                    'success': True
                }
            },
            {
                'performance': {
                    'component': 'test_component',
                    'operation': 'failed_operation',
                    'duration': 2.0,
                    'success': False
                }
            }
        ]
        
        issues = analyzer._identify_performance_issues(performance_logs)
        
        # Should identify slow operation
        slow_ops = [issue for issue in issues if issue['type'] == 'slow_operation']
        assert len(slow_ops) > 0
        
        # Should identify failed operation
        failed_ops = [issue for issue in issues if issue['type'] == 'operation_failures']
        assert len(failed_ops) > 0
    
    def test_identify_data_quality_issues(self, temp_logs_dir):
        """Test identification of data quality issues."""
        analyzer = LogAnalyzer(temp_logs_dir)
        
        # Create data quality logs with issues
        data_quality_logs = [
            {
                'timestamp': datetime.now(),
                'data_quality': {
                    'component': 'test_component',
                    'dataset_name': 'low_quality_dataset',
                    'total_records': 1000,
                    'missing_values': 200,  # High missing values
                    'outliers_detected': 50,
                    'validation_errors': 10,
                    'data_completeness_percent': 80.0,
                    'quality_score': 60.0  # Below threshold
                }
            }
        ]
        
        issues = analyzer._identify_data_quality_issues(data_quality_logs)
        
        # Should identify low quality score
        quality_issues = [issue for issue in issues if issue['type'] == 'low_quality_score']
        assert len(quality_issues) > 0
        
        # Should identify high missing values
        missing_issues = [issue for issue in issues if issue['type'] == 'high_missing_values']
        assert len(missing_issues) > 0
    
    def test_analyze_component_activity(self, temp_logs_dir):
        """Test analysis of component activity."""
        analyzer = LogAnalyzer(temp_logs_dir)
        
        logs = [
            {'component': 'component_a'},
            {'component': 'component_a'},
            {'component': 'component_b'},
            {'component': 'unknown'}
        ]
        
        activity = analyzer._analyze_component_activity(logs)
        
        assert activity['component_a'] == 2
        assert activity['component_b'] == 1
        assert activity['unknown'] == 1
    
    def test_generate_recommendations(self, temp_logs_dir):
        """Test generation of optimization recommendations."""
        analyzer = LogAnalyzer(temp_logs_dir)
        
        performance_issues = [
            {'type': 'slow_operation', 'operation': 'test_op'}
        ]
        data_quality_issues = [
            {'type': 'low_quality_score', 'dataset': 'test_dataset'}
        ]
        
        recommendations = analyzer._generate_recommendations(
            performance_issues, data_quality_issues, 5, 10
        )
        
        assert len(recommendations) > 0
        assert any('slow operations' in rec.lower() for rec in recommendations)
        assert any('data quality' in rec.lower() for rec in recommendations)
    
    def test_generate_performance_report(self, sample_log_data):
        """Test generation of performance report."""
        analyzer = LogAnalyzer(sample_log_data)
        
        report = analyzer.generate_performance_report(hours_back=24)
        
        assert isinstance(report, str)
        assert "LOG ANALYSIS REPORT" in report
        assert "COMPONENT ACTIVITY" in report
        assert "PERFORMANCE ISSUES" in report
        assert "DATA QUALITY ISSUES" in report
        assert "RECOMMENDATIONS" in report
    
    def test_export_analysis_to_csv(self, temp_logs_dir):
        """Test exporting analysis to CSV format."""
        analyzer = LogAnalyzer(temp_logs_dir)
        
        # Mock some analysis data
        analyzer.performance_metrics = [
            PerformanceMetric(
                component="test_component",
                operation="test_operation",
                start_time=0,
                end_time=1,
                duration=1.0,
                success=True
            )
        ]
        
        output_file = analyzer.export_analysis_to_csv(hours_back=24)
        
        assert output_file.exists()
        assert output_file.suffix == '.csv'
        
        # Verify CSV content
        df = pd.read_csv(output_file)
        assert len(df) > 0
        assert 'category' in df.columns
        assert 'type' in df.columns


class TestFactoryFunctions:
    """Test factory functions and utility functions."""
    
    def test_create_log_analyzer(self):
        """Test create_log_analyzer factory function."""
        analyzer = create_log_analyzer()
        assert isinstance(analyzer, LogAnalyzer)
    
    @patch('utils.log_analysis.LogAnalyzer')
    def test_generate_daily_report(self, mock_analyzer_class):
        """Test generate_daily_report function."""
        mock_analyzer = MagicMock()
        mock_analyzer.generate_performance_report.return_value = "Daily report"
        mock_analyzer_class.return_value = mock_analyzer
        
        report = generate_daily_report()
        
        assert report == "Daily report"
        mock_analyzer.generate_performance_report.assert_called_once_with(hours_back=24)
    
    @patch('utils.log_analysis.LogAnalyzer')
    def test_generate_weekly_report(self, mock_analyzer_class):
        """Test generate_weekly_report function."""
        mock_analyzer = MagicMock()
        mock_analyzer.generate_performance_report.return_value = "Weekly report"
        mock_analyzer_class.return_value = mock_analyzer
        
        report = generate_weekly_report()
        
        assert report == "Weekly report"
        mock_analyzer.generate_performance_report.assert_called_once_with(hours_back=168)


class TestIntegrationWithLoggingSystem:
    """Integration tests with the logging system."""
    
    def test_integration_with_logging_manager(self):
        """Test integration between log analysis and logging manager."""
        # Generate some test data through logging manager
        manager = LoggingManager()
        
        # Add performance metrics
        metric = PerformanceMetric(
            component="integration_test",
            operation="test_operation",
            start_time=0,
            end_time=2,
            duration=2.0,
            success=True
        )
        manager.log_performance_metric(metric)
        
        # Add data quality metrics
        quality_metric = DataQualityMetric(
            component="integration_test",
            dataset_name="test_dataset",
            total_records=1000,
            missing_values=50,
            outliers_detected=10,
            validation_errors=5,
            data_completeness_percent=95.0,
            timestamp=datetime.now(),
            quality_score=85.0
        )
        manager.log_data_quality_metric(quality_metric)
        
        # Verify data was logged
        assert len(manager.performance_metrics) > 0
        assert len(manager.data_quality_metrics) > 0
        
        # Test summaries
        perf_summary = manager.get_performance_summary()
        quality_summary = manager.get_data_quality_summary()
        
        assert "integration_test" in perf_summary
        assert quality_summary["datasets_monitored"] > 0


if __name__ == "__main__":
    pytest.main([__file__])