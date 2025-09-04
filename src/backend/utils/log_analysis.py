"""
Log analysis utilities for debugging and optimization of the Financial Returns Optimizer.

This module provides tools for analyzing log files, generating reports, and identifying
performance bottlenecks and issues.
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import pandas as pd

from config import LOGS_DIR


@dataclass
class LogAnalysisReport:
    """Log analysis report data structure."""
    analysis_period: Tuple[datetime, datetime]
    total_log_entries: int
    error_count: int
    warning_count: int
    performance_issues: List[Dict[str, Any]]
    data_quality_issues: List[Dict[str, Any]]
    component_activity: Dict[str, int]
    recommendations: List[str]


class LogAnalyzer:
    """Utility class for analyzing log files and generating insights."""
    
    def __init__(self, logs_directory: Path = LOGS_DIR):
        """Initialize log analyzer."""
        self.logs_dir = logs_directory
        self.performance_threshold = 5.0  # seconds
        self.quality_threshold = 80.0  # percentage
    
    def analyze_logs(self, hours_back: int = 24) -> LogAnalysisReport:
        """
        Analyze logs from the specified time period.
        
        Args:
            hours_back: Number of hours to look back from now
            
        Returns:
            LogAnalysisReport with analysis results
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        # Analyze different log files
        app_logs = self._parse_log_file(self.logs_dir / 'app.log', start_time, end_time)
        performance_logs = self._parse_structured_log_file(
            self.logs_dir / 'performance.log', start_time, end_time
        )
        data_quality_logs = self._parse_structured_log_file(
            self.logs_dir / 'data_quality.log', start_time, end_time
        )
        error_logs = self._parse_structured_log_file(
            self.logs_dir / 'errors.log', start_time, end_time
        )
        
        # Combine all logs for analysis
        all_logs = app_logs + performance_logs + data_quality_logs + error_logs
        
        # Generate analysis
        total_entries = len(all_logs)
        error_count = len([log for log in all_logs if log.get('level') == 'ERROR'])
        warning_count = len([log for log in all_logs if log.get('level') == 'WARNING'])
        
        performance_issues = self._identify_performance_issues(performance_logs)
        data_quality_issues = self._identify_data_quality_issues(data_quality_logs)
        component_activity = self._analyze_component_activity(all_logs)
        recommendations = self._generate_recommendations(
            performance_issues, data_quality_issues, error_count, warning_count
        )
        
        return LogAnalysisReport(
            analysis_period=(start_time, end_time),
            total_log_entries=total_entries,
            error_count=error_count,
            warning_count=warning_count,
            performance_issues=performance_issues,
            data_quality_issues=data_quality_issues,
            component_activity=component_activity,
            recommendations=recommendations
        )
    
    def _parse_log_file(self, log_file: Path, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Parse standard log file format."""
        if not log_file.exists():
            return []
        
        logs = []
        log_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (.*?) - (.*?) - (.*)'
        )
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    match = log_pattern.match(line.strip())
                    if match:
                        timestamp_str, component, level, message = match.groups()
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                        
                        if start_time <= timestamp <= end_time:
                            logs.append({
                                'timestamp': timestamp,
                                'component': component,
                                'level': level,
                                'message': message,
                                'source_file': log_file.name
                            })
        except Exception as e:
            print(f"Error parsing log file {log_file}: {e}")
        
        return logs
    
    def _parse_structured_log_file(self, log_file: Path, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Parse structured JSON log file format."""
        if not log_file.exists():
            return []
        
        logs = []
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        timestamp = datetime.fromisoformat(log_entry['timestamp'])
                        
                        if start_time <= timestamp <= end_time:
                            log_entry['timestamp'] = timestamp
                            log_entry['source_file'] = log_file.name
                            logs.append(log_entry)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception as e:
            print(f"Error parsing structured log file {log_file}: {e}")
        
        return logs
    
    def _identify_performance_issues(self, performance_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify performance issues from performance logs."""
        issues = []
        
        # Group by component and operation
        operations = defaultdict(list)
        for log in performance_logs:
            if 'performance' in log:
                perf_data = log['performance']
                key = f"{perf_data['component']}.{perf_data['operation']}"
                operations[key].append(perf_data)
        
        # Analyze each operation
        for operation, data_points in operations.items():
            durations = [dp['duration'] for dp in data_points]
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            
            # Check for slow operations
            if avg_duration > self.performance_threshold:
                issues.append({
                    'type': 'slow_operation',
                    'operation': operation,
                    'avg_duration': avg_duration,
                    'max_duration': max_duration,
                    'occurrences': len(data_points),
                    'severity': 'high' if avg_duration > self.performance_threshold * 2 else 'medium'
                })
            
            # Check for failed operations
            failed_ops = [dp for dp in data_points if not dp.get('success', True)]
            if failed_ops:
                issues.append({
                    'type': 'operation_failures',
                    'operation': operation,
                    'failure_count': len(failed_ops),
                    'total_attempts': len(data_points),
                    'failure_rate': len(failed_ops) / len(data_points) * 100,
                    'severity': 'high' if len(failed_ops) / len(data_points) > 0.1 else 'medium'
                })
        
        return issues
    
    def _identify_data_quality_issues(self, data_quality_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify data quality issues from data quality logs."""
        issues = []
        
        # Get latest quality metrics for each dataset
        latest_metrics = {}
        for log in data_quality_logs:
            if 'data_quality' in log:
                dq_data = log['data_quality']
                key = f"{dq_data['component']}_{dq_data['dataset_name']}"
                if key not in latest_metrics or log['timestamp'] > latest_metrics[key]['timestamp']:
                    latest_metrics[key] = {**dq_data, 'timestamp': log['timestamp']}
        
        # Analyze quality metrics
        for dataset, metrics in latest_metrics.items():
            quality_score = metrics['quality_score']
            
            if quality_score < self.quality_threshold:
                issues.append({
                    'type': 'low_quality_score',
                    'dataset': dataset,
                    'quality_score': quality_score,
                    'completeness': metrics['data_completeness_percent'],
                    'missing_values': metrics['missing_values'],
                    'outliers': metrics['outliers_detected'],
                    'validation_errors': metrics['validation_errors'],
                    'severity': 'high' if quality_score < 60 else 'medium'
                })
            
            # Check for specific issues
            missing_values = int(metrics.get('missing_values', 0))
            total_records = int(metrics.get('total_records', 1))
            if missing_values > total_records * 0.1:
                issues.append({
                    'type': 'high_missing_values',
                    'dataset': dataset,
                    'missing_count': missing_values,
                    'total_records': total_records,
                    'missing_percentage': missing_values / total_records * 100,
                    'severity': 'medium'
                })
            
            validation_errors = int(metrics.get('validation_errors', 0))
            if validation_errors > 0:
                issues.append({
                    'type': 'validation_errors',
                    'dataset': dataset,
                    'error_count': validation_errors,
                    'severity': 'high'
                })
        
        return issues
    
    def _analyze_component_activity(self, all_logs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze activity levels by component."""
        activity = Counter()
        
        for log in all_logs:
            component = log.get('component', 'unknown')
            activity[component] += 1
        
        return dict(activity)
    
    def _generate_recommendations(self, performance_issues: List[Dict[str, Any]], 
                                data_quality_issues: List[Dict[str, Any]], 
                                error_count: int, warning_count: int) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Performance recommendations
        slow_ops = [issue for issue in performance_issues if issue['type'] == 'slow_operation']
        if slow_ops:
            recommendations.append(
                f"Consider optimizing {len(slow_ops)} slow operations. "
                f"Focus on operations with average duration > {self.performance_threshold}s."
            )
        
        failed_ops = [issue for issue in performance_issues if issue['type'] == 'operation_failures']
        if failed_ops:
            high_failure_ops = [op for op in failed_ops if op['failure_rate'] > 10]
            if high_failure_ops:
                recommendations.append(
                    f"Investigate {len(high_failure_ops)} operations with high failure rates (>10%). "
                    "Consider adding retry logic or improving error handling."
                )
        
        # Data quality recommendations
        low_quality_datasets = [issue for issue in data_quality_issues if issue['type'] == 'low_quality_score']
        if low_quality_datasets:
            recommendations.append(
                f"Improve data quality for {len(low_quality_datasets)} datasets with scores below {self.quality_threshold}%. "
                "Focus on data validation and cleaning processes."
            )
        
        missing_value_issues = [issue for issue in data_quality_issues if issue['type'] == 'high_missing_values']
        if missing_value_issues:
            recommendations.append(
                f"Address missing value issues in {len(missing_value_issues)} datasets. "
                "Consider improving data collection or imputation strategies."
            )
        
        # Error recommendations
        if error_count > 10:
            recommendations.append(
                f"High error count ({error_count}) detected. "
                "Review error logs and implement additional error handling."
            )
        
        if warning_count > 20:
            recommendations.append(
                f"High warning count ({warning_count}) detected. "
                "Review warnings to prevent potential issues."
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append("System is operating within normal parameters. Continue monitoring.")
        
        return recommendations
    
    def generate_performance_report(self, hours_back: int = 24) -> str:
        """Generate a formatted performance report."""
        analysis = self.analyze_logs(hours_back)
        
        report = f"""
FINANCIAL RETURNS OPTIMIZER - LOG ANALYSIS REPORT
================================================

Analysis Period: {analysis.analysis_period[0].strftime('%Y-%m-%d %H:%M')} to {analysis.analysis_period[1].strftime('%Y-%m-%d %H:%M')}
Total Log Entries: {analysis.total_log_entries}
Errors: {analysis.error_count}
Warnings: {analysis.warning_count}

COMPONENT ACTIVITY
------------------
"""
        
        for component, count in sorted(analysis.component_activity.items(), key=lambda x: x[1], reverse=True):
            report += f"{component}: {count} entries\n"
        
        report += "\nPERFORMANCE ISSUES\n------------------\n"
        if analysis.performance_issues:
            for issue in analysis.performance_issues:
                report += f"• {issue['type'].replace('_', ' ').title()}: {issue.get('operation', 'N/A')}\n"
                if 'avg_duration' in issue:
                    report += f"  Average Duration: {issue['avg_duration']:.2f}s\n"
                if 'failure_rate' in issue:
                    report += f"  Failure Rate: {issue['failure_rate']:.1f}%\n"
                report += f"  Severity: {issue['severity']}\n\n"
        else:
            report += "No performance issues detected.\n\n"
        
        report += "DATA QUALITY ISSUES\n-------------------\n"
        if analysis.data_quality_issues:
            for issue in analysis.data_quality_issues:
                report += f"• {issue['type'].replace('_', ' ').title()}: {issue.get('dataset', 'N/A')}\n"
                if 'quality_score' in issue:
                    report += f"  Quality Score: {issue['quality_score']:.1f}%\n"
                if 'missing_percentage' in issue:
                    report += f"  Missing Values: {issue['missing_percentage']:.1f}%\n"
                report += f"  Severity: {issue['severity']}\n\n"
        else:
            report += "No data quality issues detected.\n\n"
        
        report += "RECOMMENDATIONS\n---------------\n"
        for i, rec in enumerate(analysis.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        return report
    
    def export_analysis_to_csv(self, hours_back: int = 24, output_file: Optional[Path] = None) -> Path:
        """Export log analysis to CSV format."""
        analysis = self.analyze_logs(hours_back)
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.logs_dir / f'log_analysis_{timestamp}.csv'
        
        # Prepare data for CSV
        rows = []
        
        # Performance issues
        for issue in analysis.performance_issues:
            rows.append({
                'category': 'performance',
                'type': issue['type'],
                'component': issue.get('operation', '').split('.')[0] if '.' in issue.get('operation', '') else '',
                'operation': issue.get('operation', ''),
                'severity': issue['severity'],
                'value': issue.get('avg_duration') or issue.get('failure_rate', 0),
                'details': json.dumps({k: v for k, v in issue.items() if k not in ['type', 'severity']})
            })
        
        # Data quality issues
        for issue in analysis.data_quality_issues:
            rows.append({
                'category': 'data_quality',
                'type': issue['type'],
                'component': issue.get('dataset', '').split('_')[0] if '_' in issue.get('dataset', '') else '',
                'operation': issue.get('dataset', ''),
                'severity': issue['severity'],
                'value': issue.get('quality_score') or issue.get('missing_percentage', 0),
                'details': json.dumps({k: v for k, v in issue.items() if k not in ['type', 'severity']})
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        
        return output_file
    
    def get_component_performance_trends(self, component: str, hours_back: int = 168) -> Dict[str, Any]:
        """Get performance trends for a specific component over time."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        performance_logs = self._parse_structured_log_file(
            self.logs_dir / 'performance.log', start_time, end_time
        )
        
        # Filter for specific component
        component_logs = [
            log for log in performance_logs 
            if log.get('performance', {}).get('component') == component
        ]
        
        if not component_logs:
            return {"message": f"No performance data found for component: {component}"}
        
        # Group by operation and time buckets (hourly)
        operations = defaultdict(lambda: defaultdict(list))
        
        for log in component_logs:
            perf_data = log['performance']
            operation = perf_data['operation']
            hour_bucket = log['timestamp'].replace(minute=0, second=0, microsecond=0)
            operations[operation][hour_bucket].append(perf_data['duration'])
        
        # Calculate trends
        trends = {}
        for operation, time_buckets in operations.items():
            hourly_averages = []
            for hour, durations in sorted(time_buckets.items()):
                hourly_averages.append({
                    'hour': hour.isoformat(),
                    'avg_duration': sum(durations) / len(durations),
                    'operation_count': len(durations)
                })
            
            trends[operation] = {
                'hourly_data': hourly_averages,
                'total_operations': sum(len(durations) for durations in time_buckets.values()),
                'overall_avg_duration': sum(
                    sum(durations) for durations in time_buckets.values()
                ) / sum(len(durations) for durations in time_buckets.values())
            }
        
        return trends


def create_log_analyzer() -> LogAnalyzer:
    """Factory function to create a log analyzer instance."""
    return LogAnalyzer()


def generate_daily_report() -> str:
    """Generate a daily log analysis report."""
    analyzer = create_log_analyzer()
    return analyzer.generate_performance_report(hours_back=24)


def generate_weekly_report() -> str:
    """Generate a weekly log analysis report."""
    analyzer = create_log_analyzer()
    return analyzer.generate_performance_report(hours_back=168)