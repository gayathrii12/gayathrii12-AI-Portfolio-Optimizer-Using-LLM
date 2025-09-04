"""
Monitoring dashboard utilities for the Financial Returns Optimizer.

This module provides utilities for creating monitoring dashboards and
real-time system health monitoring.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading
from dataclasses import asdict

from utils.logging_config import logging_manager, LoggingManager
from utils.log_analysis import LogAnalyzer, create_log_analyzer
from config import LOGS_DIR


class SystemHealthMonitor:
    """Real-time system health monitoring."""
    
    def __init__(self, update_interval: int = 60):
        """
        Initialize system health monitor.
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.health_data = {}
        self.log_analyzer = create_log_analyzer()
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                self.health_data = self._collect_health_data()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _collect_health_data(self) -> Dict[str, Any]:
        """Collect current system health data."""
        manager = logging_manager
        
        # Get recent performance data (last hour)
        recent_performance = [
            metric for metric in manager.performance_metrics
            if time.time() - metric.end_time < 3600
        ]
        
        # Get recent data quality data (last hour)
        recent_quality = [
            metric for metric in manager.data_quality_metrics
            if (datetime.now() - metric.timestamp).total_seconds() < 3600
        ]
        
        # Get recent errors (last hour)
        recent_errors = [
            error for error in manager.error_events
            if (datetime.now() - error.timestamp).total_seconds() < 3600
        ]
        
        # Calculate health metrics
        health_data = {
            "timestamp": datetime.now().isoformat(),
            "system_status": self._calculate_system_status(
                recent_performance, recent_quality, recent_errors
            ),
            "performance": {
                "total_operations": len(recent_performance),
                "successful_operations": sum(1 for m in recent_performance if m.success),
                "average_duration": (
                    sum(m.duration for m in recent_performance) / len(recent_performance)
                    if recent_performance else 0
                ),
                "slowest_operation": (
                    max(recent_performance, key=lambda x: x.duration)
                    if recent_performance else None
                )
            },
            "data_quality": {
                "datasets_monitored": len(set(m.dataset_name for m in recent_quality)),
                "average_quality_score": (
                    sum(m.quality_score for m in recent_quality) / len(recent_quality)
                    if recent_quality else 100
                ),
                "datasets_with_issues": len([
                    m for m in recent_quality 
                    if m.quality_score < 80 or m.validation_errors > 0
                ])
            },
            "errors": {
                "total_errors": len(recent_errors),
                "error_types": list(set(e.error_type for e in recent_errors)),
                "components_with_errors": list(set(e.component for e in recent_errors))
            }
        }
        
        return health_data
    
    def _calculate_system_status(self, performance_metrics, quality_metrics, error_events) -> str:
        """Calculate overall system status."""
        # Check for critical issues
        if len(error_events) > 10:  # More than 10 errors in last hour
            return "CRITICAL"
        
        # Check performance issues
        if performance_metrics:
            failure_rate = sum(1 for m in performance_metrics if not m.success) / len(performance_metrics)
            if failure_rate > 0.2:  # More than 20% failure rate
                return "WARNING"
            
            avg_duration = sum(m.duration for m in performance_metrics) / len(performance_metrics)
            if avg_duration > 10:  # Average operation takes more than 10 seconds
                return "WARNING"
        
        # Check data quality issues
        if quality_metrics:
            avg_quality = sum(m.quality_score for m in quality_metrics) / len(quality_metrics)
            if avg_quality < 70:  # Average quality below 70%
                return "WARNING"
        
        # Check for any errors
        if len(error_events) > 0:
            return "WARNING"
        
        return "HEALTHY"
    
    def get_current_health(self) -> Dict[str, Any]:
        """Get current system health data."""
        if not self.health_data:
            self.health_data = self._collect_health_data()
        return self.health_data
    
    def get_health_history(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get health data history (simulated for now)."""
        # In a real implementation, this would read from stored health snapshots
        current_health = self.get_current_health()
        return [current_health]  # Simplified for now


class DashboardGenerator:
    """Generate monitoring dashboard reports."""
    
    def __init__(self):
        """Initialize dashboard generator."""
        self.health_monitor = SystemHealthMonitor()
        self.log_analyzer = create_log_analyzer()
    
    def generate_system_overview(self) -> Dict[str, Any]:
        """Generate system overview dashboard data."""
        health_data = self.health_monitor.get_current_health()
        analysis = self.log_analyzer.analyze_logs(hours_back=24)
        
        overview = {
            "system_status": health_data["system_status"],
            "last_updated": health_data["timestamp"],
            "summary": {
                "total_log_entries": analysis.total_log_entries,
                "error_count": analysis.error_count,
                "warning_count": analysis.warning_count,
                "performance_issues": len(analysis.performance_issues),
                "data_quality_issues": len(analysis.data_quality_issues)
            },
            "component_activity": analysis.component_activity,
            "recent_performance": health_data["performance"],
            "data_quality_status": health_data["data_quality"],
            "error_summary": health_data["errors"],
            "recommendations": analysis.recommendations[:5]  # Top 5 recommendations
        }
        
        return overview
    
    def generate_performance_dashboard(self) -> Dict[str, Any]:
        """Generate performance-focused dashboard data."""
        analysis = self.log_analyzer.analyze_logs(hours_back=24)
        
        # Group performance issues by severity
        critical_issues = [
            issue for issue in analysis.performance_issues 
            if issue.get('severity') == 'high'
        ]
        warning_issues = [
            issue for issue in analysis.performance_issues 
            if issue.get('severity') == 'medium'
        ]
        
        dashboard = {
            "performance_summary": {
                "total_issues": len(analysis.performance_issues),
                "critical_issues": len(critical_issues),
                "warning_issues": len(warning_issues)
            },
            "slow_operations": [
                issue for issue in analysis.performance_issues 
                if issue['type'] == 'slow_operation'
            ],
            "failed_operations": [
                issue for issue in analysis.performance_issues 
                if issue['type'] == 'operation_failures'
            ],
            "component_performance": self._get_component_performance_breakdown(),
            "performance_trends": self._get_performance_trends()
        }
        
        return dashboard
    
    def generate_data_quality_dashboard(self) -> Dict[str, Any]:
        """Generate data quality-focused dashboard data."""
        analysis = self.log_analyzer.analyze_logs(hours_back=24)
        
        # Group data quality issues by type
        quality_issues_by_type = {}
        for issue in analysis.data_quality_issues:
            issue_type = issue['type']
            if issue_type not in quality_issues_by_type:
                quality_issues_by_type[issue_type] = []
            quality_issues_by_type[issue_type].append(issue)
        
        dashboard = {
            "quality_summary": {
                "total_issues": len(analysis.data_quality_issues),
                "issues_by_type": {
                    issue_type: len(issues) 
                    for issue_type, issues in quality_issues_by_type.items()
                }
            },
            "dataset_quality_scores": self._get_dataset_quality_scores(),
            "data_completeness_trends": self._get_data_completeness_trends(),
            "validation_error_summary": self._get_validation_error_summary()
        }
        
        return dashboard
    
    def _get_component_performance_breakdown(self) -> Dict[str, Any]:
        """Get performance breakdown by component."""
        manager = logging_manager
        
        # Group performance metrics by component
        component_metrics = {}
        for metric in manager.performance_metrics:
            if metric.component not in component_metrics:
                component_metrics[metric.component] = []
            component_metrics[metric.component].append(metric)
        
        breakdown = {}
        for component, metrics in component_metrics.items():
            durations = [m.duration for m in metrics]
            breakdown[component] = {
                "total_operations": len(metrics),
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "success_rate": sum(1 for m in metrics if m.success) / len(metrics) * 100
            }
        
        return breakdown
    
    def _get_performance_trends(self) -> List[Dict[str, Any]]:
        """Get performance trends over time."""
        # Simplified implementation - in practice, this would analyze trends over time
        manager = logging_manager
        
        # Group by hour for the last 24 hours
        hourly_performance = {}
        now = datetime.now()
        
        for metric in manager.performance_metrics:
            metric_time = datetime.fromtimestamp(metric.end_time)
            if (now - metric_time).total_seconds() < 24 * 3600:  # Last 24 hours
                hour_key = metric_time.replace(minute=0, second=0, microsecond=0)
                if hour_key not in hourly_performance:
                    hourly_performance[hour_key] = []
                hourly_performance[hour_key].append(metric)
        
        trends = []
        for hour, metrics in sorted(hourly_performance.items()):
            durations = [m.duration for m in metrics]
            trends.append({
                "hour": hour.isoformat(),
                "operation_count": len(metrics),
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "success_rate": sum(1 for m in metrics if m.success) / len(metrics) * 100
            })
        
        return trends
    
    def _get_dataset_quality_scores(self) -> Dict[str, float]:
        """Get quality scores for all monitored datasets."""
        manager = logging_manager
        
        # Get latest quality score for each dataset
        latest_scores = {}
        for metric in manager.data_quality_metrics:
            key = f"{metric.component}_{metric.dataset_name}"
            if key not in latest_scores or metric.timestamp > latest_scores[key]['timestamp']:
                latest_scores[key] = {
                    'score': metric.quality_score,
                    'timestamp': metric.timestamp
                }
        
        return {key: data['score'] for key, data in latest_scores.items()}
    
    def _get_data_completeness_trends(self) -> List[Dict[str, Any]]:
        """Get data completeness trends."""
        manager = logging_manager
        
        # Group by dataset and time
        completeness_data = []
        for metric in manager.data_quality_metrics:
            completeness_data.append({
                "dataset": f"{metric.component}_{metric.dataset_name}",
                "timestamp": metric.timestamp.isoformat(),
                "completeness": metric.data_completeness_percent,
                "quality_score": metric.quality_score
            })
        
        return sorted(completeness_data, key=lambda x: x['timestamp'])
    
    def _get_validation_error_summary(self) -> Dict[str, Any]:
        """Get validation error summary."""
        manager = logging_manager
        
        total_validation_errors = sum(
            metric.validation_errors for metric in manager.data_quality_metrics
        )
        
        datasets_with_errors = [
            f"{metric.component}_{metric.dataset_name}"
            for metric in manager.data_quality_metrics
            if metric.validation_errors > 0
        ]
        
        return {
            "total_validation_errors": total_validation_errors,
            "datasets_with_errors": len(set(datasets_with_errors)),
            "error_details": datasets_with_errors
        }
    
    def export_dashboard_data(self, output_file: Optional[Path] = None) -> Path:
        """Export complete dashboard data to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = LOGS_DIR / f'dashboard_data_{timestamp}.json'
        
        dashboard_data = {
            "generated_at": datetime.now().isoformat(),
            "system_overview": self.generate_system_overview(),
            "performance_dashboard": self.generate_performance_dashboard(),
            "data_quality_dashboard": self.generate_data_quality_dashboard()
        }
        
        with open(output_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        return output_file


def create_monitoring_dashboard() -> DashboardGenerator:
    """Factory function to create a monitoring dashboard."""
    return DashboardGenerator()


def start_system_monitoring(update_interval: int = 60) -> SystemHealthMonitor:
    """Start system health monitoring."""
    monitor = SystemHealthMonitor(update_interval)
    monitor.start_monitoring()
    return monitor


def get_system_status() -> Dict[str, Any]:
    """Get current system status."""
    dashboard = create_monitoring_dashboard()
    return dashboard.generate_system_overview()