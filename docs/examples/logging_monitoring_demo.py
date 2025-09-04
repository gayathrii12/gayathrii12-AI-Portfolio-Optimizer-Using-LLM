"""
Demonstration of comprehensive logging and monitoring system.

This example shows how to use the logging, monitoring, and analysis
capabilities of the Financial Returns Optimizer.
"""

import time
import json
from pathlib import Path
from datetime import datetime

from utils.logging_config import (
    logging_manager,
    ComponentType,
    performance_monitor,
    error_tracker,
    operation_context,
    log_data_quality
)
from utils.log_analysis import create_log_analyzer, generate_daily_report
from utils.monitoring_dashboard import (
    create_monitoring_dashboard,
    start_system_monitoring,
    get_system_status
)


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("=== Performance Monitoring Demo ===")
    
    @performance_monitor(ComponentType.DATA_CLEANING_AGENT, "demo_data_processing")
    def simulate_data_processing(processing_time: float = 1.0):
        """Simulate data processing with configurable duration."""
        print(f"Processing data for {processing_time} seconds...")
        time.sleep(processing_time)
        return {"status": "success", "records_processed": 1000}
    
    # Simulate various operations with different performance characteristics
    print("Running fast operation...")
    simulate_data_processing(0.5)
    
    print("Running normal operation...")
    simulate_data_processing(1.0)
    
    print("Running slow operation...")
    simulate_data_processing(3.0)
    
    # Get performance summary
    summary = logging_manager.get_performance_summary()
    print(f"Performance Summary: {json.dumps(summary, indent=2)}")


def demonstrate_error_tracking():
    """Demonstrate error tracking capabilities."""
    print("\n=== Error Tracking Demo ===")
    
    @error_tracker(ComponentType.DATA_CLEANING_AGENT, "DataProcessingError")
    def simulate_operation_with_error():
        """Simulate an operation that raises an error."""
        raise ValueError("Simulated data processing error")
    
    @error_tracker(ComponentType.ASSET_PREDICTOR_AGENT, "PredictionError")
    def simulate_prediction_error():
        """Simulate a prediction error."""
        raise RuntimeError("Simulated prediction failure")
    
    # Simulate various errors
    try:
        simulate_operation_with_error()
    except ValueError:
        print("Caught and logged data processing error")
    
    try:
        simulate_prediction_error()
    except RuntimeError:
        print("Caught and logged prediction error")
    
    # Get error summary
    error_summary = logging_manager.get_error_summary()
    print(f"Error Summary: {json.dumps(error_summary, indent=2)}")


def demonstrate_data_quality_monitoring():
    """Demonstrate data quality monitoring capabilities."""
    print("\n=== Data Quality Monitoring Demo ===")
    
    # Simulate data quality monitoring for different datasets
    datasets = [
        {
            "name": "historical_returns_sp500",
            "total_records": 1000,
            "missing_values": 25,
            "outliers": 5,
            "validation_errors": 0
        },
        {
            "name": "bond_returns_data",
            "total_records": 800,
            "missing_values": 100,  # High missing values
            "outliers": 15,
            "validation_errors": 3
        },
        {
            "name": "real_estate_data",
            "total_records": 500,
            "missing_values": 5,
            "outliers": 2,
            "validation_errors": 0
        }
    ]
    
    for dataset in datasets:
        print(f"Monitoring data quality for {dataset['name']}...")
        log_data_quality(
            ComponentType.DATA_CLEANING_AGENT,
            dataset["name"],
            total_records=dataset["total_records"],
            missing_values=dataset["missing_values"],
            outliers_detected=dataset["outliers"],
            validation_errors=dataset["validation_errors"]
        )
    
    # Get data quality summary
    quality_summary = logging_manager.get_data_quality_summary()
    print(f"Data Quality Summary: {json.dumps(quality_summary, indent=2)}")


def demonstrate_operation_context():
    """Demonstrate structured operation context logging."""
    print("\n=== Operation Context Demo ===")
    
    def simulate_complex_operation():
        """Simulate a complex operation with multiple steps."""
        with operation_context(
            ComponentType.ORCHESTRATOR,
            "complex_portfolio_analysis",
            {
                "user_id": "demo_user",
                "risk_profile": "moderate",
                "investment_amount": 100000
            }
        ) as logger:
            
            logger.info("Starting portfolio analysis")
            
            # Step 1: Data loading
            with operation_context(
                ComponentType.DATA_LOADER,
                "load_historical_data",
                {"data_source": "histretSP.xls"}
            ) as data_logger:
                data_logger.info("Loading historical returns data")
                time.sleep(0.5)  # Simulate processing
                data_logger.info("Data loading completed")
            
            # Step 2: Data cleaning
            with operation_context(
                ComponentType.DATA_CLEANING_AGENT,
                "clean_and_validate",
                {"strategy": "forward_fill"}
            ) as clean_logger:
                clean_logger.info("Starting data cleaning")
                time.sleep(0.3)  # Simulate processing
                clean_logger.info("Data cleaning completed")
            
            # Step 3: Portfolio allocation
            with operation_context(
                ComponentType.PORTFOLIO_ALLOCATOR_AGENT,
                "calculate_allocation",
                {"risk_profile": "moderate"}
            ) as alloc_logger:
                alloc_logger.info("Calculating portfolio allocation")
                time.sleep(0.4)  # Simulate processing
                alloc_logger.info("Portfolio allocation completed")
            
            logger.info("Complex portfolio analysis completed successfully")
    
    simulate_complex_operation()


def demonstrate_log_analysis():
    """Demonstrate log analysis capabilities."""
    print("\n=== Log Analysis Demo ===")
    
    # Create log analyzer
    analyzer = create_log_analyzer()
    
    # Generate analysis report
    print("Generating log analysis report...")
    report = analyzer.generate_performance_report(hours_back=1)
    print("Log Analysis Report:")
    print(report)
    
    # Export analysis to CSV
    csv_file = analyzer.export_analysis_to_csv(hours_back=1)
    print(f"Analysis exported to: {csv_file}")


def demonstrate_monitoring_dashboard():
    """Demonstrate monitoring dashboard capabilities."""
    print("\n=== Monitoring Dashboard Demo ===")
    
    # Create dashboard
    dashboard = create_monitoring_dashboard()
    
    # Generate system overview
    print("Generating system overview...")
    overview = dashboard.generate_system_overview()
    print(f"System Status: {overview['system_status']}")
    print(f"Total Log Entries: {overview['summary']['total_log_entries']}")
    print(f"Error Count: {overview['summary']['error_count']}")
    
    # Generate performance dashboard
    print("\nGenerating performance dashboard...")
    perf_dashboard = dashboard.generate_performance_dashboard()
    print(f"Performance Issues: {perf_dashboard['performance_summary']['total_issues']}")
    
    # Generate data quality dashboard
    print("\nGenerating data quality dashboard...")
    quality_dashboard = dashboard.generate_data_quality_dashboard()
    print(f"Data Quality Issues: {quality_dashboard['quality_summary']['total_issues']}")
    
    # Export complete dashboard data
    dashboard_file = dashboard.export_dashboard_data()
    print(f"Dashboard data exported to: {dashboard_file}")


def demonstrate_real_time_monitoring():
    """Demonstrate real-time system monitoring."""
    print("\n=== Real-Time Monitoring Demo ===")
    
    # Start system monitoring
    print("Starting real-time system monitoring...")
    monitor = start_system_monitoring(update_interval=5)  # Update every 5 seconds
    
    # Let it run for a short time
    print("Monitoring system for 10 seconds...")
    time.sleep(10)
    
    # Get current system status
    status = get_system_status()
    print(f"Current System Status: {status['system_status']}")
    print(f"Last Updated: {status['last_updated']}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("Monitoring stopped")


def run_comprehensive_demo():
    """Run comprehensive demonstration of all logging and monitoring features."""
    print("Financial Returns Optimizer - Logging and Monitoring Demo")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrate_performance_monitoring()
    demonstrate_error_tracking()
    demonstrate_data_quality_monitoring()
    demonstrate_operation_context()
    demonstrate_log_analysis()
    demonstrate_monitoring_dashboard()
    demonstrate_real_time_monitoring()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("Check the logs directory for generated log files and reports.")


if __name__ == "__main__":
    run_comprehensive_demo()