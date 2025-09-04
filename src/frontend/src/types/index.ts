// System Status Types
export type SystemStatus = 'HEALTHY' | 'WARNING' | 'CRITICAL';

// Performance Monitoring Types
export interface PerformanceMetric {
  component: string;
  operation: string;
  start_time: number;
  end_time: number;
  duration: number;
  memory_usage_mb?: number;
  cpu_usage_percent?: number;
  data_size?: number;
  success: boolean;
  error_message?: string;
}

export interface PerformanceSummary {
  [component: string]: {
    total_operations: number;
    avg_duration: number;
    min_duration: number;
    max_duration: number;
    success_rate: number;
  };
}

// Data Quality Types
export interface DataQualityMetric {
  component: string;
  dataset_name: string;
  total_records: number;
  missing_values: number;
  outliers_detected: number;
  validation_errors: number;
  data_completeness_percent: number;
  timestamp: string;
  quality_score: number;
}

export interface DataQualitySummary {
  datasets_monitored: number;
  average_quality_score: number;
  datasets: {
    [key: string]: {
      quality_score: number;
      completeness: number;
      total_records: number;
      issues: {
        missing_values: number;
        outliers: number;
        validation_errors: number;
      };
    };
  };
}

// Error Tracking Types
export interface ErrorEvent {
  component: string;
  error_type: string;
  error_message: string;
  stack_trace: string;
  timestamp: string;
  severity: string;
  context: Record<string, any>;
  user_input?: Record<string, any>;
}

export interface ErrorSummary {
  total_errors: number;
  errors_by_component: Record<string, number>;
  errors_by_type: Record<string, number>;
  recent_errors: Array<{
    component: string;
    type: string;
    message: string;
    timestamp: string;
  }>;
}

// System Health Types
export interface SystemHealthData {
  timestamp: string;
  system_status: SystemStatus;
  performance: {
    total_operations: number;
    successful_operations: number;
    average_duration: number;
    slowest_operation?: PerformanceMetric;
  };
  data_quality: {
    datasets_monitored: number;
    average_quality_score: number;
    datasets_with_issues: number;
  };
  errors: {
    total_errors: number;
    error_types: string[];
    components_with_errors: string[];
  };
}

// Dashboard Types
export interface DashboardData {
  system_status: SystemStatus;
  last_updated: string;
  summary: {
    total_log_entries: number;
    error_count: number;
    warning_count: number;
    performance_issues: number;
    data_quality_issues: number;
  };
  component_activity: Record<string, number>;
  recent_performance: SystemHealthData['performance'];
  data_quality_status: SystemHealthData['data_quality'];
  error_summary: SystemHealthData['errors'];
  recommendations: string[];
}

// Portfolio Analysis Types
export interface PortfolioAllocation {
  sp500: number;
  small_cap: number;
  bonds: number;
  real_estate: number;
  gold: number;
}

export interface PieChartDataPoint {
  name: string;
  value: number;
  color: string;
  percentage: string;
}

export interface LineChartDataPoint {
  year: number;
  portfolio_value: number;
  formatted_value: string;
  annual_return?: number;
  cumulative_return?: number;
}

export interface ComparisonChartDataPoint {
  year: number;
  portfolio_value: number;
  benchmark_value: number;
  portfolio_return: number;
  benchmark_return: number;
  outperformance: number;
}

export interface RiskMetrics {
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  beta: number;
  alpha: number;
}

export interface RiskVisualizationData {
  portfolio_metrics: Array<{
    metric: string;
    value: number;
    benchmark: number;
  }>;
  risk_score: number;
  risk_level: string;
}

// Chart Types
export interface ChartData {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    backgroundColor?: string | string[];
    borderColor?: string | string[];
    borderWidth?: number;
    fill?: boolean;
  }>;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  timestamp: string;
}

// Log Analysis Types
export interface LogAnalysisReport {
  analysis_period: [string, string];
  total_log_entries: number;
  error_count: number;
  warning_count: number;
  performance_issues: Array<{
    type: string;
    operation?: string;
    avg_duration?: number;
    max_duration?: number;
    occurrences?: number;
    failure_count?: number;
    total_attempts?: number;
    failure_rate?: number;
    severity: string;
  }>;
  data_quality_issues: Array<{
    type: string;
    dataset?: string;
    quality_score?: number;
    completeness?: number;
    missing_values?: number;
    outliers?: number;
    validation_errors?: number;
    missing_count?: number;
    total_records?: number;
    missing_percentage?: number;
    error_count?: number;
    severity: string;
  }>;
  component_activity: Record<string, number>;
  recommendations: string[];
}

// Component Props Types
export interface ChartProps {
  data: any;
  title?: string;
  height?: number;
  loading?: boolean;
  error?: string;
}

export interface MetricCardProps {
  title: string;
  value: string | number;
  change?: {
    value: number;
    type: 'positive' | 'negative';
  };
  status?: SystemStatus;
  loading?: boolean;
}

export interface StatusBadgeProps {
  status: SystemStatus;
  size?: 'small' | 'medium' | 'large';
}

// Navigation Types
export interface NavItem {
  path: string;
  label: string;
  icon: string;
  badge?: number;
}