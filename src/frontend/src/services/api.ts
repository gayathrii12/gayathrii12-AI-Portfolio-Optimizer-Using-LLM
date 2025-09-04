import axios, { AxiosResponse } from 'axios';
import {
  DashboardData,
  SystemHealthData,
  PerformanceSummary,
  DataQualitySummary,
  ErrorSummary,
  LogAnalysisReport,
  PortfolioAllocation,
  PieChartDataPoint,
  LineChartDataPoint,
  ComparisonChartDataPoint,
  RiskVisualizationData,
  ApiResponse
} from '../types';

// Base API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Response Error:', error);
    if (error.response?.status === 404) {
      console.warn('API endpoint not found, returning mock data');
    }
    return Promise.reject(error);
  }
);

// API Service Class
class ApiService {
  // Dashboard APIs
  async getDashboardData(): Promise<DashboardData> {
    try {
      const response: AxiosResponse<ApiResponse<DashboardData>> = await apiClient.get('/api/dashboard');
      return response.data.data;
    } catch (error) {
      console.warn('Dashboard API not available, returning mock data');
      return this.getMockDashboardData();
    }
  }

  async getSystemHealth(): Promise<SystemHealthData> {
    try {
      const response: AxiosResponse<ApiResponse<SystemHealthData>> = await apiClient.get('/api/system-health');
      return response.data.data;
    } catch (error) {
      console.warn('System Health API not available, returning mock data');
      return this.getMockSystemHealthData();
    }
  }

  // Performance Monitoring APIs
  async getPerformanceSummary(hoursBack: number = 24): Promise<PerformanceSummary> {
    try {
      const response: AxiosResponse<ApiResponse<PerformanceSummary>> = await apiClient.get(
        `/api/performance/summary?hours_back=${hoursBack}`
      );
      return response.data.data;
    } catch (error) {
      console.warn('Performance API not available, returning mock data');
      return this.getMockPerformanceSummary();
    }
  }

  async getPerformanceTrends(component: string, hoursBack: number = 168): Promise<any> {
    try {
      const response = await apiClient.get(
        `/api/performance/trends/${component}?hours_back=${hoursBack}`
      );
      return response.data.data;
    } catch (error) {
      console.warn('Performance Trends API not available, returning mock data');
      return this.getMockPerformanceTrends();
    }
  }

  // Data Quality APIs
  async getDataQualitySummary(): Promise<DataQualitySummary> {
    try {
      const response: AxiosResponse<ApiResponse<DataQualitySummary>> = await apiClient.get('/api/data-quality/summary');
      return response.data.data;
    } catch (error) {
      console.warn('Data Quality API not available, returning mock data');
      return this.getMockDataQualitySummary();
    }
  }

  // Error Tracking APIs
  async getErrorSummary(): Promise<ErrorSummary> {
    try {
      const response: AxiosResponse<ApiResponse<ErrorSummary>> = await apiClient.get('/api/errors/summary');
      return response.data.data;
    } catch (error) {
      console.warn('Error Tracking API not available, returning mock data');
      return this.getMockErrorSummary();
    }
  }

  // Log Analysis APIs
  async getLogAnalysis(hoursBack: number = 24): Promise<LogAnalysisReport> {
    try {
      const response: AxiosResponse<ApiResponse<LogAnalysisReport>> = await apiClient.get(
        `/api/logs/analysis?hours_back=${hoursBack}`
      );
      return response.data.data;
    } catch (error) {
      console.warn('Log Analysis API not available, returning mock data');
      return this.getMockLogAnalysisReport();
    }
  }

  // Portfolio Analysis APIs
  async getPortfolioAllocation(): Promise<PortfolioAllocation> {
    try {
      const response: AxiosResponse<ApiResponse<PortfolioAllocation>> = await apiClient.get('/api/portfolio/allocation');
      return response.data.data;
    } catch (error) {
      console.warn('Portfolio API not available, returning mock data');
      return this.getMockPortfolioAllocation();
    }
  }

  async getPieChartData(): Promise<PieChartDataPoint[]> {
    try {
      const response: AxiosResponse<ApiResponse<PieChartDataPoint[]>> = await apiClient.get('/api/portfolio/pie-chart');
      return response.data.data;
    } catch (error) {
      console.warn('Pie Chart API not available, returning mock data');
      return this.getMockPieChartData();
    }
  }

  async getLineChartData(): Promise<LineChartDataPoint[]> {
    try {
      const response: AxiosResponse<ApiResponse<LineChartDataPoint[]>> = await apiClient.get('/api/portfolio/line-chart');
      return response.data.data;
    } catch (error) {
      console.warn('Line Chart API not available, returning mock data');
      return this.getMockLineChartData();
    }
  }

  async getComparisonChartData(): Promise<ComparisonChartDataPoint[]> {
    try {
      const response: AxiosResponse<ApiResponse<ComparisonChartDataPoint[]>> = await apiClient.get('/api/portfolio/comparison-chart');
      return response.data.data;
    } catch (error) {
      console.warn('Comparison Chart API not available, returning mock data');
      return this.getMockComparisonChartData();
    }
  }

  async getRiskVisualizationData(): Promise<RiskVisualizationData> {
    try {
      const response: AxiosResponse<ApiResponse<RiskVisualizationData>> = await apiClient.get('/api/portfolio/risk-visualization');
      return response.data.data;
    } catch (error) {
      console.warn('Risk Visualization API not available, returning mock data');
      return this.getMockRiskVisualizationData();
    }
  }

  // Mock Data Methods (for development/fallback)
  private getMockDashboardData(): DashboardData {
    return {
      system_status: 'HEALTHY',
      last_updated: new Date().toISOString(),
      summary: {
        total_log_entries: 1247,
        error_count: 3,
        warning_count: 12,
        performance_issues: 2,
        data_quality_issues: 1
      },
      component_activity: {
        'data_cleaning_agent': 456,
        'asset_predictor_agent': 234,
        'portfolio_allocator_agent': 189,
        'orchestrator': 156,
        'data_loader': 212
      },
      recent_performance: {
        total_operations: 89,
        successful_operations: 86,
        average_duration: 2.34
      },
      data_quality_status: {
        datasets_monitored: 5,
        average_quality_score: 94.2,
        datasets_with_issues: 1
      },
      error_summary: {
        total_errors: 3,
        error_types: ['ValidationError', 'TimeoutError'],
        components_with_errors: ['data_cleaning_agent']
      },
      recommendations: [
        'Consider optimizing slow data processing operations',
        'Review data quality for bond_returns_data dataset',
        'Monitor error rates in data_cleaning_agent component'
      ]
    };
  }

  private getMockSystemHealthData(): SystemHealthData {
    return {
      timestamp: new Date().toISOString(),
      system_status: 'HEALTHY',
      performance: {
        total_operations: 89,
        successful_operations: 86,
        average_duration: 2.34
      },
      data_quality: {
        datasets_monitored: 5,
        average_quality_score: 94.2,
        datasets_with_issues: 1
      },
      errors: {
        total_errors: 3,
        error_types: ['ValidationError', 'TimeoutError'],
        components_with_errors: ['data_cleaning_agent']
      }
    };
  }

  private getMockPerformanceSummary(): PerformanceSummary {
    return {
      'data_cleaning_agent': {
        total_operations: 45,
        avg_duration: 3.2,
        min_duration: 0.8,
        max_duration: 12.4,
        success_rate: 96.7
      },
      'asset_predictor_agent': {
        total_operations: 23,
        avg_duration: 1.8,
        min_duration: 0.5,
        max_duration: 4.2,
        success_rate: 100.0
      },
      'portfolio_allocator_agent': {
        total_operations: 18,
        avg_duration: 2.1,
        min_duration: 1.2,
        max_duration: 3.8,
        success_rate: 100.0
      }
    };
  }

  private getMockDataQualitySummary(): DataQualitySummary {
    return {
      datasets_monitored: 5,
      average_quality_score: 94.2,
      datasets: {
        'data_cleaning_agent_historical_returns': {
          quality_score: 98.5,
          completeness: 99.2,
          total_records: 1000,
          issues: {
            missing_values: 8,
            outliers: 3,
            validation_errors: 0
          }
        },
        'data_cleaning_agent_bond_returns': {
          quality_score: 87.3,
          completeness: 92.1,
          total_records: 800,
          issues: {
            missing_values: 63,
            outliers: 12,
            validation_errors: 2
          }
        }
      }
    };
  }

  private getMockErrorSummary(): ErrorSummary {
    return {
      total_errors: 3,
      errors_by_component: {
        'data_cleaning_agent': 2,
        'asset_predictor_agent': 1
      },
      errors_by_type: {
        'ValidationError': 2,
        'TimeoutError': 1
      },
      recent_errors: [
        {
          component: 'data_cleaning_agent',
          type: 'ValidationError',
          message: 'Invalid data format in column sp500',
          timestamp: new Date(Date.now() - 3600000).toISOString()
        },
        {
          component: 'asset_predictor_agent',
          type: 'TimeoutError',
          message: 'Prediction model timeout after 30 seconds',
          timestamp: new Date(Date.now() - 7200000).toISOString()
        }
      ]
    };
  }

  private getMockLogAnalysisReport(): LogAnalysisReport {
    return {
      analysis_period: [
        new Date(Date.now() - 24 * 3600000).toISOString(),
        new Date().toISOString()
      ],
      total_log_entries: 1247,
      error_count: 3,
      warning_count: 12,
      performance_issues: [
        {
          type: 'slow_operation',
          operation: 'data_cleaning_agent.clean_data',
          avg_duration: 8.2,
          max_duration: 15.3,
          occurrences: 5,
          severity: 'medium'
        }
      ],
      data_quality_issues: [
        {
          type: 'low_quality_score',
          dataset: 'bond_returns_data',
          quality_score: 87.3,
          completeness: 92.1,
          severity: 'medium'
        }
      ],
      component_activity: {
        'data_cleaning_agent': 456,
        'asset_predictor_agent': 234,
        'portfolio_allocator_agent': 189
      },
      recommendations: [
        'Consider optimizing slow data processing operations',
        'Review data quality for bond_returns_data dataset'
      ]
    };
  }

  private getMockPortfolioAllocation(): PortfolioAllocation {
    return {
      sp500: 45.0,
      small_cap: 20.0,
      bonds: 25.0,
      real_estate: 7.5,
      gold: 2.5
    };
  }

  private getMockPieChartData(): PieChartDataPoint[] {
    return [
      { name: 'S&P 500', value: 45.0, color: '#1f77b4', percentage: '45.0%' },
      { name: 'Bonds', value: 25.0, color: '#2ca02c', percentage: '25.0%' },
      { name: 'US Small Cap', value: 20.0, color: '#ff7f0e', percentage: '20.0%' },
      { name: 'Real Estate', value: 7.5, color: '#d62728', percentage: '7.5%' },
      { name: 'Gold', value: 2.5, color: '#9467bd', percentage: '2.5%' }
    ];
  }

  private getMockLineChartData(): LineChartDataPoint[] {
    const data: LineChartDataPoint[] = [];
    let value = 100000;
    
    for (let year = 0; year <= 10; year++) {
      const annualReturn = year === 0 ? 0 : (Math.random() * 0.15 + 0.05); // 5-20% return
      value = year === 0 ? value : value * (1 + annualReturn);
      const cumulativeReturn = year === 0 ? 0 : ((value / 100000 - 1) * 100);
      
      data.push({
        year,
        portfolio_value: Math.round(value),
        formatted_value: `$${Math.round(value).toLocaleString()}`,
        annual_return: Math.round(annualReturn * 100 * 100) / 100,
        cumulative_return: Math.round(cumulativeReturn * 100) / 100
      });
    }
    
    return data;
  }

  private getMockComparisonChartData(): ComparisonChartDataPoint[] {
    const data: ComparisonChartDataPoint[] = [];
    let portfolioValue = 100000;
    let benchmarkValue = 100000;
    
    for (let year = 0; year <= 10; year++) {
      const portfolioReturn = year === 0 ? 0 : (Math.random() * 0.15 + 0.06); // 6-21% return
      const benchmarkReturn = year === 0 ? 0 : 0.105; // 10.5% S&P 500 average
      
      portfolioValue = year === 0 ? portfolioValue : portfolioValue * (1 + portfolioReturn);
      benchmarkValue = year === 0 ? benchmarkValue : benchmarkValue * (1 + benchmarkReturn);
      
      const portfolioCumReturn = year === 0 ? 0 : ((portfolioValue / 100000 - 1) * 100);
      const benchmarkCumReturn = year === 0 ? 0 : ((benchmarkValue / 100000 - 1) * 100);
      
      data.push({
        year,
        portfolio_value: Math.round(portfolioValue),
        benchmark_value: Math.round(benchmarkValue),
        portfolio_return: Math.round(portfolioCumReturn * 100) / 100,
        benchmark_return: Math.round(benchmarkCumReturn * 100) / 100,
        outperformance: Math.round((portfolioCumReturn - benchmarkCumReturn) * 100) / 100
      });
    }
    
    return data;
  }

  private getMockRiskVisualizationData(): RiskVisualizationData {
    return {
      portfolio_metrics: [
        { metric: 'Volatility (%)', value: 12.4, benchmark: 16.0 },
        { metric: 'Sharpe Ratio', value: 0.89, benchmark: 0.65 },
        { metric: 'Max Drawdown (%)', value: 18.2, benchmark: 37.0 },
        { metric: 'Beta', value: 0.78, benchmark: 1.0 },
        { metric: 'Alpha (%)', value: 2.3, benchmark: 0.0 }
      ],
      risk_score: 32.5,
      risk_level: 'Moderate'
    };
  }

  private getMockPerformanceTrends(): any {
    return {
      'load_data': {
        hourly_data: Array.from({ length: 24 }, (_, i) => ({
          hour: new Date(Date.now() - (23 - i) * 3600000).toISOString(),
          avg_duration: Math.random() * 3 + 1,
          operation_count: Math.floor(Math.random() * 10) + 1
        })),
        total_operations: 156,
        overall_avg_duration: 2.1
      }
    };
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;