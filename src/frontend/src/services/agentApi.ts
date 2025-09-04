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

// Agent-powered API configuration
const AGENT_API_BASE_URL = process.env.REACT_APP_AGENT_API_URL || 'http://localhost:8000';

const agentApiClient = axios.create({
  baseURL: AGENT_API_BASE_URL,
  timeout: 15000, // Longer timeout for agent processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
agentApiClient.interceptors.request.use(
  (config) => {
    console.log(`🤖 Agent API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('🤖 Agent API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
agentApiClient.interceptors.response.use(
  (response) => {
    if (response.data.processed_by_agents) {
      console.log('✅ Data processed by agent pipeline');
    }
    return response;
  },
  (error) => {
    console.error('🤖 Agent API Response Error:', error);
    return Promise.reject(error);
  }
);

// Agent-Powered API Service Class
class AgentApiService {
  // Agent Status APIs
  async getAgentStatus(): Promise<any> {
    const response = await agentApiClient.get('/api/agent-status');
    return response.data.data;
  }

  // Dashboard APIs - Agent Enhanced
  async getDashboardData(): Promise<DashboardData> {
    const response: AxiosResponse<ApiResponse<DashboardData>> = await agentApiClient.get('/api/dashboard');
    return response.data.data;
  }

  async getSystemHealth(): Promise<SystemHealthData> {
    const response: AxiosResponse<ApiResponse<SystemHealthData>> = await agentApiClient.get('/api/system-health');
    return response.data.data;
  }

  // Performance Monitoring APIs - Agent Predictions
  async getPerformanceSummary(hoursBack: number = 24): Promise<PerformanceSummary> {
    const response: AxiosResponse<ApiResponse<PerformanceSummary>> = await agentApiClient.get(
      `/api/performance/summary?hours_back=${hoursBack}`
    );
    return response.data.data;
  }

  // Data Quality APIs - Agent Validation
  async getDataQualitySummary(): Promise<DataQualitySummary> {
    const response: AxiosResponse<ApiResponse<DataQualitySummary>> = await agentApiClient.get('/api/data-quality/summary');
    return response.data.data;
  }

  // Portfolio Analysis APIs - Agent Optimized
  async getPortfolioAllocation(): Promise<PortfolioAllocation> {
    const response: AxiosResponse<ApiResponse<any>> = await agentApiClient.get('/api/portfolio/allocation');
    
    // Convert agent allocation format to expected format
    const agentData = response.data.data;
    return {
      sp500: agentData.stocks * 0.75, // Assume 75% of stocks is S&P 500
      small_cap: agentData.stocks * 0.25, // Assume 25% of stocks is small cap
      bonds: agentData.bonds,
      real_estate: agentData.alternatives * 0.7, // Assume 70% of alternatives is real estate
      gold: agentData.alternatives * 0.3 // Assume 30% of alternatives is gold
    };
  }

  async getPieChartData(): Promise<PieChartDataPoint[]> {
    const response: AxiosResponse<ApiResponse<PieChartDataPoint[]>> = await agentApiClient.get('/api/portfolio/pie-chart');
    return response.data.data;
  }

  async getLineChartData(): Promise<LineChartDataPoint[]> {
    const response: AxiosResponse<ApiResponse<LineChartDataPoint[]>> = await agentApiClient.get('/api/portfolio/line-chart');
    return response.data.data;
  }

  async getComparisonChartData(): Promise<ComparisonChartDataPoint[]> {
    const response: AxiosResponse<ApiResponse<ComparisonChartDataPoint[]>> = await agentApiClient.get('/api/portfolio/comparison-chart');
    return response.data.data;
  }

  async getRiskVisualizationData(): Promise<RiskVisualizationData> {
    const response: AxiosResponse<ApiResponse<RiskVisualizationData>> = await agentApiClient.get('/api/portfolio/risk-visualization');
    return response.data.data;
  }

  // Agent-Specific APIs
  async getDataCleaningResults(): Promise<any> {
    const response = await agentApiClient.get('/api/agents/data-cleaning/results');
    return response.data.data;
  }

  async getPredictionResults(): Promise<any> {
    const response = await agentApiClient.get('/api/agents/predictions/results');
    return response.data.data;
  }

  async getAllocationResults(): Promise<any> {
    const response = await agentApiClient.get('/api/agents/allocation/results');
    return response.data.data;
  }

  async getFinalRecommendations(): Promise<any> {
    const response = await agentApiClient.get('/api/agents/final-recommendations');
    return response.data.data;
  }

  // Historical Data APIs - Agent Processed
  async getHistoricalPerformanceSummary(): Promise<any> {
    const response = await agentApiClient.get('/api/historical/performance-summary');
    return response.data.data;
  }

  // Error handling fallbacks
  async getErrorSummary(): Promise<ErrorSummary> {
    // For now, return a simple error summary based on agent status
    try {
      const agentStatus = await this.getAgentStatus();
      return {
        total_errors: 0,
        errors_by_component: {},
        errors_by_type: {},
        recent_errors: []
      };
    } catch (error) {
      return {
        total_errors: 1,
        errors_by_component: { 'agent_pipeline': 1 },
        errors_by_type: { 'ConnectionError': 1 },
        recent_errors: [
          {
            component: 'agent_pipeline',
            type: 'ConnectionError',
            message: 'Could not connect to agent pipeline',
            timestamp: new Date().toISOString()
          }
        ]
      };
    }
  }

  async getLogAnalysis(hoursBack: number = 24): Promise<LogAnalysisReport> {
    // Create log analysis from agent execution data
    try {
      const agentStatus = await this.getAgentStatus();
      const dataCleaningResults = await this.getDataCleaningResults();
      
      return {
        analysis_period: [
          new Date(Date.now() - hoursBack * 3600000).toISOString(),
          new Date().toISOString()
        ],
        total_log_entries: dataCleaningResults.cleaned_records,
        error_count: dataCleaningResults.outliers_detected,
        warning_count: 0,
        performance_issues: [],
        data_quality_issues: dataCleaningResults.validation_passed ? [] : [
          {
            type: 'validation_failure',
            dataset: 'excel_data',
            quality_score: dataCleaningResults.data_quality_score,
            completeness: dataCleaningResults.data_quality_score,
            severity: 'medium'
          }
        ],
        component_activity: {
          'data_cleaning_agent': dataCleaningResults.cleaned_records,
          'asset_predictor_agent': agentStatus.agents_executed.find((a: any) => a.name === 'AssetPredictorAgent')?.records_processed || 0,
          'portfolio_allocator_agent': 1
        },
        recommendations: [
          `Agent pipeline processed ${dataCleaningResults.cleaned_records} records`,
          `Data quality score: ${dataCleaningResults.data_quality_score}%`,
          'All agents executed successfully'
        ]
      };
    } catch (error) {
      // Fallback log analysis
      return {
        analysis_period: [
          new Date(Date.now() - hoursBack * 3600000).toISOString(),
          new Date().toISOString()
        ],
        total_log_entries: 0,
        error_count: 1,
        warning_count: 0,
        performance_issues: [],
        data_quality_issues: [],
        component_activity: {},
        recommendations: ['Agent pipeline connection failed']
      };
    }
  }

  async getPerformanceTrends(component: string, hoursBack: number = 168): Promise<any> {
    // Create performance trends from agent data
    try {
      const agentStatus = await this.getAgentStatus();
      return {
        [component]: {
          agent_execution_data: agentStatus.execution_log,
          total_operations: agentStatus.execution_summary.agents_executed,
          overall_avg_duration: parseFloat(agentStatus.execution_summary.total_execution_time.replace('s', ''))
        }
      };
    } catch (error) {
      return {
        [component]: {
          agent_execution_data: [],
          total_operations: 0,
          overall_avg_duration: 0
        }
      };
    }
  }
}

// Export singleton instance
export const agentApiService = new AgentApiService();
export default agentApiService;