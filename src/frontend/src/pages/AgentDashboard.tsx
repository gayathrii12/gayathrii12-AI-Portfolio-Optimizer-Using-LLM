import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { DashboardData } from '../types';
import { agentApiService } from '../services/agentApi';
import MetricCard from '../components/Common/MetricCard';
import StatusBadge from '../components/Common/StatusBadge';
import BarChart from '../components/Charts/BarChart';
import LoadingSpinner from '../components/Common/LoadingSpinner';

const DashboardContainer = styled.div`
  padding: 20px;
`;

const PageTitle = styled.h1`
  font-size: 2rem;
  font-weight: 700;
  color: #1a202c;
  margin: 0 0 30px;
`;

const AgentBadge = styled.div`
  display: inline-flex;
  align-items: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 8px 16px;
  border-radius: 20px;
  font-size: 0.875rem;
  font-weight: 600;
  margin-left: 10px;
  
  &:before {
    content: '🤖';
    margin-right: 6px;
  }
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
`;

const ChartsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
`;

const Card = styled.div`
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  padding: 20px;
`;

const CardTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 15px;
`;

const AgentCard = styled(Card)`
  border-left: 4px solid #667eea;
`;

const AgentName = styled.div`
  font-weight: 600;
  color: #667eea;
  margin-bottom: 8px;
`;

const AgentMetric = styled.div`
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
  font-size: 0.875rem;
`;

const RecommendationsList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const RecommendationItem = styled.li`
  padding: 10px 0;
  border-bottom: 1px solid #e2e8f0;
  color: #4a5568;
  
  &:last-child {
    border-bottom: none;
  }
  
  &:before {
    content: '🤖';
    margin-right: 8px;
  }
`;

const StatusSection = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
`;

const LastUpdated = styled.span`
  font-size: 0.875rem;
  color: #718096;
`;

const ErrorMessage = styled.div`
  color: #e53e3e;
  background-color: #fed7d7;
  border: 1px solid #feb2b2;
  padding: 12px;
  border-radius: 4px;
  margin: 20px 0;
`;

interface AgentStatus {
  pipeline_status: string;
  execution_summary: {
    total_execution_time: string;
    agents_executed: number;
    data_source: string;
    processing_timestamp: string;
  };
  agents_executed: Array<{
    name: string;
    status: string;
    quality_score?: number;
    records_processed?: number;
    expected_return?: number;
    market_regime?: string;
    risk_level?: string;
    allocation_method?: string;
  }>;
}

const AgentDashboard: React.FC = () => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Fetch both dashboard data and agent status
      const [dashData, agentData] = await Promise.all([
        agentApiService.getDashboardData(),
        agentApiService.getAgentStatus()
      ]);
      
      setDashboardData(dashData);
      setAgentStatus(agentData);
    } catch (err) {
      setError('Failed to load agent dashboard data');
      console.error('Agent dashboard data fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    
    return () => clearInterval(interval);
  }, []);

  if (loading && !dashboardData) {
    return (
      <DashboardContainer>
        <PageTitle>Agent-Powered Dashboard</PageTitle>
        <LoadingSpinner size="large" />
      </DashboardContainer>
    );
  }

  if (error && !dashboardData) {
    return (
      <DashboardContainer>
        <PageTitle>Agent-Powered Dashboard</PageTitle>
        <ErrorMessage>{error}</ErrorMessage>
      </DashboardContainer>
    );
  }

  if (!dashboardData || !agentStatus) {
    return (
      <DashboardContainer>
        <PageTitle>Agent-Powered Dashboard</PageTitle>
        <ErrorMessage>No agent dashboard data available</ErrorMessage>
      </DashboardContainer>
    );
  }

  // Prepare component activity chart data
  const componentActivityData = {
    labels: Object.keys(dashboardData.component_activity),
    datasets: [
      {
        label: 'Records Processed by Agents',
        data: Object.values(dashboardData.component_activity),
        backgroundColor: [
          '#667eea',
          '#764ba2',
          '#f093fb',
          '#f5576c',
          '#4facfe'
        ],
        borderColor: [
          '#5a67d8',
          '#6b46c1',
          '#e879f9',
          '#ef4444',
          '#3b82f6'
        ],
        borderWidth: 2
      }
    ]
  };

  return (
    <DashboardContainer>
      <StatusSection>
        <div>
          <PageTitle>Agent-Powered Dashboard</PageTitle>
          <AgentBadge>Pipeline Status: {agentStatus.pipeline_status}</AgentBadge>
        </div>
        <div>
          <StatusBadge status={dashboardData.system_status} size="large" />
          <LastUpdated>
            Last updated: {dashboardData.last_updated ? 
              new Date(dashboardData.last_updated).toLocaleTimeString() : 
              new Date().toLocaleTimeString()}
          </LastUpdated>
        </div>
      </StatusSection>

      <MetricsGrid>
        <MetricCard
          title="Records Processed"
          value={dashboardData.summary.total_log_entries}
          loading={loading}
        />
        <MetricCard
          title="Agents Executed"
          value={agentStatus.execution_summary.agents_executed}
          status="HEALTHY"
          loading={loading}
        />
        <MetricCard
          title="Processing Time"
          value={agentStatus.execution_summary.total_execution_time}
          loading={loading}
        />
        <MetricCard
          title="Data Quality Score"
          value={`${dashboardData.data_quality_status.average_quality_score.toFixed(1)}%`}
          status={dashboardData.data_quality_status.average_quality_score >= 95 ? 'HEALTHY' : 'WARNING'}
          loading={loading}
        />
        <MetricCard
          title="Success Rate"
          value={`${((dashboardData.recent_performance.successful_operations / 
                     dashboardData.recent_performance.total_operations) * 100).toFixed(1)}%`}
          status={((dashboardData.recent_performance.successful_operations / 
                   dashboardData.recent_performance.total_operations) * 100) >= 95 ? 'HEALTHY' : 'WARNING'}
          loading={loading}
        />
        <MetricCard
          title="Outliers Detected"
          value={dashboardData.summary.error_count}
          status={dashboardData.summary.error_count > 5 ? 'CRITICAL' : 
                  dashboardData.summary.error_count > 0 ? 'WARNING' : 'HEALTHY'}
          loading={loading}
        />
      </MetricsGrid>

      <ChartsGrid>
        <Card>
          <BarChart
            data={componentActivityData}
            title="Agent Processing Activity"
            height={300}
            loading={loading}
          />
        </Card>

        <Card>
          <CardTitle>Agent Pipeline Execution</CardTitle>
          <div style={{ marginBottom: '15px' }}>
            <strong>Execution Summary:</strong>
            <ul style={{ margin: '5px 0', paddingLeft: '20px' }}>
              <li>Total Time: {agentStatus.execution_summary.total_execution_time}</li>
              <li>Agents: {agentStatus.execution_summary.agents_executed}</li>
              <li>Data Source: {agentStatus.execution_summary.data_source}</li>
              <li>Processed: {agentStatus.execution_summary.processing_timestamp ? 
                (() => {
                  const date = new Date(agentStatus.execution_summary.processing_timestamp);
                  return isNaN(date.getTime()) ? 'Just now' : date.toLocaleString();
                })() : 'Unknown'}</li>
            </ul>
          </div>
        </Card>
      </ChartsGrid>

      <ChartsGrid>
        {agentStatus.agents_executed.map((agent, index) => (
          <AgentCard key={index}>
            <AgentName>{agent.name}</AgentName>
            <AgentMetric>
              <span>Status:</span>
              <StatusBadge status={agent.status === 'completed' ? 'HEALTHY' : 'WARNING'} size="small" />
            </AgentMetric>
            
            {agent.quality_score && (
              <AgentMetric>
                <span>Quality Score:</span>
                <span>{agent.quality_score}%</span>
              </AgentMetric>
            )}
            
            {agent.records_processed && (
              <AgentMetric>
                <span>Records Processed:</span>
                <span>{agent.records_processed}</span>
              </AgentMetric>
            )}
            
            {agent.expected_return && (
              <AgentMetric>
                <span>Expected Return:</span>
                <span>{agent.expected_return}%</span>
              </AgentMetric>
            )}
            
            {agent.market_regime && (
              <AgentMetric>
                <span>Market Regime:</span>
                <span>{agent.market_regime}</span>
              </AgentMetric>
            )}
            
            {agent.risk_level && (
              <AgentMetric>
                <span>Risk Level:</span>
                <span>{agent.risk_level}</span>
              </AgentMetric>
            )}
            
            {agent.allocation_method && (
              <AgentMetric>
                <span>Method:</span>
                <span>{agent.allocation_method}</span>
              </AgentMetric>
            )}
          </AgentCard>
        ))}

        <Card>
          <CardTitle>Agent Recommendations</CardTitle>
          {dashboardData.recommendations.length > 0 ? (
            <RecommendationsList>
              {dashboardData.recommendations.map((recommendation, index) => (
                <RecommendationItem key={index}>
                  {recommendation}
                </RecommendationItem>
              ))}
            </RecommendationsList>
          ) : (
            <p style={{ color: '#718096', fontStyle: 'italic' }}>
              No recommendations from agents at this time.
            </p>
          )}
        </Card>
      </ChartsGrid>
    </DashboardContainer>
  );
};

export default AgentDashboard;