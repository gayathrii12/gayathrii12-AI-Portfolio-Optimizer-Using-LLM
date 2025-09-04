import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { SystemHealthData, LogAnalysisReport } from '../types';
import { apiService } from '../services/api';
import MetricCard from '../components/Common/MetricCard';
import StatusBadge from '../components/Common/StatusBadge';
import BarChart from '../components/Charts/BarChart';
import LoadingSpinner from '../components/Common/LoadingSpinner';

const PageContainer = styled.div`
  padding: 20px;
`;

const PageTitle = styled.h1`
  font-size: 2rem;
  font-weight: 700;
  color: #1a202c;
  margin: 0 0 30px;
`;

const StatusHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
  padding: 20px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const StatusInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 20px;
`;

const StatusText = styled.div`
  h2 {
    margin: 0 0 5px;
    font-size: 1.5rem;
    color: #1a202c;
  }
  
  p {
    margin: 0;
    color: #718096;
    font-size: 0.875rem;
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

const IssuesList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const IssueItem = styled.li<{ severity: string }>`
  padding: 12px;
  margin-bottom: 8px;
  border-radius: 4px;
  border-left: 4px solid ${props => {
    switch (props.severity) {
      case 'high': return '#e53e3e';
      case 'medium': return '#d69e2e';
      default: return '#38a169';
    }
  }};
  background: ${props => {
    switch (props.severity) {
      case 'high': return '#fed7d7';
      case 'medium': return '#fffbeb';
      default: return '#f0fff4';
    }
  }};
`;

const IssueType = styled.div`
  font-weight: 600;
  color: #1a202c;
  margin-bottom: 4px;
`;

const IssueDetails = styled.div`
  font-size: 0.875rem;
  color: #4a5568;
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
    content: '💡';
    margin-right: 8px;
  }
`;

const HealthIndicator = styled.div<{ status: string }>`
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 1.125rem;
  font-weight: 600;
  color: ${props => {
    switch (props.status) {
      case 'HEALTHY': return '#38a169';
      case 'WARNING': return '#d69e2e';
      case 'CRITICAL': return '#e53e3e';
      default: return '#718096';
    }
  }};
`;

const ErrorMessage = styled.div`
  color: #e53e3e;
  background-color: #fed7d7;
  border: 1px solid #feb2b2;
  padding: 12px;
  border-radius: 4px;
  margin: 20px 0;
`;

const SystemHealth: React.FC = () => {
  const [healthData, setHealthData] = useState<SystemHealthData | null>(null);
  const [analysisData, setAnalysisData] = useState<LogAnalysisReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHealthData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [health, analysis] = await Promise.all([
        apiService.getSystemHealth(),
        apiService.getLogAnalysis(24)
      ]);
      
      setHealthData(health);
      setAnalysisData(analysis);
    } catch (err) {
      setError('Failed to load system health data');
      console.error('System health fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealthData();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchHealthData, 30000);
    
    return () => clearInterval(interval);
  }, []);

  if (loading && !healthData) {
    return (
      <PageContainer>
        <PageTitle>System Health</PageTitle>
        <LoadingSpinner size="large" />
      </PageContainer>
    );
  }

  if (error && !healthData) {
    return (
      <PageContainer>
        <PageTitle>System Health</PageTitle>
        <ErrorMessage>{error}</ErrorMessage>
      </PageContainer>
    );
  }

  if (!healthData || !analysisData) {
    return (
      <PageContainer>
        <PageTitle>System Health</PageTitle>
        <ErrorMessage>No system health data available</ErrorMessage>
      </PageContainer>
    );
  }

  // Prepare chart data
  const componentActivity = Object.keys(analysisData.component_activity);
  const activityCounts = Object.values(analysisData.component_activity);

  const activityData = {
    labels: componentActivity.map(name => name.replace(/_/g, ' ')),
    datasets: [
      {
        label: 'Activity Count',
        data: activityCounts,
        backgroundColor: '#3182ce',
        borderColor: '#2c5aa0',
        borderWidth: 1
      }
    ]
  };

  const issuesData = {
    labels: ['Performance Issues', 'Data Quality Issues', 'Errors', 'Warnings'],
    datasets: [
      {
        label: 'Issue Count',
        data: [
          analysisData.performance_issues.length,
          analysisData.data_quality_issues.length,
          analysisData.error_count,
          analysisData.warning_count
        ],
        backgroundColor: ['#e53e3e', '#d69e2e', '#9467bd', '#3182ce'],
        borderWidth: 1
      }
    ]
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'HEALTHY': return '✅';
      case 'WARNING': return '⚠️';
      case 'CRITICAL': return '🚨';
      default: return '❓';
    }
  };

  const formatTimestamp = (timestamp: string): string => {
    if (!timestamp) return 'Unknown';
    const date = new Date(timestamp);
    return isNaN(date.getTime()) ? 'Just now' : date.toLocaleString();
  };

  return (
    <PageContainer>
      <PageTitle>System Health</PageTitle>

      <StatusHeader>
        <StatusInfo>
          <StatusBadge status={healthData.system_status} size="large" />
          <StatusText>
            <h2>System Status</h2>
            <p>Last updated: {formatTimestamp(healthData.timestamp)}</p>
          </StatusText>
        </StatusInfo>
        <HealthIndicator status={healthData.system_status}>
          {getStatusIcon(healthData.system_status)}
          {healthData.system_status}
        </HealthIndicator>
      </StatusHeader>

      <MetricsGrid>
        <MetricCard
          title="Total Operations"
          value={healthData.performance.total_operations}
          loading={loading}
        />
        <MetricCard
          title="Success Rate"
          value={`${((healthData.performance.successful_operations / healthData.performance.total_operations) * 100).toFixed(1)}%`}
          status={((healthData.performance.successful_operations / healthData.performance.total_operations) * 100) >= 95 ? 'HEALTHY' : 'WARNING'}
          loading={loading}
        />
        <MetricCard
          title="Average Duration"
          value={`${healthData.performance.average_duration.toFixed(2)}s`}
          status={healthData.performance.average_duration <= 2 ? 'HEALTHY' : 'WARNING'}
          loading={loading}
        />
        <MetricCard
          title="Data Quality Score"
          value={`${healthData.data_quality.average_quality_score.toFixed(1)}%`}
          status={healthData.data_quality.average_quality_score >= 90 ? 'HEALTHY' : 
                  healthData.data_quality.average_quality_score >= 70 ? 'WARNING' : 'CRITICAL'}
          loading={loading}
        />
        <MetricCard
          title="Active Errors"
          value={healthData.errors.total_errors}
          status={healthData.errors.total_errors === 0 ? 'HEALTHY' : 
                  healthData.errors.total_errors <= 3 ? 'WARNING' : 'CRITICAL'}
          loading={loading}
        />
        <MetricCard
          title="Datasets Monitored"
          value={healthData.data_quality.datasets_monitored}
          loading={loading}
        />
      </MetricsGrid>

      <ChartsGrid>
        <Card>
          <BarChart
            data={activityData}
            title="Component Activity (Last 24 Hours)"
            height={300}
            loading={loading}
          />
        </Card>

        <Card>
          <BarChart
            data={issuesData}
            title="Issues Overview"
            height={300}
            loading={loading}
          />
        </Card>
      </ChartsGrid>

      <ChartsGrid>
        <Card>
          <CardTitle>Performance Issues</CardTitle>
          {analysisData.performance_issues.length > 0 ? (
            <IssuesList>
              {analysisData.performance_issues.map((issue, index) => (
                <IssueItem key={index} severity={issue.severity}>
                  <IssueType>
                    {issue.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </IssueType>
                  <IssueDetails>
                    {issue.operation && `Operation: ${issue.operation}`}
                    {issue.avg_duration && ` • Avg Duration: ${issue.avg_duration.toFixed(2)}s`}
                    {issue.failure_rate && ` • Failure Rate: ${issue.failure_rate.toFixed(1)}%`}
                  </IssueDetails>
                </IssueItem>
              ))}
            </IssuesList>
          ) : (
            <p style={{ color: '#718096', fontStyle: 'italic' }}>
              No performance issues detected.
            </p>
          )}
        </Card>

        <Card>
          <CardTitle>Data Quality Issues</CardTitle>
          {analysisData.data_quality_issues.length > 0 ? (
            <IssuesList>
              {analysisData.data_quality_issues.map((issue, index) => (
                <IssueItem key={index} severity={issue.severity}>
                  <IssueType>
                    {issue.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </IssueType>
                  <IssueDetails>
                    {issue.dataset && `Dataset: ${issue.dataset.replace(/_/g, ' ')}`}
                    {issue.quality_score && ` • Quality Score: ${issue.quality_score.toFixed(1)}%`}
                    {issue.missing_percentage && ` • Missing: ${issue.missing_percentage.toFixed(1)}%`}
                  </IssueDetails>
                </IssueItem>
              ))}
            </IssuesList>
          ) : (
            <p style={{ color: '#718096', fontStyle: 'italic' }}>
              No data quality issues detected.
            </p>
          )}
        </Card>
      </ChartsGrid>

      <Card>
        <CardTitle>System Recommendations</CardTitle>
        {analysisData.recommendations.length > 0 ? (
          <RecommendationsList>
            {analysisData.recommendations.map((recommendation, index) => (
              <RecommendationItem key={index}>
                {recommendation}
              </RecommendationItem>
            ))}
          </RecommendationsList>
        ) : (
          <p style={{ color: '#718096', fontStyle: 'italic' }}>
            No recommendations at this time. System is operating optimally.
          </p>
        )}
      </Card>
    </PageContainer>
  );
};

export default SystemHealth;