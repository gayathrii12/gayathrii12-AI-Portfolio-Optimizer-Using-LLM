import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { DashboardData } from '../types';
import { apiService } from '../services/api';
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

const Dashboard: React.FC = () => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getDashboardData();
      setDashboardData(data);
    } catch (err) {
      setError('Failed to load dashboard data');
      console.error('Dashboard data fetch error:', err);
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
        <PageTitle>Dashboard</PageTitle>
        <LoadingSpinner size="large" />
      </DashboardContainer>
    );
  }

  if (error && !dashboardData) {
    return (
      <DashboardContainer>
        <PageTitle>Dashboard</PageTitle>
        <ErrorMessage>{error}</ErrorMessage>
      </DashboardContainer>
    );
  }

  if (!dashboardData) {
    return (
      <DashboardContainer>
        <PageTitle>Dashboard</PageTitle>
        <ErrorMessage>No dashboard data available</ErrorMessage>
      </DashboardContainer>
    );
  }

  // Prepare component activity chart data
  const componentActivityData = {
    labels: Object.keys(dashboardData.component_activity),
    datasets: [
      {
        label: 'Log Entries',
        data: Object.values(dashboardData.component_activity),
        backgroundColor: [
          '#3182ce',
          '#38a169',
          '#d69e2e',
          '#e53e3e',
          '#9467bd',
          '#8c564b'
        ],
        borderColor: [
          '#2c5aa0',
          '#2f855a',
          '#b7791f',
          '#c53030',
          '#805ad5',
          '#744210'
        ],
        borderWidth: 1
      }
    ]
  };

  return (
    <DashboardContainer>
      <StatusSection>
        <PageTitle>System Dashboard</PageTitle>
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
          title="Total Log Entries"
          value={dashboardData.summary.total_log_entries}
          loading={loading}
        />
        <MetricCard
          title="Error Count"
          value={dashboardData.summary.error_count}
          status={dashboardData.summary.error_count > 5 ? 'CRITICAL' : 
                  dashboardData.summary.error_count > 0 ? 'WARNING' : 'HEALTHY'}
          loading={loading}
        />
        <MetricCard
          title="Warning Count"
          value={dashboardData.summary.warning_count}
          status={dashboardData.summary.warning_count > 10 ? 'WARNING' : 'HEALTHY'}
          loading={loading}
        />
        <MetricCard
          title="Performance Issues"
          value={dashboardData.summary.performance_issues}
          status={dashboardData.summary.performance_issues > 0 ? 'WARNING' : 'HEALTHY'}
          loading={loading}
        />
        <MetricCard
          title="Data Quality Issues"
          value={dashboardData.summary.data_quality_issues}
          status={dashboardData.summary.data_quality_issues > 0 ? 'WARNING' : 'HEALTHY'}
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
      </MetricsGrid>

      <ChartsGrid>
        <Card>
          <BarChart
            data={componentActivityData}
            title="Component Activity (Last 24 Hours)"
            height={300}
            loading={loading}
          />
        </Card>

        <Card>
          <CardTitle>System Health Summary</CardTitle>
          <div style={{ marginBottom: '15px' }}>
            <strong>Performance:</strong>
            <ul style={{ margin: '5px 0', paddingLeft: '20px' }}>
              <li>Total Operations: {dashboardData.recent_performance.total_operations}</li>
              <li>Success Rate: {((dashboardData.recent_performance.successful_operations / 
                                  dashboardData.recent_performance.total_operations) * 100).toFixed(1)}%</li>
              <li>Average Duration: {dashboardData.recent_performance.average_duration.toFixed(2)}s</li>
            </ul>
          </div>
          
          <div style={{ marginBottom: '15px' }}>
            <strong>Data Quality:</strong>
            <ul style={{ margin: '5px 0', paddingLeft: '20px' }}>
              <li>Datasets Monitored: {dashboardData.data_quality_status.datasets_monitored}</li>
              <li>Average Quality Score: {dashboardData.data_quality_status.average_quality_score.toFixed(1)}%</li>
              <li>Datasets with Issues: {dashboardData.data_quality_status.datasets_with_issues}</li>
            </ul>
          </div>

          <div>
            <strong>Errors:</strong>
            <ul style={{ margin: '5px 0', paddingLeft: '20px' }}>
              <li>Total Errors: {dashboardData.error_summary.total_errors}</li>
              <li>Error Types: {dashboardData.error_summary.error_types.join(', ') || 'None'}</li>
              <li>Components with Errors: {dashboardData.error_summary.components_with_errors.join(', ') || 'None'}</li>
            </ul>
          </div>
        </Card>
      </ChartsGrid>

      <ChartsGrid>
        <Card>
          <CardTitle>Recommendations</CardTitle>
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
              No recommendations at this time. System is operating normally.
            </p>
          )}
        </Card>

        <Card>
          <CardTitle>Historical Data Status</CardTitle>
          <div style={{ marginBottom: '15px' }}>
            <strong>Dataset:</strong> S&P 500 Historical Returns (1927-2024)
          </div>
          <div style={{ marginBottom: '15px' }}>
            <strong>Data Points:</strong> 98 years of historical data
          </div>
          <div style={{ marginBottom: '15px' }}>
            <strong>Data Quality:</strong> 
            <span style={{ marginLeft: '8px' }}>
              <StatusBadge status="HEALTHY" size="small" />
            </span>
          </div>
          <div>
            <strong>Last Updated:</strong> {new Date().toLocaleDateString()}
          </div>
        </Card>
      </ChartsGrid>
    </DashboardContainer>
  );
};

export default Dashboard;