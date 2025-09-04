import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { ErrorSummary } from '../types';
import { apiService } from '../services/api';
import MetricCard from '../components/Common/MetricCard';
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

const ErrorCard = styled(Card)`
  border-left: 4px solid #e53e3e;
  margin-bottom: 15px;
`;

const ErrorHeader = styled.div`
  display: flex;
  justify-content: between;
  align-items: flex-start;
  margin-bottom: 10px;
`;

const ErrorType = styled.h4`
  font-size: 1rem;
  font-weight: 600;
  color: #e53e3e;
  margin: 0;
  flex: 1;
`;

const ErrorTimestamp = styled.span`
  font-size: 0.75rem;
  color: #718096;
  background: #f7fafc;
  padding: 2px 8px;
  border-radius: 12px;
`;

const ErrorComponent = styled.div`
  font-size: 0.875rem;
  color: #4a5568;
  margin-bottom: 8px;
  font-weight: 500;
`;

const ErrorMessage = styled.div`
  font-size: 0.875rem;
  color: #2d3748;
  line-height: 1.4;
  background: #f7fafc;
  padding: 10px;
  border-radius: 4px;
  border-left: 3px solid #e2e8f0;
`;

const NoErrorsMessage = styled.div`
  text-align: center;
  padding: 40px;
  color: #718096;
  font-style: italic;
`;

const ErrorIcon = styled.span`
  font-size: 1.5rem;
  margin-bottom: 10px;
  display: block;
`;

const RefreshButton = styled.button`
  background: #3182ce;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 8px 16px;
  font-size: 0.875rem;
  cursor: pointer;
  transition: background-color 0.2s;
  margin-bottom: 20px;

  &:hover {
    background: #2c5aa0;
  }

  &:disabled {
    background: #a0aec0;
    cursor: not-allowed;
  }
`;

const ErrorMessage404 = styled.div`
  color: #e53e3e;
  background-color: #fed7d7;
  border: 1px solid #feb2b2;
  padding: 12px;
  border-radius: 4px;
  margin: 20px 0;
`;

const ErrorTracking: React.FC = () => {
  const [errorData, setErrorData] = useState<ErrorSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchErrorData = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getErrorSummary();
      setErrorData(data);
    } catch (err) {
      setError('Failed to load error tracking data');
      console.error('Error tracking fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchErrorData();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchErrorData, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const formatTimestamp = (timestamp: string): string => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  if (loading && !errorData) {
    return (
      <PageContainer>
        <PageTitle>Error Tracking</PageTitle>
        <LoadingSpinner size="large" />
      </PageContainer>
    );
  }

  if (error && !errorData) {
    return (
      <PageContainer>
        <PageTitle>Error Tracking</PageTitle>
        <ErrorMessage404>{error}</ErrorMessage404>
      </PageContainer>
    );
  }

  if (!errorData) {
    return (
      <PageContainer>
        <PageTitle>Error Tracking</PageTitle>
        <ErrorMessage404>No error tracking data available</ErrorMessage404>
      </PageContainer>
    );
  }

  // Prepare chart data with null checks
  const componentNames = errorData.errors_by_component ? Object.keys(errorData.errors_by_component) : [];
  const componentErrorCounts = errorData.errors_by_component ? Object.values(errorData.errors_by_component) : [];
  
  const errorTypeNames = errorData.errors_by_type ? Object.keys(errorData.errors_by_type) : [];
  const errorTypeCounts = errorData.errors_by_type ? Object.values(errorData.errors_by_type) : [];

  const componentErrorsData = {
    labels: componentNames.map(name => name.replace(/_/g, ' ')),
    datasets: [
      {
        label: 'Error Count',
        data: componentErrorCounts,
        backgroundColor: '#e53e3e',
        borderColor: '#c53030',
        borderWidth: 1
      }
    ]
  };

  const errorTypesData = {
    labels: errorTypeNames,
    datasets: [
      {
        label: 'Error Count',
        data: errorTypeCounts,
        backgroundColor: [
          '#e53e3e',
          '#d69e2e',
          '#9467bd',
          '#3182ce',
          '#38a169'
        ],
        borderWidth: 1
      }
    ]
  };

  // Calculate metrics
  const errorRate = errorData.total_errors > 0 ? 
    (errorData.total_errors / (errorData.total_errors + 100)) * 100 : 0; // Assuming 100 successful operations
  
  const mostProblematicComponent = componentNames.length > 0 ? 
    componentNames.reduce((a, b) => 
      errorData.errors_by_component[a] > errorData.errors_by_component[b] ? a : b
    ) : 'None';

  const mostCommonErrorType = errorTypeNames.length > 0 ?
    errorTypeNames.reduce((a, b) => 
      errorData.errors_by_type[a] > errorData.errors_by_type[b] ? a : b
    ) : 'None';

  return (
    <PageContainer>
      <PageTitle>Error Tracking</PageTitle>

      <RefreshButton onClick={fetchErrorData} disabled={loading}>
        {loading ? 'Refreshing...' : 'Refresh Data'}
      </RefreshButton>

      <MetricsGrid>
        <MetricCard
          title="Total Errors"
          value={errorData.total_errors}
          status={errorData.total_errors === 0 ? 'HEALTHY' : 
                  errorData.total_errors <= 5 ? 'WARNING' : 'CRITICAL'}
          loading={loading}
        />
        <MetricCard
          title="Error Types"
          value={errorTypeNames.length}
          loading={loading}
        />
        <MetricCard
          title="Components Affected"
          value={componentNames.length}
          loading={loading}
        />
        <MetricCard
          title="Most Common Error"
          value={mostCommonErrorType.replace(/([A-Z])/g, ' $1').trim()}
          loading={loading}
        />
      </MetricsGrid>

      {errorData.total_errors > 0 ? (
        <ChartsGrid>
          <Card>
            <BarChart
              data={componentErrorsData}
              title="Errors by Component"
              height={300}
              loading={loading}
            />
          </Card>

          <Card>
            <BarChart
              data={errorTypesData}
              title="Errors by Type"
              height={300}
              loading={loading}
            />
          </Card>
        </ChartsGrid>
      ) : null}

      <Card>
        <CardTitle>Recent Errors</CardTitle>
        {errorData.recent_errors.length > 0 ? (
          <div>
            {errorData.recent_errors.map((error, index) => (
              <ErrorCard key={index}>
                <ErrorHeader>
                  <ErrorType>{error.type.replace(/([A-Z])/g, ' $1').trim()}</ErrorType>
                  <ErrorTimestamp>{formatTimestamp(error.timestamp)}</ErrorTimestamp>
                </ErrorHeader>
                <ErrorComponent>
                  📍 {error.component.replace(/_/g, ' ')}
                </ErrorComponent>
                <ErrorMessage>
                  {error.message}
                </ErrorMessage>
              </ErrorCard>
            ))}
          </div>
        ) : (
          <NoErrorsMessage>
            <ErrorIcon>✅</ErrorIcon>
            No recent errors found. System is running smoothly!
          </NoErrorsMessage>
        )}
      </Card>
    </PageContainer>
  );
};

export default ErrorTracking;