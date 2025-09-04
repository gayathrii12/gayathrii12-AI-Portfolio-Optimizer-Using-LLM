import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { PerformanceSummary } from '../types';
import { apiService } from '../services/api';
import MetricCard from '../components/Common/MetricCard';
import BarChart from '../components/Charts/BarChart';
import LineChart from '../components/Charts/LineChart';
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

const TabsContainer = styled.div`
  display: flex;
  border-bottom: 2px solid #e2e8f0;
  margin-bottom: 20px;
`;

const Tab = styled.button<{ active: boolean }>`
  padding: 12px 24px;
  background: none;
  border: none;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  color: ${props => props.active ? '#3182ce' : '#718096'};
  border-bottom: 2px solid ${props => props.active ? '#3182ce' : 'transparent'};
  transition: all 0.2s;

  &:hover {
    color: #4a5568;
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

const ComponentSelector = styled.select`
  padding: 8px 12px;
  border: 1px solid #e2e8f0;
  border-radius: 4px;
  font-size: 14px;
  margin-bottom: 20px;
  background: white;
`;

const ErrorMessage = styled.div`
  color: #e53e3e;
  background-color: #fed7d7;
  border: 1px solid #feb2b2;
  padding: 12px;
  border-radius: 4px;
  margin: 20px 0;
`;

const PerformanceTable = styled.table`
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
`;

const TableHeader = styled.th`
  padding: 12px;
  text-align: left;
  border-bottom: 2px solid #e2e8f0;
  background-color: #f7fafc;
  font-weight: 600;
  color: #4a5568;
`;

const TableCell = styled.td`
  padding: 12px;
  border-bottom: 1px solid #e2e8f0;
`;

const TableRow = styled.tr`
  &:hover {
    background-color: #f7fafc;
  }
`;

const PerformanceMonitoring: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'trends' | 'details'>('overview');
  const [performanceData, setPerformanceData] = useState<PerformanceSummary | null>(null);
  const [selectedComponent, setSelectedComponent] = useState<string>('');
  const [trendsData, setTrendsData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPerformanceData = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getPerformanceSummary(24);
      setPerformanceData(data);
      
      // Set default component if not selected
      if (!selectedComponent && Object.keys(data).length > 0) {
        setSelectedComponent(Object.keys(data)[0]);
      }
    } catch (err) {
      setError('Failed to load performance data');
      console.error('Performance data fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchTrendsData = async (component: string) => {
    if (!component) return;
    
    try {
      const data = await apiService.getPerformanceTrends(component, 168);
      setTrendsData(data);
    } catch (err) {
      console.error('Trends data fetch error:', err);
    }
  };

  useEffect(() => {
    fetchPerformanceData();
  }, []);

  useEffect(() => {
    if (selectedComponent && activeTab === 'trends') {
      fetchTrendsData(selectedComponent);
    }
  }, [selectedComponent, activeTab]);

  if (loading && !performanceData) {
    return (
      <PageContainer>
        <PageTitle>Performance Monitoring</PageTitle>
        <LoadingSpinner size="large" />
      </PageContainer>
    );
  }

  if (error && !performanceData) {
    return (
      <PageContainer>
        <PageTitle>Performance Monitoring</PageTitle>
        <ErrorMessage>{error}</ErrorMessage>
      </PageContainer>
    );
  }

  if (!performanceData) {
    return (
      <PageContainer>
        <PageTitle>Performance Monitoring</PageTitle>
        <ErrorMessage>No performance data available</ErrorMessage>
      </PageContainer>
    );
  }

  // Prepare chart data
  const components = Object.keys(performanceData);
  const avgDurationData = {
    labels: components,
    datasets: [
      {
        label: 'Average Duration (seconds)',
        data: components.map(comp => performanceData[comp].avg_duration),
        backgroundColor: '#3182ce',
        borderColor: '#2c5aa0',
        borderWidth: 1
      }
    ]
  };

  const successRateData = {
    labels: components,
    datasets: [
      {
        label: 'Success Rate (%)',
        data: components.map(comp => performanceData[comp].success_rate),
        backgroundColor: components.map(comp => 
          performanceData[comp].success_rate >= 95 ? '#38a169' : 
          performanceData[comp].success_rate >= 90 ? '#d69e2e' : '#e53e3e'
        ),
        borderWidth: 1
      }
    ]
  };

  const operationsCountData = {
    labels: components,
    datasets: [
      {
        label: 'Total Operations',
        data: components.map(comp => performanceData[comp].total_operations),
        backgroundColor: '#9467bd',
        borderColor: '#805ad5',
        borderWidth: 1
      }
    ]
  };

  // Calculate overall metrics
  const totalOperations = components.reduce((sum, comp) => sum + performanceData[comp].total_operations, 0);
  const avgSuccessRate = components.reduce((sum, comp) => sum + performanceData[comp].success_rate, 0) / components.length;
  const avgDuration = components.reduce((sum, comp) => sum + performanceData[comp].avg_duration, 0) / components.length;
  const slowestComponent = components.reduce((slowest, comp) => 
    performanceData[comp].avg_duration > performanceData[slowest].avg_duration ? comp : slowest
  );

  const renderOverview = () => (
    <>
      <MetricsGrid>
        <MetricCard
          title="Total Operations"
          value={totalOperations}
          loading={loading}
        />
        <MetricCard
          title="Average Success Rate"
          value={`${avgSuccessRate.toFixed(1)}%`}
          status={avgSuccessRate >= 95 ? 'HEALTHY' : avgSuccessRate >= 90 ? 'WARNING' : 'CRITICAL'}
          loading={loading}
        />
        <MetricCard
          title="Average Duration"
          value={`${avgDuration.toFixed(2)}s`}
          status={avgDuration <= 2 ? 'HEALTHY' : avgDuration <= 5 ? 'WARNING' : 'CRITICAL'}
          loading={loading}
        />
        <MetricCard
          title="Slowest Component"
          value={slowestComponent.replace(/_/g, ' ')}
          status={performanceData[slowestComponent].avg_duration <= 2 ? 'HEALTHY' : 'WARNING'}
          loading={loading}
        />
      </MetricsGrid>

      <ChartsGrid>
        <Card>
          <BarChart
            data={avgDurationData}
            title="Average Duration by Component"
            height={300}
            loading={loading}
          />
        </Card>

        <Card>
          <BarChart
            data={successRateData}
            title="Success Rate by Component"
            height={300}
            loading={loading}
          />
        </Card>

        <Card>
          <BarChart
            data={operationsCountData}
            title="Total Operations by Component"
            height={300}
            loading={loading}
          />
        </Card>
      </ChartsGrid>
    </>
  );

  const renderTrends = () => {
    const trendChartData = trendsData && selectedComponent && trendsData[Object.keys(trendsData)[0]] ? 
      trendsData[Object.keys(trendsData)[0]].hourly_data.map((item: any, index: number) => ({
        year: index,
        portfolio_value: item.avg_duration,
        formatted_value: `${item.avg_duration.toFixed(2)}s`,
        annual_return: item.operation_count
      })) : [];

    return (
      <>
        <ComponentSelector
          value={selectedComponent}
          onChange={(e) => setSelectedComponent(e.target.value)}
        >
          {components.map(comp => (
            <option key={comp} value={comp}>
              {comp.replace(/_/g, ' ')}
            </option>
          ))}
        </ComponentSelector>

        <ChartsGrid>
          <Card>
            <LineChart
              data={trendChartData}
              title={`Performance Trends - ${selectedComponent.replace(/_/g, ' ')}`}
              height={400}
              loading={loading}
            />
          </Card>
        </ChartsGrid>
      </>
    );
  };

  const renderDetails = () => (
    <Card>
      <CardTitle>Performance Details</CardTitle>
      <PerformanceTable>
        <thead>
          <tr>
            <TableHeader>Component</TableHeader>
            <TableHeader>Total Operations</TableHeader>
            <TableHeader>Avg Duration</TableHeader>
            <TableHeader>Min Duration</TableHeader>
            <TableHeader>Max Duration</TableHeader>
            <TableHeader>Success Rate</TableHeader>
          </tr>
        </thead>
        <tbody>
          {components.map(comp => (
            <TableRow key={comp}>
              <TableCell style={{ fontWeight: 600 }}>
                {comp.replace(/_/g, ' ')}
              </TableCell>
              <TableCell>{performanceData[comp].total_operations}</TableCell>
              <TableCell>{performanceData[comp].avg_duration.toFixed(2)}s</TableCell>
              <TableCell>{performanceData[comp].min_duration.toFixed(2)}s</TableCell>
              <TableCell>{performanceData[comp].max_duration.toFixed(2)}s</TableCell>
              <TableCell>
                <span style={{ 
                  color: performanceData[comp].success_rate >= 95 ? '#38a169' : 
                         performanceData[comp].success_rate >= 90 ? '#d69e2e' : '#e53e3e'
                }}>
                  {performanceData[comp].success_rate.toFixed(1)}%
                </span>
              </TableCell>
            </TableRow>
          ))}
        </tbody>
      </PerformanceTable>
    </Card>
  );

  return (
    <PageContainer>
      <PageTitle>Performance Monitoring</PageTitle>

      <TabsContainer>
        <Tab 
          active={activeTab === 'overview'} 
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </Tab>
        <Tab 
          active={activeTab === 'trends'} 
          onClick={() => setActiveTab('trends')}
        >
          Trends
        </Tab>
        <Tab 
          active={activeTab === 'details'} 
          onClick={() => setActiveTab('details')}
        >
          Details
        </Tab>
      </TabsContainer>

      {activeTab === 'overview' && renderOverview()}
      {activeTab === 'trends' && renderTrends()}
      {activeTab === 'details' && renderDetails()}
    </PageContainer>
  );
};

export default PerformanceMonitoring;