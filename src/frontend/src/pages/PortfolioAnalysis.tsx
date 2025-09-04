import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { 
  PortfolioAllocation, 
  PieChartDataPoint, 
  LineChartDataPoint, 
  ComparisonChartDataPoint,
  RiskVisualizationData 
} from '../types';
import { apiService } from '../services/api';
import PieChart from '../components/Charts/PieChart';
import LineChart from '../components/Charts/LineChart';
import BarChart from '../components/Charts/BarChart';
import MetricCard from '../components/Common/MetricCard';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import PerformanceSummary from '../components/Historical/PerformanceSummary';

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

const ChartsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
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

const AllocationTable = styled.table`
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

const ColorDot = styled.span<{ color: string }>`
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background-color: ${props => props.color};
  margin-right: 8px;
`;

const ErrorMessage = styled.div`
  color: #e53e3e;
  background-color: #fed7d7;
  border: 1px solid #feb2b2;
  padding: 12px;
  border-radius: 4px;
  margin: 20px 0;
`;

const RiskLevelBadge = styled.span<{ level: string }>`
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  
  ${props => {
    switch (props.level.toLowerCase()) {
      case 'low':
        return `
          color: #38a169;
          background-color: #f0fff4;
          border: 1px solid #9ae6b4;
        `;
      case 'moderate':
        return `
          color: #d69e2e;
          background-color: #fffbeb;
          border: 1px solid #fbd38d;
        `;
      case 'high':
        return `
          color: #e53e3e;
          background-color: #fed7d7;
          border: 1px solid #feb2b2;
        `;
      default:
        return `
          color: #718096;
          background-color: #f7fafc;
          border: 1px solid #e2e8f0;
        `;
    }
  }}
`;

const PortfolioAnalysis: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'allocation' | 'performance' | 'comparison' | 'risk'>('allocation');
  const [pieChartData, setPieChartData] = useState<PieChartDataPoint[]>([]);
  const [lineChartData, setLineChartData] = useState<LineChartDataPoint[]>([]);
  const [comparisonData, setComparisonData] = useState<ComparisonChartDataPoint[]>([]);
  const [riskData, setRiskData] = useState<RiskVisualizationData | null>(null);
  const [allocation, setAllocation] = useState<PortfolioAllocation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPortfolioData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [pieData, lineData, compData, riskVisualization, allocationData] = await Promise.all([
        apiService.getPieChartData(),
        apiService.getLineChartData(),
        apiService.getComparisonChartData(),
        apiService.getRiskVisualizationData(),
        apiService.getPortfolioAllocation()
      ]);
      
      setPieChartData(pieData);
      setLineChartData(lineData);
      setComparisonData(compData);
      setRiskData(riskVisualization);
      setAllocation(allocationData);
    } catch (err) {
      setError('Failed to load portfolio data');
      console.error('Portfolio data fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPortfolioData();
  }, []);

  if (loading) {
    return (
      <PageContainer>
        <PageTitle>Portfolio Analysis</PageTitle>
        <LoadingSpinner size="large" />
      </PageContainer>
    );
  }

  if (error) {
    return (
      <PageContainer>
        <PageTitle>Portfolio Analysis</PageTitle>
        <ErrorMessage>{error}</ErrorMessage>
      </PageContainer>
    );
  }

  const renderAllocation = () => {
    const totalValue = lineChartData.length > 0 ? lineChartData[lineChartData.length - 1].portfolio_value : 100000;
    const initialValue = lineChartData.length > 0 ? lineChartData[0].portfolio_value : 100000;
    const totalReturn = lineChartData.length > 0 ? 
      ((totalValue - initialValue) / initialValue * 100) : 0;

    return (
      <>
        <MetricsGrid>
          <MetricCard
            title="Current Portfolio Value"
            value={`$${totalValue.toLocaleString()}`}
            loading={loading}
          />
          <MetricCard
            title="Total Return"
            value={`${totalReturn.toFixed(1)}%`}
            change={{
              value: totalReturn,
              type: totalReturn >= 0 ? 'positive' : 'negative'
            }}
            status={totalReturn >= 10 ? 'HEALTHY' : totalReturn >= 0 ? 'WARNING' : 'CRITICAL'}
            loading={loading}
          />
          <MetricCard
            title="Asset Classes"
            value={pieChartData.length}
            loading={loading}
          />
          <MetricCard
            title="Largest Allocation"
            value={pieChartData.length > 0 ? pieChartData[0].name : 'N/A'}
            loading={loading}
          />
        </MetricsGrid>

        <ChartsGrid>
          <Card>
            <PieChart
              data={pieChartData}
              title="Portfolio Allocation"
              height={400}
              loading={loading}
            />
          </Card>

          <Card>
            <CardTitle>Allocation Details</CardTitle>
            <AllocationTable>
              <thead>
                <tr>
                  <TableHeader>Asset Class</TableHeader>
                  <TableHeader>Allocation</TableHeader>
                  <TableHeader>Value</TableHeader>
                </tr>
              </thead>
              <tbody>
                {pieChartData.map((item, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      <ColorDot color={item.color} />
                      {item.name}
                    </TableCell>
                    <TableCell>{item.percentage}</TableCell>
                    <TableCell>
                      ${Math.round(totalValue * item.value / 100).toLocaleString()}
                    </TableCell>
                  </TableRow>
                ))}
              </tbody>
            </AllocationTable>
          </Card>
        </ChartsGrid>
      </>
    );
  };

  const renderPerformance = () => (
    <>
      <PerformanceSummary />
      <ChartsGrid>
        <Card style={{ gridColumn: '1 / -1' }}>
          <LineChart
            data={lineChartData}
            title="S&P 500 Historical Performance (1927-2024)"
            height={400}
            showReturns={true}
            loading={loading}
          />
        </Card>
      </ChartsGrid>
    </>
  );

  const renderComparison = () => {
    const comparisonChartData = {
      labels: comparisonData.map(item => `Year ${item.year}`),
      datasets: [
        {
          label: 'Portfolio Value',
          data: comparisonData.map(item => item.portfolio_value),
          borderColor: '#3182ce',
          backgroundColor: 'rgba(49, 130, 206, 0.1)',
          borderWidth: 3,
          fill: false,
          tension: 0.4
        },
        {
          label: 'Benchmark (S&P 500)',
          data: comparisonData.map(item => item.benchmark_value),
          borderColor: '#e53e3e',
          backgroundColor: 'rgba(229, 62, 62, 0.1)',
          borderWidth: 3,
          fill: false,
          tension: 0.4
        }
      ]
    };

    const outperformanceData = {
      labels: comparisonData.map(item => `Year ${item.year}`),
      datasets: [
        {
          label: 'Outperformance (%)',
          data: comparisonData.map(item => item.outperformance),
          backgroundColor: comparisonData.map(item => 
            item.outperformance >= 0 ? '#38a169' : '#e53e3e'
          ),
          borderWidth: 1
        }
      ]
    };

    const finalOutperformance = comparisonData.length > 0 ? 
      comparisonData[comparisonData.length - 1].outperformance : 0;

    return (
      <>
        <MetricsGrid>
          <MetricCard
            title="vs S&P 500"
            value={`${finalOutperformance >= 0 ? '+' : ''}${finalOutperformance.toFixed(1)}%`}
            change={{
              value: Math.abs(finalOutperformance),
              type: finalOutperformance >= 0 ? 'positive' : 'negative'
            }}
            status={finalOutperformance >= 2 ? 'HEALTHY' : finalOutperformance >= 0 ? 'WARNING' : 'CRITICAL'}
            loading={loading}
          />
        </MetricsGrid>

        <ChartsGrid>
          <Card style={{ gridColumn: '1 / -1' }}>
            <LineChart
              data={comparisonChartData.labels.map((label, index) => ({
                year: index,
                portfolio_value: comparisonChartData.datasets[0].data[index],
                formatted_value: `$${comparisonChartData.datasets[0].data[index].toLocaleString()}`,
                annual_return: index > 0 ? 
                  ((comparisonChartData.datasets[0].data[index] / comparisonChartData.datasets[0].data[index - 1] - 1) * 100) : 0
              }))}
              title="Portfolio vs Benchmark Comparison"
              height={400}
              loading={loading}
            />
          </Card>

          <Card>
            <BarChart
              data={outperformanceData}
              title="Annual Outperformance vs S&P 500"
              height={300}
              loading={loading}
            />
          </Card>
        </ChartsGrid>
      </>
    );
  };

  const renderRisk = () => {
    if (!riskData) {
      return <ErrorMessage>Risk data not available</ErrorMessage>;
    }

    const riskMetricsData = {
      labels: riskData.portfolio_metrics.map(metric => metric.metric),
      datasets: [
        {
          label: 'Portfolio',
          data: riskData.portfolio_metrics.map(metric => metric.value),
          backgroundColor: '#3182ce',
          borderColor: '#2c5aa0',
          borderWidth: 1
        },
        {
          label: 'Benchmark',
          data: riskData.portfolio_metrics.map(metric => metric.benchmark),
          backgroundColor: '#e53e3e',
          borderColor: '#c53030',
          borderWidth: 1
        }
      ]
    };

    return (
      <>
        <MetricsGrid>
          <MetricCard
            title="Risk Score"
            value={`${riskData.risk_score}/100`}
            status={riskData.risk_score <= 30 ? 'HEALTHY' : 
                   riskData.risk_score <= 60 ? 'WARNING' : 'CRITICAL'}
            loading={loading}
          />
          <MetricCard
            title="Risk Level"
            value={riskData.risk_level}
            loading={loading}
          />
          <MetricCard
            title="Volatility"
            value={`${riskData.portfolio_metrics.find(m => m.metric === 'Volatility (%)')?.value.toFixed(1) || 'N/A'}%`}
            loading={loading}
          />
          <MetricCard
            title="Sharpe Ratio"
            value={riskData.portfolio_metrics.find(m => m.metric === 'Sharpe Ratio')?.value.toFixed(2) || 'N/A'}
            loading={loading}
          />
        </MetricsGrid>

        <ChartsGrid>
          <Card>
            <BarChart
              data={riskMetricsData}
              title="Risk Metrics Comparison"
              height={400}
              loading={loading}
            />
          </Card>

          <Card>
            <CardTitle>Risk Analysis</CardTitle>
            <AllocationTable>
              <thead>
                <tr>
                  <TableHeader>Metric</TableHeader>
                  <TableHeader>Portfolio</TableHeader>
                  <TableHeader>Benchmark</TableHeader>
                  <TableHeader>Difference</TableHeader>
                </tr>
              </thead>
              <tbody>
                {riskData.portfolio_metrics.map((metric, index) => {
                  const difference = metric.value - metric.benchmark;
                  const isPositive = metric.metric === 'Sharpe Ratio' || metric.metric === 'Alpha (%)' ? 
                    difference > 0 : difference < 0;
                  
                  return (
                    <TableRow key={index}>
                      <TableCell style={{ fontWeight: 600 }}>
                        {metric.metric}
                      </TableCell>
                      <TableCell>{metric.value.toFixed(2)}</TableCell>
                      <TableCell>{metric.benchmark.toFixed(2)}</TableCell>
                      <TableCell>
                        <span style={{ 
                          color: isPositive ? '#38a169' : '#e53e3e'
                        }}>
                          {difference > 0 ? '+' : ''}{difference.toFixed(2)}
                        </span>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </tbody>
            </AllocationTable>
          </Card>
        </ChartsGrid>
      </>
    );
  };

  return (
    <PageContainer>
      <PageTitle>Portfolio Analysis</PageTitle>

      <TabsContainer>
        <Tab 
          active={activeTab === 'allocation'} 
          onClick={() => setActiveTab('allocation')}
        >
          Allocation
        </Tab>
        <Tab 
          active={activeTab === 'performance'} 
          onClick={() => setActiveTab('performance')}
        >
          Performance
        </Tab>
        <Tab 
          active={activeTab === 'comparison'} 
          onClick={() => setActiveTab('comparison')}
        >
          Comparison
        </Tab>
        <Tab 
          active={activeTab === 'risk'} 
          onClick={() => setActiveTab('risk')}
        >
          Risk Analysis
        </Tab>
      </TabsContainer>

      {activeTab === 'allocation' && renderAllocation()}
      {activeTab === 'performance' && renderPerformance()}
      {activeTab === 'comparison' && renderComparison()}
      {activeTab === 'risk' && renderRisk()}
    </PageContainer>
  );
};

export default PortfolioAnalysis;