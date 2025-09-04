import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { apiService } from '../../services/api';
import MetricCard from '../Common/MetricCard';
import LoadingSpinner from '../Common/LoadingSpinner';

const SummaryContainer = styled.div`
  margin-bottom: 30px;
`;

const SummaryTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 20px;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
`;

const ErrorMessage = styled.div`
  color: #e53e3e;
  background-color: #fed7d7;
  border: 1px solid #feb2b2;
  padding: 12px;
  border-radius: 4px;
  margin: 20px 0;
`;

interface PerformanceMetrics {
  total_return: number;
  annualized_return: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  best_year: number;
  worst_year: number;
}

const PerformanceSummary: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPerformanceData = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch('http://localhost:8000/api/historical/performance-summary');
        const result = await response.json();
        
        if (result.success) {
          setMetrics(result.data);
        } else {
          throw new Error('Failed to fetch performance data');
        }
      } catch (err) {
        setError('Failed to load historical performance data');
        console.error('Performance data fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchPerformanceData();
  }, []);

  if (loading) {
    return (
      <SummaryContainer>
        <SummaryTitle>Historical S&P 500 Performance (1927-2024)</SummaryTitle>
        <LoadingSpinner size="medium" />
      </SummaryContainer>
    );
  }

  if (error || !metrics) {
    return (
      <SummaryContainer>
        <SummaryTitle>Historical S&P 500 Performance (1927-2024)</SummaryTitle>
        <ErrorMessage>{error || 'No performance data available'}</ErrorMessage>
      </SummaryContainer>
    );
  }

  return (
    <SummaryContainer>
      <SummaryTitle>Historical S&P 500 Performance (1927-2024)</SummaryTitle>
      <MetricsGrid>
        <MetricCard
          title="Total Return"
          value={`${metrics.total_return.toLocaleString()}%`}
          change={{
            value: metrics.total_return,
            type: 'positive'
          }}
          status="HEALTHY"
          loading={false}
        />
        <MetricCard
          title="Annualized Return"
          value={`${metrics.annualized_return.toFixed(1)}%`}
          status={metrics.annualized_return >= 10 ? 'HEALTHY' : 'WARNING'}
          loading={false}
        />
        <MetricCard
          title="Volatility"
          value={`${metrics.volatility.toFixed(1)}%`}
          status={metrics.volatility <= 20 ? 'HEALTHY' : 'WARNING'}
          loading={false}
        />
        <MetricCard
          title="Sharpe Ratio"
          value={metrics.sharpe_ratio.toFixed(2)}
          status={metrics.sharpe_ratio >= 0.5 ? 'HEALTHY' : 'WARNING'}
          loading={false}
        />
        <MetricCard
          title="Max Drawdown"
          value={`${Math.abs(metrics.max_drawdown).toFixed(1)}%`}
          status={Math.abs(metrics.max_drawdown) <= 30 ? 'HEALTHY' : 'WARNING'}
          loading={false}
        />
        <MetricCard
          title="Best Year"
          value={`${metrics.best_year.toFixed(1)}%`}
          change={{
            value: metrics.best_year,
            type: 'positive'
          }}
          status="HEALTHY"
          loading={false}
        />
        <MetricCard
          title="Worst Year"
          value={`${metrics.worst_year.toFixed(1)}%`}
          change={{
            value: Math.abs(metrics.worst_year),
            type: 'negative'
          }}
          status="CRITICAL"
          loading={false}
        />
      </MetricsGrid>
    </SummaryContainer>
  );
};

export default PerformanceSummary;