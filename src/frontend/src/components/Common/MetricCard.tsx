import React from 'react';
import styled from 'styled-components';
import { MetricCardProps } from '../../types';
import LoadingSpinner from './LoadingSpinner';

const Card = styled.div`
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  padding: 20px;
  text-align: center;
  transition: transform 0.2s, box-shadow 0.2s;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
  }
`;

const Title = styled.h3`
  font-size: 0.875rem;
  color: #718096;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin: 0 0 8px;
  font-weight: 600;
`;

const Value = styled.div<{ status?: string }>`
  font-size: 2rem;
  font-weight: bold;
  margin-bottom: 8px;
  color: ${props => {
    switch (props.status) {
      case 'HEALTHY':
        return '#38a169';
      case 'WARNING':
        return '#d69e2e';
      case 'CRITICAL':
        return '#e53e3e';
      default:
        return '#1a202c';
    }
  }};
`;

const Change = styled.div<{ type: 'positive' | 'negative' }>`
  font-size: 0.875rem;
  font-weight: 500;
  color: ${props => props.type === 'positive' ? '#38a169' : '#e53e3e'};
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
`;

const ChangeIcon = styled.span`
  font-size: 0.75rem;
`;

const LoadingContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 80px;
`;

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  status,
  loading = false
}) => {
  if (loading) {
    return (
      <Card>
        <Title>{title}</Title>
        <LoadingContainer>
          <LoadingSpinner size="small" />
        </LoadingContainer>
      </Card>
    );
  }

  const formatValue = (val: string | number): string => {
    if (typeof val === 'number') {
      // Format large numbers with commas
      if (val >= 1000) {
        return val.toLocaleString();
      }
      // Format percentages and decimals
      if (val < 1 && val > 0) {
        return `${(val * 100).toFixed(1)}%`;
      }
      return val.toString();
    }
    return val;
  };

  const getChangeIcon = (type: 'positive' | 'negative') => {
    return type === 'positive' ? '↗️' : '↘️';
  };

  const formatChangeValue = (changeValue: number): string => {
    const absValue = Math.abs(changeValue);
    if (absValue >= 1) {
      return absValue.toFixed(1);
    }
    return (absValue * 100).toFixed(1) + '%';
  };

  return (
    <Card>
      <Title>{title}</Title>
      <Value status={status}>
        {formatValue(value)}
      </Value>
      {change && (
        <Change type={change.type}>
          <ChangeIcon>{getChangeIcon(change.type)}</ChangeIcon>
          {change.type === 'positive' ? '+' : '-'}{formatChangeValue(change.value)}
        </Change>
      )}
    </Card>
  );
};

export default MetricCard;