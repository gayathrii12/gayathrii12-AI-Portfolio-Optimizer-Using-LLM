import React from 'react';
import styled from 'styled-components';
import { ChartProps } from '../../types';
import LoadingSpinner from '../Common/LoadingSpinner';

const ChartContainer = styled.div<{ height?: number }>`
  position: relative;
  height: ${props => props.height || 300}px;
  width: 100%;
  display: flex;
  flex-direction: column;
`;

const ChartTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 20px;
  text-align: center;
`;

const ErrorMessage = styled.div`
  color: #e53e3e;
  background-color: #fed7d7;
  border: 1px solid #feb2b2;
  padding: 12px;
  border-radius: 4px;
  text-align: center;
`;

const ChartArea = styled.div`
  flex: 1;
  display: flex;
  align-items: end;
  gap: 4px;
  padding: 20px 20px 40px 20px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: white;
  overflow-x: auto;
  min-height: 250px;
`;

const BarContainer = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 100%;
  min-width: 40px;
  position: relative;
`;

const Bar = styled.div<{ height: number; color?: string }>`
  width: 100%;
  max-width: 60px;
  height: ${props => props.height}%;
  background-color: ${props => props.color || '#3182ce'};
  border-radius: 4px 4px 0 0;
  margin-bottom: 8px;
  transition: all 0.3s ease;
  
  &:hover {
    opacity: 0.8;
    transform: translateY(-2px);
  }
`;

const BarLabel = styled.div`
  font-size: 10px;
  color: #4a5568;
  text-align: center;
  word-break: break-word;
  max-width: 60px;
  line-height: 1.2;
  transform: rotate(-45deg);
  transform-origin: center;
  margin-top: 8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const BarValue = styled.div`
  font-size: 10px;
  color: #718096;
  margin-bottom: 4px;
`;

interface BarChartData {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    backgroundColor?: string | string[];
    borderColor?: string | string[];
    borderWidth?: number;
  }>;
}

interface BarChartProps extends ChartProps {
  data: BarChartData;
  horizontal?: boolean;
  stacked?: boolean;
}

const Legend = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
  flex-wrap: wrap;
`;

const LegendItem = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #4a5568;
`;

const LegendColor = styled.div<{ color: string }>`
  width: 12px;
  height: 12px;
  border-radius: 2px;
  background-color: ${props => props.color};
`;

const BarChart: React.FC<BarChartProps> = ({
  data,
  title,
  height = 300,
  loading = false,
  error,
  horizontal = false,
  stacked = false
}) => {
  if (loading) {
    return (
      <ChartContainer height={height}>
        {title && <ChartTitle>{title}</ChartTitle>}
        <LoadingSpinner />
      </ChartContainer>
    );
  }

  if (error) {
    return (
      <ChartContainer height={height}>
        {title && <ChartTitle>{title}</ChartTitle>}
        <ErrorMessage>{error}</ErrorMessage>
      </ChartContainer>
    );
  }

  if (!data || !data.labels || data.labels.length === 0) {
    return (
      <ChartContainer height={height}>
        {title && <ChartTitle>{title}</ChartTitle>}
        <ErrorMessage>No data available</ErrorMessage>
      </ChartContainer>
    );
  }

  // Get the first dataset for simplicity
  const dataset = data.datasets[0];
  if (!dataset) {
    return (
      <ChartContainer height={height}>
        {title && <ChartTitle>{title}</ChartTitle>}
        <ErrorMessage>No dataset available</ErrorMessage>
      </ChartContainer>
    );
  }

  const maxValue = Math.max(...dataset.data);
  const colors = Array.isArray(dataset.backgroundColor) 
    ? dataset.backgroundColor 
    : [dataset.backgroundColor || '#3182ce'];

  // Create legend items for positive/negative values
  const hasNegativeValues = dataset.data.some(value => value < 0);
  const hasPositiveValues = dataset.data.some(value => value > 0);

  return (
    <ChartContainer height={height}>
      {title && <ChartTitle>{title}</ChartTitle>}
      
      {/* Legend for accessibility */}
      {(hasNegativeValues || hasPositiveValues) && (
        <Legend>
          {hasPositiveValues && (
            <LegendItem>
              <LegendColor color="#10b981" />
              <span>Positive Returns</span>
            </LegendItem>
          )}
          {hasNegativeValues && (
            <LegendItem>
              <LegendColor color="#ef4444" />
              <span>Negative Returns</span>
            </LegendItem>
          )}
        </Legend>
      )}
      
      <ChartArea>
        {data.labels.map((label, index) => {
          const value = dataset.data[index];
          const barHeight = (value / maxValue) * 80; // 80% max height
          const color = colors[index % colors.length];
          
          return (
            <BarContainer key={index}>
              <BarValue>{value}</BarValue>
              <Bar height={barHeight} color={color} />
              <BarLabel>{label}</BarLabel>
            </BarContainer>
          );
        })}
      </ChartArea>
    </ChartContainer>
  );
};

export default BarChart;