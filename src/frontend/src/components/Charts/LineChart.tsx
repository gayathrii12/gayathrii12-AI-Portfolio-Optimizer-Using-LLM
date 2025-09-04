import React from 'react';
import styled from 'styled-components';
import { ChartProps, LineChartDataPoint } from '../../types';
import LoadingSpinner from '../Common/LoadingSpinner';

const ChartContainer = styled.div<{ height?: number }>`
  position: relative;
  height: ${props => props.height || 400}px;
  width: 100%;
  max-width: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-sizing: border-box;
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
  position: relative;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 16px;
  background: white;
  overflow: hidden;
  max-width: 100%;
  box-sizing: border-box;
`;

const LineContainer = styled.svg`
  width: 100%;
  height: 100%;
  max-width: 100%;
  overflow: visible;
`;

const DataTable = styled.div`
  margin-top: 16px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
  max-height: 200px;
  overflow-y: auto;
`;

const TableRow = styled.div<{ isHeader?: boolean }>`
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  padding: 12px;
  border-bottom: 1px solid #e2e8f0;
  background: ${props => props.isHeader ? '#f7fafc' : 'white'};
  font-weight: ${props => props.isHeader ? '600' : '400'};
  
  &:last-child {
    border-bottom: none;
  }
`;

interface LineChartProps extends ChartProps {
  data: LineChartDataPoint[];
  showReturns?: boolean;
}

const LineChart: React.FC<LineChartProps> = ({
  data,
  title,
  height = 400,
  loading = false,
  error,
  showReturns = false
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

  if (!data || data.length === 0) {
    return (
      <ChartContainer height={height}>
        {title && <ChartTitle>{title}</ChartTitle>}
        <ErrorMessage>No data available</ErrorMessage>
      </ChartContainer>
    );
  }

  // Calculate chart dimensions and scaling
  const chartWidth = 100; // percentage
  const chartHeight = 80; // percentage
  const maxValue = Math.max(...data.map(d => d.portfolio_value));
  const minValue = Math.min(...data.map(d => d.portfolio_value));
  const valueRange = maxValue - minValue;

  // Generate SVG path for line
  const points = data.map((item, index) => {
    const x = (index / (data.length - 1)) * chartWidth;
    const y = chartHeight - ((item.portfolio_value - minValue) / valueRange) * chartHeight;
    return `${x},${y}`;
  }).join(' ');

  return (
    <ChartContainer height={height}>
      {title && <ChartTitle>{title}</ChartTitle>}
      <ChartArea>
        <LineContainer viewBox={`0 0 ${chartWidth} ${chartHeight + 10}`}>
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map(y => (
            <line
              key={y}
              x1="0"
              y1={y * chartHeight / 100}
              x2={chartWidth}
              y2={y * chartHeight / 100}
              stroke="#e2e8f0"
              strokeWidth="0.5"
            />
          ))}
          
          {/* Line chart */}
          <polyline
            points={points}
            fill="none"
            stroke="#3182ce"
            strokeWidth="2"
          />
          
          {/* Data points */}
          {data.map((item, index) => {
            const x = (index / (data.length - 1)) * chartWidth;
            const y = chartHeight - ((item.portfolio_value - minValue) / valueRange) * chartHeight;
            return (
              <circle
                key={index}
                cx={x}
                cy={y}
                r="3"
                fill="#3182ce"
              />
            );
          })}
        </LineContainer>
      </ChartArea>
      
      <DataTable>
        <TableRow isHeader>
          <div>Year</div>
          <div>Portfolio Value</div>
          {showReturns && <div>Return %</div>}
        </TableRow>
        {data.map((item, index) => (
          <TableRow key={index}>
            <div>Year {item.year}</div>
            <div>{item.formatted_value}</div>
            {showReturns && <div>{item.cumulative_return?.toFixed(1)}%</div>}
          </TableRow>
        ))}
      </DataTable>
    </ChartContainer>
  );
};

export default LineChart;