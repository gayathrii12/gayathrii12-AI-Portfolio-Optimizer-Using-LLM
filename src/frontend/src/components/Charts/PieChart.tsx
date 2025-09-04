import React from 'react';
import styled from 'styled-components';
import { ChartProps, PieChartDataPoint } from '../../types';
import LoadingSpinner from '../Common/LoadingSpinner';

const ChartContainer = styled.div<{ height?: number }>`
  position: relative;
  height: ${props => props.height || 300}px;
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
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

const PieContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 30px;
  width: 100%;
  justify-content: center;
`;

const PieVisual = styled.div`
  width: 200px;
  height: 200px;
  border-radius: 50%;
  position: relative;
  background: conic-gradient(
    ${props => props.theme}
  );
`;

const Legend = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const LegendItem = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
`;

const LegendColor = styled.div<{ color: string }>`
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background-color: ${props => props.color};
`;

interface PieChartProps extends ChartProps {
  data: PieChartDataPoint[];
}

const PieChart: React.FC<PieChartProps> = ({
  data,
  title,
  height = 300,
  loading = false,
  error
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

  // Create conic gradient for pie chart
  let currentAngle = 0;
  const gradientStops = data.map(item => {
    const percentage = item.value;
    const startAngle = currentAngle;
    const endAngle = currentAngle + (percentage * 3.6); // 360deg / 100%
    currentAngle = endAngle;
    
    return `${item.color} ${startAngle}deg ${endAngle}deg`;
  }).join(', ');

  return (
    <ChartContainer height={height}>
      {title && <ChartTitle>{title}</ChartTitle>}
      <PieContainer>
        <PieVisual theme={gradientStops} />
        <Legend>
          {data.map((item, index) => (
            <LegendItem key={index}>
              <LegendColor color={item.color} />
              <span>{item.name}: {item.percentage}</span>
            </LegendItem>
          ))}
        </Legend>
      </PieContainer>
    </ChartContainer>
  );
};

export default PieChart;