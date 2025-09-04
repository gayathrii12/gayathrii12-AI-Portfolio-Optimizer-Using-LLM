import React from 'react';
import styled from 'styled-components';
import { PieChartDataPoint, ChartProps } from '../../types';
import LoadingSpinner from '../Common/LoadingSpinner';

const ChartContainer = styled.div<{ height?: number }>`
  position: relative;
  height: ${props => props.height || 280}px;
  width: 100%;
  max-width: 100%;
  display: flex;
  flex-direction: column;
  margin-bottom: 16px;
  box-sizing: border-box;
`;

const ChartTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 16px;
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

const ChartContent = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 24px;
  flex: 1;

  @media (max-width: 768px) {
    flex-direction: column;
    gap: 16px;
  }
`;

const PieContainer = styled.div`
  position: relative;
  width: 160px;
  height: 160px;
  flex-shrink: 0;
`;

const PieChart = styled.div.withConfig({
  shouldForwardProp: (prop) => prop !== 'gradientStops',
})<{ gradientStops: string }>`
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background: conic-gradient(${props => props.gradientStops});
  position: relative;
  
  &::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 60px;
    height: 60px;
    background: white;
    border-radius: 50%;
    transform: translate(-50%, -50%);
  }
`;

const CenterLabel = styled.div`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  z-index: 1;
  font-size: 0.65rem;
  font-weight: 600;
  color: #4a5568;
  line-height: 1.2;
`;

const Legend = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 160px;
`;

const LegendItem = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  padding: 4px 0;
`;

const LegendColor = styled.div<{ color: string }>`
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background-color: ${props => props.color};
  flex-shrink: 0;
`;

const LegendText = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex: 1;
`;

const AssetName = styled.span`
  font-weight: 500;
  color: #2d3748;
  font-size: 11px;
`;

const AssetPercentage = styled.span`
  font-weight: 600;
  color: #4a5568;
  font-size: 11px;
`;

const SummaryStats = styled.div`
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #e2e8f0;
  font-size: 0.75rem;
  color: #718096;
`;

interface AllocationPieChartProps extends ChartProps {
  data: PieChartDataPoint[];
}

const AllocationPieChart: React.FC<AllocationPieChartProps> = ({
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
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flex: 1 }}>
          <LoadingSpinner />
        </div>
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
        <ErrorMessage>No allocation data available</ErrorMessage>
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

  // Calculate total allocation
  const totalAllocation = data.reduce((sum, item) => sum + item.value, 0);
  const assetCount = data.length;

  return (
    <ChartContainer height={height}>
      {title && <ChartTitle>{title}</ChartTitle>}
      
      <ChartContent>
        <PieContainer>
          <PieChart gradientStops={gradientStops}>
            <CenterLabel>
              <div>Portfolio</div>
              <div>{assetCount} Assets</div>
            </CenterLabel>
          </PieChart>
        </PieContainer>

        <Legend>
          {data.map((item, index) => (
            <LegendItem key={index}>
              <LegendColor color={item.color} />
              <LegendText>
                <AssetName>{item.name}</AssetName>
                <AssetPercentage>{item.percentage}</AssetPercentage>
              </LegendText>
            </LegendItem>
          ))}
          
          <SummaryStats>
            <div>Total Allocation: {totalAllocation.toFixed(1)}%</div>
            <div>Asset Classes: {assetCount}</div>
          </SummaryStats>
        </Legend>
      </ChartContent>
    </ChartContainer>
  );
};

export default AllocationPieChart;