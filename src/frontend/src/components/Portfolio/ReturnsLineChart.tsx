import React from 'react';
import styled from 'styled-components';
import { LineChartDataPoint, ChartProps } from '../../types';
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

const ChartArea = styled.div`
  flex: 1;
  position: relative;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 12px;
  background: #fafafa;
  margin-bottom: 12px;
  min-height: 160px;
  max-height: 180px;
  overflow: hidden;
`;

const SVGContainer = styled.svg`
  width: 100%;
  height: 100%;
  overflow: visible;
`;

const GridLine = styled.line`
  stroke: #f1f5f9;
  stroke-width: 1;
`;

// const AxisLine = styled.line`
//   stroke: #cbd5e0;
//   stroke-width: 2;
// `;

const AxisLabel = styled.text`
  font-size: 8px;
  fill: #718096;
  text-anchor: middle;
`;

const YAxisLabel = styled.text`
  font-size: 8px;
  fill: #718096;
  text-anchor: end;
`;

const ChartLine = styled.polyline`
  fill: none;
  stroke: #3182ce;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
`;

const DataPoint = styled.circle`
  fill: #3182ce;
  stroke: white;
  stroke-width: 1;
  cursor: pointer;
  transition: r 0.2s ease;

  &:hover {
    r: 2.5;
  }
`;

const Tooltip = styled.div.withConfig({
  shouldForwardProp: (prop) => !['x', 'y', 'visible'].includes(prop),
})<{ x: number; y: number; visible: boolean }>`
  position: absolute;
  left: ${props => props.x}px;
  top: ${props => props.y}px;
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 8px 12px;
  border-radius: 4px;
  font-size: 12px;
  pointer-events: none;
  transform: translate(-50%, -100%);
  opacity: ${props => props.visible ? 1 : 0};
  transition: opacity 0.2s ease;
  z-index: 10;
  white-space: nowrap;
`;

const StatsContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
  padding: 12px;
  background: #f8fafc;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
`;

const StatItem = styled.div`
  text-align: center;
  padding: 8px;
`;

const StatValue = styled.div`
  font-size: 1rem;
  font-weight: 600;
  color: #2d3748;
`;

const StatLabel = styled.div`
  font-size: 0.75rem;
  color: #718096;
  margin-top: 2px;
`;

interface ReturnsLineChartProps extends ChartProps {
  data: LineChartDataPoint[];
  showReturns?: boolean;
}

const ReturnsLineChart: React.FC<ReturnsLineChartProps> = ({
  data,
  title,
  height = 300,
  loading = false,
  error,
  showReturns = false
}) => {
  const [tooltip, setTooltip] = React.useState<{
    visible: boolean;
    x: number;
    y: number;
    content: string;
  }>({ visible: false, x: 0, y: 0, content: '' });

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
        <ErrorMessage>No returns data available</ErrorMessage>
      </ChartContainer>
    );
  }

  // Chart dimensions and scaling
  const chartWidth = 120;
  const chartHeight = 80;
  const padding = { top: 10, right: 12, bottom: 20, left: 18 };

  const maxValue = Math.max(...data.map(d => showReturns ? (d.cumulative_return || 0) : d.portfolio_value));
  const minValue = Math.min(...data.map(d => showReturns ? (d.cumulative_return || 0) : d.portfolio_value));
  const valueRange = maxValue - minValue || 1;

  // Generate SVG path points
  const points = data.map((item, index) => {
    const x = data.length === 1 
      ? padding.left + (chartWidth - padding.left - padding.right) / 2
      : padding.left + (index / (data.length - 1)) * (chartWidth - padding.left - padding.right);
    const value = showReturns ? (item.cumulative_return || 0) : item.portfolio_value;
    const y = valueRange === 0 
      ? padding.top + (chartHeight - padding.top - padding.bottom) / 2
      : padding.top + chartHeight - padding.bottom - ((value - minValue) / valueRange) * (chartHeight - padding.top - padding.bottom);
    return { x, y, value, item };
  });

  const pathData = points.map(p => `${p.x},${p.y}`).join(' ');

  // Calculate statistics
  const finalValue = data[data.length - 1];
  const initialValue = data[0];
  const totalReturn = showReturns 
    ? (finalValue.cumulative_return || 0)
    : ((finalValue.portfolio_value - initialValue.portfolio_value) / initialValue.portfolio_value) * 100;
  
  const avgAnnualReturn = data.length > 1 
    ? data.slice(1).reduce((sum, item) => sum + (item.annual_return || 0), 0) / (data.length - 1)
    : 0;

  const handleMouseEnter = (event: React.MouseEvent, point: typeof points[0]) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const containerRect = event.currentTarget.closest('div')?.getBoundingClientRect();
    
    if (containerRect) {
      const x = rect.left - containerRect.left + rect.width / 2;
      const y = rect.top - containerRect.top;
      
      const content = showReturns
        ? `Year ${point.item.year}: ${point.value.toFixed(1)}% return`
        : `Year ${point.item.year}: ${point.item.formatted_value}`;
      
      setTooltip({ visible: true, x, y, content });
    }
  };

  const handleMouseLeave = () => {
    setTooltip({ visible: false, x: 0, y: 0, content: '' });
  };

  return (
    <ChartContainer height={height}>
      {title && <ChartTitle>{title}</ChartTitle>}
      
      <ChartArea>
        <SVGContainer viewBox={`0 0 ${chartWidth} ${chartHeight}`}>
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map(percent => {
            const y = padding.top + (percent / 100) * (chartHeight - padding.top - padding.bottom);
            return (
              <GridLine
                key={percent}
                x1={padding.left}
                y1={y}
                x2={chartWidth - padding.right}
                y2={y}
              />
            );
          })}

          {/* Y-axis labels */}
          {[0, 25, 50, 75, 100].map(percent => {
            const y = padding.top + (percent / 100) * (chartHeight - padding.top - padding.bottom);
            const value = maxValue - (percent / 100) * valueRange;
            const displayValue = showReturns 
              ? `${value.toFixed(0)}%`
              : `${(value / 1000).toFixed(0)}K`;
            
            return (
              <YAxisLabel
                key={percent}
                x={padding.left - 2}
                y={y + 4}
              >
                {displayValue}
              </YAxisLabel>
            );
          })}

          {/* X-axis labels */}
          {data.map((item, index) => {
            const shouldShow = data.length <= 5 
              ? true 
              : (index % Math.ceil(data.length / 3) === 0 || index === data.length - 1);
            
            if (shouldShow) {
              const x = data.length === 1 
                ? padding.left + (chartWidth - padding.left - padding.right) / 2
                : padding.left + (index / (data.length - 1)) * (chartWidth - padding.left - padding.right);
              return (
                <AxisLabel
                  key={index}
                  x={x}
                  y={chartHeight - 6}
                >
                  Y{item.year}
                </AxisLabel>
              );
            }
            return null;
          })}

          {/* Chart line */}
          <ChartLine points={pathData} />

          {/* Data points */}
          {points.map((point, index) => (
            <DataPoint
              key={index}
              cx={point.x}
              cy={point.y}
              r="1.5"
              onMouseEnter={(e) => handleMouseEnter(e, point)}
              onMouseLeave={handleMouseLeave}
            />
          ))}
        </SVGContainer>

        <Tooltip
          x={tooltip.x}
          y={tooltip.y}
          visible={tooltip.visible}
        >
          {tooltip.content}
        </Tooltip>
      </ChartArea>

      <StatsContainer>
        <StatItem>
          <StatValue>{showReturns ? `${totalReturn.toFixed(1)}%` : finalValue.formatted_value}</StatValue>
          <StatLabel>{showReturns ? 'Total Return' : 'Final Value'}</StatLabel>
        </StatItem>
        <StatItem>
          <StatValue>{avgAnnualReturn.toFixed(1)}%</StatValue>
          <StatLabel>Avg Annual Return</StatLabel>
        </StatItem>
        <StatItem>
          <StatValue>{data.length - 1}</StatValue>
          <StatLabel>Years</StatLabel>
        </StatItem>
        <StatItem>
          <StatValue>{showReturns ? initialValue.formatted_value : `${totalReturn.toFixed(1)}%`}</StatValue>
          <StatLabel>{showReturns ? 'Initial Value' : 'Total Growth'}</StatLabel>
        </StatItem>
      </StatsContainer>
    </ChartContainer>
  );
};

export default ReturnsLineChart;