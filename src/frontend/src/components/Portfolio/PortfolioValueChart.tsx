import React from 'react';
import styled from 'styled-components';
import { LineChartDataPoint, ChartProps } from '../../types';
import LoadingSpinner from '../Common/LoadingSpinner';

const ChartContainer = styled.div<{ height?: number }>`
  position: relative;
  height: ${props => props.height || 350}px;
  width: 100%;
  max-width: 100%;
  display: flex;
  flex-direction: column;
  margin-bottom: 16px;
  box-sizing: border-box;
`;

const ChartTitle = styled.h3`
  font-size: 1.25rem;
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
  background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
  margin-bottom: 12px;
  min-height: 180px;
  max-height: 220px;
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
  stroke-dasharray: 1,2;
`;

// const AxisLine = styled.line`
//   stroke: #a0aec0;
//   stroke-width: 2;
// `;

const AxisLabel = styled.text`
  font-size: 8px;
  fill: #4a5568;
  text-anchor: middle;
  font-weight: 400;
`;

const YAxisLabel = styled.text`
  font-size: 8px;
  fill: #4a5568;
  text-anchor: end;
  font-weight: 400;
`;

const ChartLine = styled.polyline`
  fill: none;
  stroke: url(#gradient);
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
  filter: drop-shadow(0 1px 2px rgba(59, 130, 246, 0.2));
`;

const ChartArea2 = styled.polygon`
  fill: url(#areaGradient);
  opacity: 0.3;
`;

const DataPoint = styled.circle`
  fill: #3b82f6;
  stroke: white;
  stroke-width: 1;
  cursor: pointer;
  transition: all 0.2s ease;
  filter: drop-shadow(0 1px 2px rgba(59, 130, 246, 0.3));

  &:hover {
    r: 2.5;
    stroke-width: 1.5;
  }
`;

const Tooltip = styled.div.withConfig({
  shouldForwardProp: (prop) => !['x', 'y', 'visible'].includes(prop),
})<{ x: number; y: number; visible: boolean }>`
  position: absolute;
  left: ${props => props.x}px;
  top: ${props => props.y}px;
  background: rgba(0, 0, 0, 0.9);
  color: white;
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 13px;
  pointer-events: none;
  transform: translate(-50%, -100%);
  opacity: ${props => props.visible ? 1 : 0};
  transition: opacity 0.2s ease;
  z-index: 10;
  white-space: nowrap;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  padding: 16px;
  background: linear-gradient(135deg, #f1f5f9 0%, #ffffff 100%);
  border-radius: 8px;
  border: 1px solid #e2e8f0;

  @media (max-width: 768px) {
    grid-template-columns: repeat(2, 1fr);
  }
`;

const StatCard = styled.div`
  text-align: center;
  padding: 12px;
  background: white;
  border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #f1f5f9;
`;

const StatValue = styled.div`
  font-size: 1.125rem;
  font-weight: 700;
  color: #1a202c;
  margin-bottom: 4px;
`;

const StatLabel = styled.div`
  font-size: 0.75rem;
  color: #718096;
  font-weight: 500;
`;

const StatChange = styled.div.withConfig({
  shouldForwardProp: (prop) => prop !== 'positive',
})<{ positive: boolean }>`
  font-size: 0.75rem;
  color: ${props => props.positive ? '#10b981' : '#ef4444'};
  font-weight: 600;
  margin-top: 4px;
`;

const InvestmentSummary = styled.div`
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 12px;
  margin-bottom: 12px;
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;

  @media (max-width: 768px) {
    grid-template-columns: repeat(2, 1fr);
  }
`;

const SummaryItem = styled.div`
  text-align: center;
`;

const SummaryLabel = styled.div`
  font-size: 0.75rem;
  color: #718096;
  margin-bottom: 2px;
`;

const SummaryValue = styled.div`
  font-size: 0.875rem;
  font-weight: 600;
  color: #2d3748;
`;

interface PortfolioValueChartProps extends ChartProps {
  data: LineChartDataPoint[];
  userInput?: {
    investment_amount: number;
    investment_tenure: number;
    risk_profile: 'low' | 'moderate' | 'high';
    investment_type: 'lump_sum' | 'sip' | 'swp';
  };
}

const PortfolioValueChart: React.FC<PortfolioValueChartProps> = ({
  data,
  title,
  height = 400,
  loading = false,
  error,
  userInput
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
        <ErrorMessage>No portfolio value data available</ErrorMessage>
      </ChartContainer>
    );
  }

  // Chart dimensions and scaling
  const chartWidth = 120;
  const chartHeight = 80;
  const padding = { top: 10, right: 12, bottom: 20, left: 20 };

  const maxValue = Math.max(...data.map(d => d.portfolio_value));
  const minValue = Math.min(...data.map(d => d.portfolio_value));
  const valueRange = maxValue - minValue || 1;

  // Generate SVG path points
  const points = data.map((item, index) => {
    const x = data.length === 1 
      ? padding.left + (chartWidth - padding.left - padding.right) / 2
      : padding.left + (index / (data.length - 1)) * (chartWidth - padding.left - padding.right);
    const y = valueRange === 0 
      ? padding.top + (chartHeight - padding.top - padding.bottom) / 2
      : padding.top + chartHeight - padding.bottom - ((item.portfolio_value - minValue) / valueRange) * (chartHeight - padding.top - padding.bottom);
    return { x, y, value: item.portfolio_value, item };
  });

  const pathData = points.map(p => `${p.x},${p.y}`).join(' ');
  
  // Create area path for gradient fill
  const areaPoints = [
    `${padding.left},${chartHeight - padding.bottom}`,
    ...points.map(p => `${p.x},${p.y}`),
    `${chartWidth - padding.right},${chartHeight - padding.bottom}`
  ].join(' ');

  // Calculate statistics
  const finalValue = data[data.length - 1];
  const initialValue = data[0];
  const totalGrowth = finalValue.portfolio_value - initialValue.portfolio_value;
  const totalReturn = ((finalValue.portfolio_value - initialValue.portfolio_value) / initialValue.portfolio_value) * 100;
  
  const avgAnnualReturn = data.length > 1 
    ? data.slice(1).reduce((sum, item) => sum + (item.annual_return || 0), 0) / (data.length - 1)
    : 0;

  const cagr = data.length > 1 
    ? (Math.pow(finalValue.portfolio_value / initialValue.portfolio_value, 1 / (data.length - 1)) - 1) * 100
    : 0;

  const handleMouseEnter = (event: React.MouseEvent, point: typeof points[0]) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const containerRect = event.currentTarget.closest('div')?.getBoundingClientRect();
    
    if (containerRect) {
      const x = rect.left - containerRect.left + rect.width / 2;
      const y = rect.top - containerRect.top;
      
      const content = `Year ${point.item.year}: ${point.item.formatted_value}`;
      
      setTooltip({ visible: true, x, y, content });
    }
  };

  const handleMouseLeave = () => {
    setTooltip({ visible: false, x: 0, y: 0, content: '' });
  };

  return (
    <ChartContainer height={height}>
      {title && <ChartTitle>{title}</ChartTitle>}
      
      {userInput && (
        <InvestmentSummary>
          <SummaryItem>
            <SummaryLabel>Investment Type</SummaryLabel>
            <SummaryValue>{userInput.investment_type.toUpperCase()}</SummaryValue>
          </SummaryItem>
          <SummaryItem>
            <SummaryLabel>Initial Amount</SummaryLabel>
            <SummaryValue>₹{userInput.investment_amount.toLocaleString()}</SummaryValue>
          </SummaryItem>
          <SummaryItem>
            <SummaryLabel>Investment Period</SummaryLabel>
            <SummaryValue>{userInput.investment_tenure} years</SummaryValue>
          </SummaryItem>
          <SummaryItem>
            <SummaryLabel>Risk Profile</SummaryLabel>
            <SummaryValue>{userInput.risk_profile.charAt(0).toUpperCase() + userInput.risk_profile.slice(1)}</SummaryValue>
          </SummaryItem>
        </InvestmentSummary>
      )}
      
      <ChartArea>
        <SVGContainer viewBox={`0 0 ${chartWidth} ${chartHeight}`}>
          <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#3b82f6" />
              <stop offset="100%" stopColor="#1d4ed8" />
            </linearGradient>
            <linearGradient id="areaGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.1" />
            </linearGradient>
          </defs>

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
            const displayValue = value >= 1000000 
              ? `${(value / 1000000).toFixed(1)}M`
              : `${(value / 1000).toFixed(0)}K`;
            
            return (
              <YAxisLabel
                key={percent}
                x={padding.left - 1}
                y={y + 3}
              >
                {displayValue}
              </YAxisLabel>
            );
          })}

          {/* X-axis labels */}
          {data.map((item, index) => {
            const shouldShow = data.length <= 5 
              ? true 
              : (index % Math.ceil(data.length / 4) === 0 || index === data.length - 1);
            
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

          {/* Area fill */}
          <ChartArea2 points={areaPoints} />

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

      <StatsGrid>
        <StatCard>
          <StatValue>₹{finalValue.formatted_value}</StatValue>
          <StatLabel>Final Portfolio Value</StatLabel>
          <StatChange positive={totalGrowth > 0}>
            +₹{totalGrowth.toLocaleString()}
          </StatChange>
        </StatCard>
        
        <StatCard>
          <StatValue>{totalReturn.toFixed(1)}%</StatValue>
          <StatLabel>Total Return</StatLabel>
          <StatChange positive={totalReturn > 0}>
            {totalReturn > 0 ? '+' : ''}{totalReturn.toFixed(1)}%
          </StatChange>
        </StatCard>
        
        <StatCard>
          <StatValue>{cagr.toFixed(1)}%</StatValue>
          <StatLabel>CAGR</StatLabel>
          <StatChange positive={cagr > 0}>
            Compound Annual Growth Rate
          </StatChange>
        </StatCard>
        
        <StatCard>
          <StatValue>{avgAnnualReturn.toFixed(1)}%</StatValue>
          <StatLabel>Avg Annual Return</StatLabel>
          <StatChange positive={avgAnnualReturn > 0}>
            Over {data.length - 1} years
          </StatChange>
        </StatCard>
      </StatsGrid>
    </ChartContainer>
  );
};

export default PortfolioValueChart;