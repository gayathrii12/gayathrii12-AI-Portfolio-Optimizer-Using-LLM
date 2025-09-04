import React from 'react';
import styled from 'styled-components';

// Types for portfolio recommendation
export interface AssetAllocation {
  sp500: number;
  small_cap: number;
  bonds: number;
  real_estate: number;
  gold: number;
}

export interface ProjectionData {
  year: number;
  portfolio_value: number;
  annual_return: number;
  cumulative_return: number;
}

export interface PortfolioRecommendationData {
  allocation: AssetAllocation;
  projections: ProjectionData[];
  risk_metrics: {
    expected_return: number;
    volatility: number;
    sharpe_ratio: number;
  };
  summary: {
    initial_investment: number;
    final_value: number;
    total_return: number;
    investment_type: string;
    tenure_years: number;
    risk_profile: string;
  };
}

interface PortfolioRecommendationProps {
  data?: PortfolioRecommendationData;
  loading?: boolean;
}

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 2rem;
`;

const Title = styled.h2`
  color: #1a202c;
  font-size: 1.8rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
`;

const Subtitle = styled.p`
  color: #718096;
  font-size: 1rem;
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin-bottom: 2rem;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const Card = styled.div`
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const CardTitle = styled.h3`
  color: #2d3748;
  font-size: 1.2rem;
  font-weight: 600;
  margin-bottom: 1rem;
`;

const AllocationList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
`;

const AllocationItem = styled.div`
  display: flex;
  justify-content: between;
  align-items: center;
  padding: 0.5rem;
  background: #f7fafc;
  border-radius: 8px;
`;

const AssetName = styled.span`
  font-weight: 500;
  color: #2d3748;
  flex: 1;
`;

const AssetPercentage = styled.span`
  font-weight: 600;
  color: #3182ce;
  margin-left: auto;
`;

const AssetColor = styled.div<{ color: string }>`
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background-color: ${props => props.color};
  margin-right: 0.75rem;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 1rem;
  margin-bottom: 1rem;
`;

const MetricItem = styled.div`
  text-align: center;
  padding: 1rem;
  background: #f7fafc;
  border-radius: 8px;
`;

const MetricValue = styled.div`
  font-size: 1.5rem;
  font-weight: 600;
  color: #3182ce;
  margin-bottom: 0.25rem;
`;

const MetricLabel = styled.div`
  font-size: 0.875rem;
  color: #718096;
`;

const SummaryGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin-top: 2rem;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const SummaryCard = styled(Card)`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
`;

const SummaryTitle = styled.h3`
  color: white;
  font-size: 1.2rem;
  font-weight: 600;
  margin-bottom: 1rem;
`;

const SummaryItem = styled.div`
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.75rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);

  &:last-child {
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
  }
`;

const SummaryLabel = styled.span`
  opacity: 0.9;
`;

const SummaryValue = styled.span`
  font-weight: 600;
`;

const LoadingContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 400px;
  color: #718096;
  font-size: 1.1rem;
`;

// Asset colors for consistent visualization
const ASSET_COLORS = {
  sp500: '#1f77b4',
  small_cap: '#ff7f0e',
  bonds: '#2ca02c',
  real_estate: '#d62728',
  gold: '#9467bd'
};

const ASSET_NAMES = {
  sp500: 'S&P 500',
  small_cap: 'US Small Cap',
  bonds: 'Bonds',
  real_estate: 'Real Estate',
  gold: 'Gold'
};

const PortfolioRecommendation: React.FC<PortfolioRecommendationProps> = ({ data, loading = false }) => {
  // Debug logging
  console.log('PortfolioRecommendation render:', { data, loading });

  if (loading) {
    return (
      <Container>
        <LoadingContainer>
          Generating your personalized portfolio recommendation...
        </LoadingContainer>
      </Container>
    );
  }

  // Handle case where data is undefined or incomplete
  if (!data || !data.summary || !data.allocation || !data.risk_metrics || !data.projections) {
    console.error('PortfolioRecommendation: Missing required data', {
      hasData: !!data,
      hasSummary: !!(data?.summary),
      hasAllocation: !!(data?.allocation),
      hasRiskMetrics: !!(data?.risk_metrics),
      hasProjections: !!(data?.projections),
      data
    });
    return (
      <Container>
        <LoadingContainer>
          Unable to load portfolio recommendation. Please try again.
        </LoadingContainer>
      </Container>
    );
  }

  // Note: Pie chart data preparation for future chart implementation
  // const pieData = Object.entries(data.allocation).map(([asset, percentage]) => ({
  //   name: ASSET_NAMES[asset as keyof typeof ASSET_NAMES],
  //   value: percentage,
  //   color: ASSET_COLORS[asset as keyof typeof ASSET_COLORS]
  // }));

  // Format currency values
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  // Format percentage values
  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`;
  };

  return (
    <Container>
      <Header>
        <Title>Your Personalized Portfolio Recommendation</Title>
        <Subtitle>
          Based on your {data.summary.risk_profile.toLowerCase()} risk profile and {data.summary.tenure_years}-year investment horizon
        </Subtitle>
      </Header>

      <Grid>
        {/* Asset Allocation */}
        <Card>
          <CardTitle>Recommended Asset Allocation</CardTitle>
          <div style={{ height: '300px', marginBottom: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f8f9fa', borderRadius: '8px' }}>
            <div style={{ textAlign: 'center', color: '#6c757d' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📊</div>
              <div>Portfolio Allocation Chart</div>
              <div style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
                Visual representation of your recommended asset allocation
              </div>
            </div>
          </div>
          
          <AllocationList>
            {Object.entries(data.allocation).map(([asset, percentage]) => (
              <AllocationItem key={asset}>
                <AssetColor color={ASSET_COLORS[asset as keyof typeof ASSET_COLORS]} />
                <AssetName>{ASSET_NAMES[asset as keyof typeof ASSET_NAMES]}</AssetName>
                <AssetPercentage>{formatPercentage(percentage)}</AssetPercentage>
              </AllocationItem>
            ))}
          </AllocationList>
        </Card>

        {/* Portfolio Growth Projection */}
        <Card>
          <CardTitle>Portfolio Growth Projection</CardTitle>
          <div style={{ height: '300px', marginBottom: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f8f9fa', borderRadius: '8px' }}>
            <div style={{ textAlign: 'center', color: '#6c757d' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📈</div>
              <div>Growth Projection Chart</div>
              <div style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
                Visual representation of your portfolio growth over time
              </div>
            </div>
          </div>

          <MetricsGrid>
            <MetricItem>
              <MetricValue>{formatPercentage(data.risk_metrics.expected_return)}</MetricValue>
              <MetricLabel>Expected Return</MetricLabel>
            </MetricItem>
            <MetricItem>
              <MetricValue>{formatPercentage(data.risk_metrics.volatility)}</MetricValue>
              <MetricLabel>Volatility</MetricLabel>
            </MetricItem>
            <MetricItem>
              <MetricValue>{data.risk_metrics.sharpe_ratio.toFixed(2)}</MetricValue>
              <MetricLabel>Sharpe Ratio</MetricLabel>
            </MetricItem>
          </MetricsGrid>

          {/* Simple projection table */}
          <div style={{ marginTop: '1rem', maxHeight: '200px', overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
              <thead>
                <tr style={{ background: '#f8f9fa' }}>
                  <th style={{ padding: '0.5rem', textAlign: 'left', borderBottom: '1px solid #dee2e6' }}>Year</th>
                  <th style={{ padding: '0.5rem', textAlign: 'right', borderBottom: '1px solid #dee2e6' }}>Portfolio Value</th>
                  <th style={{ padding: '0.5rem', textAlign: 'right', borderBottom: '1px solid #dee2e6' }}>Annual Return</th>
                </tr>
              </thead>
              <tbody>
                {data.projections.slice(0, 11).map((projection, index) => (
                  <tr key={projection.year} style={{ borderBottom: '1px solid #f1f3f4' }}>
                    <td style={{ padding: '0.5rem' }}>{projection.year}</td>
                    <td style={{ padding: '0.5rem', textAlign: 'right', fontWeight: '500' }}>
                      {formatCurrency(projection.portfolio_value)}
                    </td>
                    <td style={{ padding: '0.5rem', textAlign: 'right', color: projection.annual_return >= 0 ? '#38a169' : '#e53e3e' }}>
                      {formatPercentage(projection.annual_return)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </Grid>

      <SummaryGrid>
        <SummaryCard>
          <SummaryTitle>Investment Summary</SummaryTitle>
          <SummaryItem>
            <SummaryLabel>Investment Type:</SummaryLabel>
            <SummaryValue>{data.summary.investment_type === 'lumpsum' ? 'Lump Sum' : 'SIP'}</SummaryValue>
          </SummaryItem>
          <SummaryItem>
            <SummaryLabel>Initial Investment:</SummaryLabel>
            <SummaryValue>{formatCurrency(data.summary.initial_investment)}</SummaryValue>
          </SummaryItem>
          <SummaryItem>
            <SummaryLabel>Investment Tenure:</SummaryLabel>
            <SummaryValue>{data.summary.tenure_years} years</SummaryValue>
          </SummaryItem>
          <SummaryItem>
            <SummaryLabel>Risk Profile:</SummaryLabel>
            <SummaryValue>{data.summary.risk_profile}</SummaryValue>
          </SummaryItem>
        </SummaryCard>

        <SummaryCard>
          <SummaryTitle>Projected Returns</SummaryTitle>
          <SummaryItem>
            <SummaryLabel>Final Portfolio Value:</SummaryLabel>
            <SummaryValue>{formatCurrency(data.summary.final_value)}</SummaryValue>
          </SummaryItem>
          <SummaryItem>
            <SummaryLabel>Total Return:</SummaryLabel>
            <SummaryValue>{formatCurrency(data.summary.total_return)}</SummaryValue>
          </SummaryItem>
          <SummaryItem>
            <SummaryLabel>Return Percentage:</SummaryLabel>
            <SummaryValue>
              {formatPercentage((data.summary.total_return / data.summary.initial_investment) * 100)}
            </SummaryValue>
          </SummaryItem>
          <SummaryItem>
            <SummaryLabel>Annualized Return:</SummaryLabel>
            <SummaryValue>{formatPercentage(data.risk_metrics.expected_return)}</SummaryValue>
          </SummaryItem>
        </SummaryCard>
      </SummaryGrid>
    </Container>
  );
};

export default PortfolioRecommendation;