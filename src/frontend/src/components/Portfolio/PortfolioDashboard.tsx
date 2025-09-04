import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { apiService } from '../../services/api';
import { PieChartDataPoint, LineChartDataPoint, PortfolioAllocation } from '../../types';
import LoadingSpinner from '../Common/LoadingSpinner';
import AllocationPieChart from './AllocationPieChart';
import ReturnsLineChart from './ReturnsLineChart';
import PortfolioValueChart from './PortfolioValueChart';
import KeyRecommendations from './KeyRecommendations';

const DashboardContainer = styled.div`
  padding: 20px;
  background-color: #f8fafc;
  min-height: 100vh;
  max-width: 1200px;
  margin: 0 auto;
`;

const DashboardHeader = styled.div`
  margin-bottom: 24px;
  text-align: center;
`;

const Title = styled.h1`
  font-size: 1.75rem;
  font-weight: 700;
  color: #1a202c;
  margin: 0 0 8px;
`;

const Subtitle = styled.p`
  font-size: 0.875rem;
  color: #718096;
  margin: 0;
`;

const ChartsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 20px;
  align-items: start;

  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
    gap: 16px;
  }
  
  @media (max-width: 768px) {
    gap: 12px;
  }
`;

const ChartCard = styled.div`
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  border: 1px solid #e2e8f0;
  overflow: hidden;
  max-width: 100%;
  box-sizing: border-box;
  min-height: 350px;
  
  @media (max-width: 768px) {
    padding: 16px;
    min-height: 300px;
  }
`;

const FullWidthChart = styled(ChartCard)`
  grid-column: 1 / -1;
  margin-top: 0;
  max-width: 100%;
  overflow: hidden;
  min-height: 450px;
  
  @media (max-width: 768px) {
    min-height: 400px;
  }
`;

const ErrorMessage = styled.div`
  color: #e53e3e;
  background-color: #fed7d7;
  border: 1px solid #feb2b2;
  padding: 16px;
  border-radius: 8px;
  text-align: center;
  margin: 16px 0;
`;

const LoadingContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 400px;
`;

interface PortfolioDashboardProps {
  userInput?: {
    investment_amount: number;
    investment_tenure: number;
    risk_profile: 'low' | 'moderate' | 'high';
    investment_type: 'lump_sum' | 'sip' | 'swp';
  };
}

const PortfolioDashboard: React.FC<PortfolioDashboardProps> = ({ userInput }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [allocationData, setAllocationData] = useState<PieChartDataPoint[]>([]);
  const [returnsData, setReturnsData] = useState<LineChartDataPoint[]>([]);
  const [portfolioValueData, setPortfolioValueData] = useState<LineChartDataPoint[]>([]);
  const [, setPortfolioAllocation] = useState<PortfolioAllocation | null>(null);

  useEffect(() => {
    loadDashboardData();
  }, [userInput]);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load all required data
      const [allocation, pieData, lineData, portfolioData] = await Promise.all([
        apiService.getPortfolioAllocation(),
        apiService.getPieChartData(),
        apiService.getLineChartData(),
        apiService.getLineChartData() // Using same data for portfolio value chart for now
      ]);

      setPortfolioAllocation(allocation);
      setAllocationData(pieData);
      setReturnsData(lineData);
      setPortfolioValueData(portfolioData);
    } catch (err) {
      console.error('Error loading dashboard data:', err);
      setError('Failed to load portfolio dashboard data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <DashboardContainer>
        <LoadingContainer>
          <LoadingSpinner />
        </LoadingContainer>
      </DashboardContainer>
    );
  }

  if (error) {
    return (
      <DashboardContainer>
        <DashboardHeader>
          <Title>Portfolio Dashboard</Title>
          <Subtitle>Investment portfolio analysis and projections</Subtitle>
        </DashboardHeader>
        <ErrorMessage>{error}</ErrorMessage>
      </DashboardContainer>
    );
  }

  return (
    <DashboardContainer>
      <DashboardHeader>
        <Title>Portfolio Dashboard</Title>
        <Subtitle>
          {userInput 
            ? `Investment analysis for ${userInput.risk_profile} risk profile`
            : 'Investment portfolio analysis and projections'
          }
        </Subtitle>
      </DashboardHeader>

      <ChartsGrid>
        <ChartCard>
          <AllocationPieChart 
            data={allocationData}
            title="Portfolio Allocation"
            loading={loading}
            error={error}
          />
        </ChartCard>

        <ChartCard>
          <ReturnsLineChart 
            data={returnsData}
            title="Expected Returns Over Time"
            loading={loading}
            error={error}
            showReturns={true}
          />
        </ChartCard>
      </ChartsGrid>

      <FullWidthChart>
        <PortfolioValueChart 
          data={portfolioValueData}
          title="Portfolio Value Growth"
          loading={loading}
          error={error}
          userInput={userInput}
        />
      </FullWidthChart>

      <KeyRecommendations userInput={userInput} />
    </DashboardContainer>
  );
};

export default PortfolioDashboard;