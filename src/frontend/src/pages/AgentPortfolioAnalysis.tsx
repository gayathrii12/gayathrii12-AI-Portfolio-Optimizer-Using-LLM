import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { agentApiService } from '../services/agentApi';
import PieChart from '../components/Charts/PieChart';
import LineChart from '../components/Charts/LineChart';
import BarChart from '../components/Charts/BarChart';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import MetricCard from '../components/Common/MetricCard';
import StatusBadge from '../components/Common/StatusBadge';

const AnalysisContainer = styled.div`
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
  
  @media (max-width: 768px) {
    padding: 16px;
  }
`;

const PageTitle = styled.h1`
  font-size: 2rem;
  font-weight: 700;
  color: #1a202c;
  margin: 0 0 30px;
`;

const AgentBadge = styled.div`
  display: inline-flex;
  align-items: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 8px 16px;
  border-radius: 20px;
  font-size: 0.875rem;
  font-weight: 600;
  margin-left: 10px;
  
  &:before {
    content: '🤖';
    margin-right: 6px;
  }
`;

const ChartsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 30px;
  align-items: start;

  @media (max-width: 1200px) {
    grid-template-columns: 1fr;
    gap: 16px;
  }
  
  @media (max-width: 768px) {
    gap: 12px;
  }
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
`;

const Card = styled.div`
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  padding: 20px;
  overflow: hidden;
  max-width: 100%;
  box-sizing: border-box;
  
  @media (max-width: 768px) {
    padding: 16px;
  }
`;

const CardTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 15px;
`;

const ErrorMessage = styled.div`
  color: #e53e3e;
  background-color: #fed7d7;
  border: 1px solid #feb2b2;
  padding: 12px;
  border-radius: 4px;
  margin: 20px 0;
`;

const AgentInsight = styled.div`
  background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
  border-left: 4px solid #667eea;
  padding: 16px;
  margin: 20px 0;
  border-radius: 4px;
`;

const InsightTitle = styled.h4`
  color: #667eea;
  margin: 0 0 8px;
  font-size: 1rem;
  font-weight: 600;
`;

const InsightText = styled.p`
  color: #4a5568;
  margin: 0;
  font-size: 0.875rem;
  line-height: 1.5;
`;

const AgentPortfolioAnalysis: React.FC = () => {
  const [pieChartData, setPieChartData] = useState<any[]>([]);
  const [lineChartData, setLineChartData] = useState<any[]>([]);
  const [barChartData, setBarChartData] = useState<any>(null);
  const [agentPredictions, setAgentPredictions] = useState<any>(null);
  const [agentAllocation, setAgentAllocation] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch all portfolio data from agents
        const [pieData, lineData, predictions, allocation] = await Promise.all([
          agentApiService.getPieChartData(),
          agentApiService.getLineChartData(),
          agentApiService.getPredictionResults(),
          agentApiService.getAllocationResults()
        ]);

        setPieChartData(pieData);
        setLineChartData(lineData);
        setAgentPredictions(predictions);
        setAgentAllocation(allocation);

        // Prepare bar chart data from line data
        if (lineData && lineData.length > 0) {
          const recent20Years = lineData.slice(-20);
          const barData = {
            labels: recent20Years.map((item: any) => item.year.toString()),
            datasets: [
              {
                label: 'Annual Returns (%) - Agent Processed',
                data: recent20Years.map((item: any) => item.annual_return),
                backgroundColor: recent20Years.map((item: any) => 
                  item.annual_return >= 0 ? '#38a169' : '#e53e3e'
                ),
                borderColor: recent20Years.map((item: any) => 
                  item.annual_return >= 0 ? '#2f855a' : '#c53030'
                ),
                borderWidth: 2
              }
            ]
          };
          setBarChartData(barData);
        }

      } catch (err) {
        setError('Failed to load agent portfolio data');
        console.error('Agent portfolio data fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <AnalysisContainer>
        <PageTitle>Agent Portfolio Analysis</PageTitle>
        <LoadingSpinner size="large" />
      </AnalysisContainer>
    );
  }

  if (error) {
    return (
      <AnalysisContainer>
        <PageTitle>Agent Portfolio Analysis</PageTitle>
        <ErrorMessage>{error}</ErrorMessage>
      </AnalysisContainer>
    );
  }

  return (
    <AnalysisContainer>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '30px' }}>
        <PageTitle style={{ margin: 0 }}>Agent Portfolio Analysis</PageTitle>
        <AgentBadge>AI-Optimized</AgentBadge>
      </div>

      {/* Agent Insights */}
      {agentPredictions && (
        <AgentInsight>
          <InsightTitle>🤖 Agent Predictions</InsightTitle>
          <InsightText>
            Expected Annual Return: <strong>{agentPredictions.expected_annual_return}%</strong> | 
            Predicted Volatility: <strong>{agentPredictions.predicted_volatility}%</strong> | 
            Market Regime: <strong>{agentPredictions.market_regime}</strong> | 
            Confidence: <strong>{agentPredictions.confidence_interval}</strong>
          </InsightText>
        </AgentInsight>
      )}

      {agentAllocation && (
        <AgentInsight>
          <InsightTitle>💼 Agent Allocation Strategy</InsightTitle>
          <InsightText>
            Risk Level: <strong>{agentAllocation.portfolio_metrics.risk_level}</strong> | 
            Method: <strong>{agentAllocation.portfolio_metrics.allocation_method}</strong> | 
            Expected Portfolio Return: <strong>{agentAllocation.portfolio_metrics.expected_return}%</strong>
          </InsightText>
        </AgentInsight>
      )}

      {/* Performance Metrics */}
      {agentPredictions && (
        <MetricsGrid>
          <MetricCard
            title="Expected Return"
            value={`${agentPredictions.expected_annual_return}%`}
            status={agentPredictions.expected_annual_return >= 10 ? 'HEALTHY' : 'WARNING'}
            loading={false}
          />
          <MetricCard
            title="Predicted Volatility"
            value={`${agentPredictions.predicted_volatility}%`}
            status={agentPredictions.predicted_volatility <= 20 ? 'HEALTHY' : 'WARNING'}
            loading={false}
          />
          <MetricCard
            title="Predicted Sharpe Ratio"
            value={agentPredictions.predicted_sharpe_ratio.toFixed(2)}
            status={agentPredictions.predicted_sharpe_ratio >= 0.5 ? 'HEALTHY' : 'WARNING'}
            loading={false}
          />
          <MetricCard
            title="Market Regime"
            value={agentPredictions.market_regime}
            status="HEALTHY"
            loading={false}
          />
        </MetricsGrid>
      )}

      {/* Charts */}
      <ChartsGrid>
        <Card>
          <PieChart
            data={pieChartData}
            title="Agent-Optimized Portfolio Allocation"
            height={280}
            loading={loading}
          />
        </Card>

        <Card>
          <LineChart
            data={lineChartData}
            title="Historical S&P 500 Performance (Agent Processed)"
            height={280}
            loading={loading}
          />
        </Card>
      </ChartsGrid>

      <ChartsGrid>
        {barChartData && (
          <Card>
            <BarChart
              data={barChartData}
              title="Annual Returns - Last 20 Years (Agent Analysis)"
              height={280}
              loading={loading}
            />
          </Card>
        )}

        {agentAllocation && (
          <Card>
            <CardTitle>Agent Allocation Rationale</CardTitle>
            <div style={{ marginBottom: '15px', fontSize: '0.9rem' }}>
              <strong>Asset Allocation:</strong>
              <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                <li>Stocks: {agentAllocation.asset_allocation.stocks}%</li>
                <li>Bonds: {agentAllocation.asset_allocation.bonds}%</li>
                <li>Alternatives: {agentAllocation.asset_allocation.alternatives}%</li>
              </ul>
            </div>
            
            <div style={{ marginBottom: '15px', fontSize: '0.9rem' }}>
              <strong>Recommendations:</strong>
              <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                {agentAllocation.recommendations.map((rec: string, index: number) => (
                  <li key={index} style={{ marginBottom: '6px', lineHeight: '1.4' }}>{rec}</li>
                ))}
              </ul>
            </div>
          </Card>
        )}
      </ChartsGrid>

      {/* Agent Key Insights */}
      {agentPredictions && agentPredictions.key_insights && (
        <Card>
          <CardTitle>🤖 Agent Key Insights</CardTitle>
          <ul style={{ margin: '10px 0', paddingLeft: '20px' }}>
            {agentPredictions.key_insights.map((insight: string, index: number) => (
              <li key={index} style={{ marginBottom: '8px', color: '#4a5568' }}>
                {insight}
              </li>
            ))}
          </ul>
        </Card>
      )}
    </AnalysisContainer>
  );
};

export default AgentPortfolioAnalysis;