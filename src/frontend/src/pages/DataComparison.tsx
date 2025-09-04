import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

/**
 * DATA COMPARISON COMPONENT
 * 
 * This component demonstrates the difference between:
 * 1. Raw Excel Data: Direct processing from historical_data_loader.py
 * 2. Agent-Processed Data: Full pipeline through DataCleaningAgent → AssetPredictorAgent → PortfolioAllocatorAgent
 * 
 * DATA FLOW VISUALIZATION:
 * 
 * RAW PATH:
 * Excel Sheet → historical_data_loader.py → /data/raw endpoint → This Component
 * 
 * AGENT PATH: 
 * Excel Sheet → DataCleaningAgent → AssetPredictorAgent → PortfolioAllocatorAgent → 
 * Orchestrator → /data/agent endpoint → This Component
 * 
 * The component fetches from both endpoints and displays results side-by-side
 * so you can see how agents enhance the raw data processing.
 */

interface RawDataResponse {
  processing_method: string;
  data_source: string;
  processed_by: string;
  agent_processing: boolean;
  timestamp: string;
  data: {
    performance_summary: any;
    risk_metrics: any;
    data_quality: any;
    line_chart_data: any[];
    annual_returns: any[];
  };
  metadata: {
    total_data_points: number;
    date_range: string;
    data_completeness: string;
  };
}

interface AgentDataResponse {
  processing_method: string;
  data_source: string;
  processed_by: string;
  agent_processing: boolean;
  timestamp: string;
  data: {
    cleaned_data_summary?: any;
    predictions?: any;
    portfolio_allocation?: any;
    agent_insights?: string[];
    // For real agent processing
    data_cleaning_results?: any;
    prediction_results?: any;
    allocation_results?: any;
    final_recommendations?: any;
  };
  agent_execution_log?: any[];
  execution_log?: any[];
}

// Styled Components
const Container = styled.div`
  min-height: 100vh;
  background-color: #f8fafc;
  padding: 24px;
`;

const MaxWidthContainer = styled.div`
  max-width: 1280px;
  margin: 0 auto;
`;

const PageHeader = styled.div`
  margin-bottom: 32px;
`;

const PageTitle = styled.h1`
  font-size: 2rem;
  font-weight: 700;
  color: #1a202c;
  margin: 0 0 8px;
`;

const PageSubtitle = styled.p`
  color: #718096;
  margin-bottom: 16px;
`;

const FlowDiagram = styled.div`
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  padding: 24px;
  margin-bottom: 24px;
`;

const FlowTitle = styled.h2`
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 16px;
`;

const FlowRow = styled.div`
  display: flex;
  align-items: center;
  padding: 12px;
  margin-bottom: 16px;
  border-radius: 4px;
  
  &.raw {
    background-color: #ebf8ff;
  }
  
  &.agent {
    background-color: #f0fff4;
  }
`;

const FlowLabel = styled.div`
  font-weight: 600;
  margin-right: 16px;
  min-width: 60px;
  
  &.raw {
    color: #3182ce;
  }
  
  &.agent {
    color: #38a169;
  }
`;

const FlowSteps = styled.div`
  display: flex;
  align-items: center;
  font-size: 0.875rem;
  gap: 8px;
`;

const FlowStep = styled.span`
  padding: 4px 8px;
  border-radius: 4px;
  
  &.raw {
    background-color: #bee3f8;
  }
  
  &.agent {
    background-color: #c6f6d5;
  }
`;

const FlowArrow = styled.span`
  margin: 0 4px;
`;

const RefreshButton = styled.button`
  padding: 8px 24px;
  background-color: #3182ce;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: #2c5aa0;
  }
`;

const GridContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: 24px;
  margin-bottom: 32px;
  
  @media (min-width: 1024px) {
    grid-template-columns: 1fr 1fr;
  }
`;

const DataCard = styled.div<{ borderColor: string }>`
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  border-left: 4px solid ${props => props.borderColor};
  padding: 24px;
`;

const CardHeader = styled.div`
  margin-bottom: 16px;
`;

const CardTitle = styled.h2`
  font-size: 1.25rem;
  font-weight: 700;
  color: #1a202c;
  margin: 0;
`;

const CardSubtitle = styled.p`
  font-size: 0.875rem;
  color: #718096;
  margin: 4px 0 8px;
`;

const StatusIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
`;

const StatusDot = styled.div<{ color: string }>`
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background-color: ${props => props.color};
`;

const Spinner = styled.div`
  width: 16px;
  height: 16px;
  border: 2px solid #e2e8f0;
  border-top: 2px solid #3182ce;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const StatusText = styled.span<{ color: string }>`
  font-size: 0.875rem;
  color: ${props => props.color};
`;

const LoadingPlaceholder = styled.div`
  .animate-pulse {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  }
  
  @keyframes pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: .5;
    }
  }
`;

const LoadingBar = styled.div<{ width: string; height: string }>`
  height: ${props => props.height};
  background-color: #e2e8f0;
  border-radius: 4px;
  width: ${props => props.width};
  margin-bottom: 8px;
`;

const ErrorContainer = styled.div`
  background-color: #fed7d7;
  border: 1px solid #feb2b2;
  border-radius: 4px;
  padding: 16px;
`;

const ErrorText = styled.p`
  color: #c53030;
  margin: 0 0 8px;
`;

const RetryButton = styled.button`
  padding: 8px 16px;
  background-color: #e53e3e;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: #c53030;
  }
`;

const InfoSection = styled.div`
  background-color: #f7fafc;
  border-radius: 4px;
  padding: 16px;
  margin-bottom: 16px;
`;

const InfoTitle = styled.h3`
  font-weight: 600;
  color: #4a5568;
  margin: 0 0 8px;
`;

const InfoList = styled.div`
  font-size: 0.875rem;
  
  p {
    margin: 4px 0;
    
    strong {
      font-weight: 600;
    }
  }
`;

const JsonContainer = styled.div`
  h3 {
    font-weight: 600;
    color: #4a5568;
    margin: 0 0 8px;
  }
`;

const JsonDisplay = styled.pre`
  background-color: #1a202c;
  color: #68d391;
  padding: 16px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  overflow: auto;
  max-height: 384px;
  white-space: pre-wrap;
`;

const ComparisonSection = styled.div`
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  padding: 24px;
`;

const ComparisonTitle = styled.h2`
  font-size: 1.25rem;
  font-weight: 700;
  margin-bottom: 16px;
`;

const ComparisonGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: 24px;
  
  @media (min-width: 768px) {
    grid-template-columns: 1fr 1fr;
  }
`;

const ComparisonColumn = styled.div`
  h3 {
    font-weight: 600;
    margin-bottom: 8px;
    
    &.raw {
      color: #3182ce;
    }
    
    &.agent {
      color: #38a169;
    }
  }
  
  ul {
    font-size: 0.875rem;
    color: #718096;
    list-style: none;
    padding: 0;
    margin: 0;
    
    li {
      margin-bottom: 4px;
    }
  }
`;

const DataComparison: React.FC = () => {
  // State for raw data (direct Excel processing)
  const [rawData, setRawData] = useState<RawDataResponse | null>(null);
  const [rawLoading, setRawLoading] = useState(true);
  const [rawError, setRawError] = useState<string | null>(null);

  // State for agent-processed data
  const [agentData, setAgentData] = useState<AgentDataResponse | null>(null);
  const [agentLoading, setAgentLoading] = useState(true);
  const [agentError, setAgentError] = useState<string | null>(null);

  /**
   * Fetch raw data from /data/raw endpoint
   * This shows data processed directly from Excel without agent enhancement
   */
  const fetchRawData = async () => {
    try {
      setRawLoading(true);
      setRawError(null);
      
      console.log('🔄 Fetching raw Excel data from /data/raw...');
      const response = await fetch('http://localhost:8000/data/raw');
      const result = await response.json();
      
      if (result.success) {
        setRawData(result.data);
        console.log('✅ Raw data loaded successfully:', result.data.processing_method);
      } else {
        throw new Error('Failed to fetch raw data');
      }
    } catch (error) {
      console.error('❌ Error fetching raw data:', error);
      setRawError(error instanceof Error ? error.message : 'Unknown error');
    } finally {
      setRawLoading(false);
    }
  };

  /**
   * Fetch agent-processed data from /data/agent endpoint
   * This shows data enhanced by the full agent pipeline
   */
  const fetchAgentData = async () => {
    try {
      setAgentLoading(true);
      setAgentError(null);
      
      console.log('🔄 Fetching agent-processed data from /data/agent...');
      const response = await fetch('http://localhost:8000/data/agent');
      const result = await response.json();
      
      if (result.success) {
        setAgentData(result.data);
        console.log('✅ Agent data loaded successfully:', result.data.processing_method);
      } else {
        throw new Error('Failed to fetch agent data');
      }
    } catch (error) {
      console.error('❌ Error fetching agent data:', error);
      setAgentError(error instanceof Error ? error.message : 'Unknown error');
    } finally {
      setAgentLoading(false);
    }
  };

  // Fetch both datasets when component mounts
  useEffect(() => {
    console.log('🚀 DataComparison component mounted - fetching both raw and agent data...');
    fetchRawData();
    fetchAgentData();
  }, []);

  /**
   * Render a data card with title, status, and JSON content
   */
  const renderDataCard = (
    title: string,
    subtitle: string,
    data: any,
    loading: boolean,
    error: string | null,
    borderColor: string
  ) => (
    <DataCard borderColor={borderColor}>
      {/* Card Header */}
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardSubtitle>{subtitle}</CardSubtitle>
        
        {/* Status Indicator */}
        <StatusIndicator>
          {loading && (
            <>
              <Spinner />
              <StatusText color="#3182ce">Loading...</StatusText>
            </>
          )}
          {error && (
            <>
              <StatusDot color="#e53e3e" />
              <StatusText color="#e53e3e">Error: {error}</StatusText>
            </>
          )}
          {!loading && !error && data && (
            <>
              <StatusDot color="#38a169" />
              <StatusText color="#38a169">Data loaded successfully</StatusText>
            </>
          )}
        </StatusIndicator>
      </CardHeader>

      {/* Card Content */}
      <div>
        {loading && (
          <LoadingPlaceholder className="animate-pulse">
            <LoadingBar width="75%" height="16px" />
            <LoadingBar width="50%" height="16px" />
            <LoadingBar width="100%" height="128px" />
          </LoadingPlaceholder>
        )}

        {error && (
          <ErrorContainer>
            <ErrorText>Failed to load data: {error}</ErrorText>
            <RetryButton onClick={title.includes('Raw') ? fetchRawData : fetchAgentData}>
              Retry
            </RetryButton>
          </ErrorContainer>
        )}

        {!loading && !error && data && (
          <>
            {/* Data Summary */}
            <InfoSection>
              <InfoTitle>Processing Info</InfoTitle>
              <InfoList>
                <p><strong>Method:</strong> {data.processing_method}</p>
                <p><strong>Source:</strong> {data.data_source}</p>
                <p><strong>Processed by:</strong> {data.processed_by}</p>
                <p><strong>Agent Processing:</strong> {data.agent_processing ? '✅ Yes' : '❌ No'}</p>
                <p><strong>Timestamp:</strong> {new Date(data.timestamp).toLocaleString()}</p>
              </InfoList>
            </InfoSection>

            {/* JSON Data Display */}
            <JsonContainer>
              <h3>Raw JSON Output</h3>
              <JsonDisplay>
                {JSON.stringify(data, null, 2)}
              </JsonDisplay>
            </JsonContainer>
          </>
        )}
      </div>
    </DataCard>
  );

  return (
    <Container>
      {/* Page Header */}
      <MaxWidthContainer>
        <PageHeader>
          <PageTitle>Data Processing Comparison</PageTitle>
          <PageSubtitle>
            Compare raw Excel processing vs. enhanced agent pipeline processing
          </PageSubtitle>
          
          {/* Data Flow Diagram */}
          <FlowDiagram>
            <FlowTitle>Data Flow Architecture</FlowTitle>
            
            <div>
              {/* Raw Data Flow */}
              <FlowRow className="raw">
                <FlowLabel className="raw">RAW:</FlowLabel>
                <FlowSteps>
                  <FlowStep className="raw">Excel Sheet</FlowStep>
                  <FlowArrow>→</FlowArrow>
                  <FlowStep className="raw">historical_data_loader.py</FlowStep>
                  <FlowArrow>→</FlowArrow>
                  <FlowStep className="raw">/data/raw</FlowStep>
                  <FlowArrow>→</FlowArrow>
                  <FlowStep className="raw">Frontend</FlowStep>
                </FlowSteps>
              </FlowRow>
              
              {/* Agent Data Flow */}
              <FlowRow className="agent">
                <FlowLabel className="agent">AGENT:</FlowLabel>
                <FlowSteps>
                  <FlowStep className="agent">Excel Sheet</FlowStep>
                  <FlowArrow>→</FlowArrow>
                  <FlowStep className="agent">DataCleaningAgent</FlowStep>
                  <FlowArrow>→</FlowArrow>
                  <FlowStep className="agent">AssetPredictorAgent</FlowStep>
                  <FlowArrow>→</FlowArrow>
                  <FlowStep className="agent">PortfolioAllocatorAgent</FlowStep>
                  <FlowArrow>→</FlowArrow>
                  <FlowStep className="agent">/data/agent</FlowStep>
                  <FlowArrow>→</FlowArrow>
                  <FlowStep className="agent">Frontend</FlowStep>
                </FlowSteps>
              </FlowRow>
            </div>
          </FlowDiagram>

          {/* Refresh Button */}
          <div>
            <RefreshButton
              onClick={() => {
                fetchRawData();
                fetchAgentData();
              }}
            >
              🔄 Refresh Both Datasets
            </RefreshButton>
          </div>
        </PageHeader>

        {/* Side-by-Side Data Display */}
        <GridContainer>
          {/* Raw Data Card */}
          {renderDataCard(
            "Raw Excel Data",
            "Direct processing from historical_data_loader.py",
            rawData,
            rawLoading,
            rawError,
            "#3182ce"
          )}

          {/* Agent-Processed Data Card */}
          {renderDataCard(
            "Agent Processed Data", 
            "Enhanced processing through full agent pipeline",
            agentData,
            agentLoading,
            agentError,
            "#38a169"
          )}
        </GridContainer>

        {/* Comparison Summary */}
        {!rawLoading && !agentLoading && rawData && agentData && (
          <ComparisonSection>
            <ComparisonTitle>Key Differences</ComparisonTitle>
            
            <ComparisonGrid>
              <ComparisonColumn>
                <h3 className="raw">Raw Data Processing</h3>
                <ul>
                  <li>• Direct Excel file parsing</li>
                  <li>• Basic data validation</li>
                  <li>• Historical metrics calculation</li>
                  <li>• No predictive analysis</li>
                  <li>• No portfolio optimization</li>
                </ul>
              </ComparisonColumn>
              
              <ComparisonColumn>
                <h3 className="agent">Agent Processing</h3>
                <ul>
                  <li>• Advanced data cleaning & validation</li>
                  <li>• Pattern recognition & analysis</li>
                  <li>• Forward-looking predictions</li>
                  <li>• Risk assessment & modeling</li>
                  <li>• Optimized portfolio allocation</li>
                  <li>• Actionable recommendations</li>
                </ul>
              </ComparisonColumn>
            </ComparisonGrid>
          </ComparisonSection>
        )}
      </MaxWidthContainer>
    </Container>
  );
};

export default DataComparison;