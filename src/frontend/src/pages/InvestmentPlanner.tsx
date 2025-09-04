import React, { useState } from 'react';
import styled from 'styled-components';
import { UserInputForm, UserInputData } from '../components/UserInput';
import { PortfolioRecommendation, PortfolioRecommendationData } from '../components/Portfolio';
import { investmentPlannerService } from '../services/investmentPlannerService';

const Container = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 2rem 0;
`;

const ContentWrapper = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 3rem;
  color: white;
`;

const Title = styled.h1`
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 1rem;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
`;

const Subtitle = styled.p`
  font-size: 1.2rem;
  opacity: 0.9;
  max-width: 600px;
  margin: 0 auto;
  line-height: 1.6;
`;

const StepIndicator = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 2rem;
  gap: 1rem;
`;

const Step = styled.div<{ active: boolean; completed: boolean }>`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  border-radius: 25px;
  background: ${props => 
    props.completed ? '#38a169' : 
    props.active ? 'rgba(255, 255, 255, 0.2)' : 
    'rgba(255, 255, 255, 0.1)'
  };
  color: white;
  font-weight: 500;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
`;

const StepNumber = styled.div<{ active: boolean; completed: boolean }>`
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: ${props => 
    props.completed ? 'white' : 
    props.active ? 'rgba(255, 255, 255, 0.3)' : 
    'rgba(255, 255, 255, 0.1)'
  };
  color: ${props => props.completed ? '#38a169' : 'white'};
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.875rem;
  font-weight: 600;
`;

const ErrorContainer = styled.div`
  background: #fed7d7;
  border: 1px solid #feb2b2;
  border-radius: 8px;
  padding: 1rem;
  margin: 1rem 0;
  color: #c53030;
`;

const ErrorTitle = styled.h3`
  margin: 0 0 0.5rem 0;
  font-size: 1rem;
  font-weight: 600;
`;

const ErrorMessage = styled.p`
  margin: 0;
  font-size: 0.875rem;
`;

const BackButton = styled.button`
  background: rgba(255, 255, 255, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.3);
  color: white;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  backdrop-filter: blur(10px);
  transition: all 0.2s;
  margin-bottom: 2rem;

  &:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: translateY(-1px);
  }
`;

interface InvestmentPlannerState {
  step: 'input' | 'results';
  userInput: UserInputData | null;
  recommendation: PortfolioRecommendationData | null;
  loading: boolean;
  error: string | null;
}

const InvestmentPlanner: React.FC = () => {
  const [state, setState] = useState<InvestmentPlannerState>({
    step: 'input',
    userInput: null,
    recommendation: null,
    loading: false,
    error: null
  });

  const handleUserInput = async (inputData: UserInputData) => {
    setState(prev => ({
      ...prev,
      loading: true,
      error: null,
      userInput: inputData
    }));

    try {
      // Call the investment planner service to get portfolio recommendation
      const recommendation = await investmentPlannerService.generatePortfolioRecommendation(inputData);
      
      // Validate that the recommendation has all required data
      if (!recommendation || !recommendation.summary || !recommendation.allocation || !recommendation.risk_metrics || !recommendation.projections) {
        throw new Error('Incomplete portfolio recommendation data received');
      }
      
      setState(prev => ({
        ...prev,
        step: 'results',
        recommendation,
        loading: false
      }));
    } catch (error) {
      console.error('Error generating portfolio recommendation:', error);
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to generate portfolio recommendation'
      }));
    }
  };

  const handleBackToInput = () => {
    setState(prev => ({
      ...prev,
      step: 'input',
      recommendation: null,
      error: null
    }));
  };

  const renderStepIndicator = () => (
    <StepIndicator>
      <Step active={state.step === 'input'} completed={state.step === 'results'}>
        <StepNumber active={state.step === 'input'} completed={state.step === 'results'}>
          {state.step === 'results' ? '✓' : '1'}
        </StepNumber>
        Investment Details
      </Step>
      <div style={{ width: '2rem', height: '2px', background: 'rgba(255, 255, 255, 0.3)' }} />
      <Step active={state.step === 'results'} completed={false}>
        <StepNumber active={state.step === 'results'} completed={false}>
          2
        </StepNumber>
        Portfolio Recommendation
      </Step>
    </StepIndicator>
  );

  const renderError = () => {
    if (!state.error) return null;

    return (
      <ErrorContainer>
        <ErrorTitle>Unable to Generate Recommendation</ErrorTitle>
        <ErrorMessage>{state.error}</ErrorMessage>
      </ErrorContainer>
    );
  };

  return (
    <Container>
      <ContentWrapper>
        <Header>
          <Title>AI-Powered Investment Planner</Title>
          <Subtitle>
            Get personalized portfolio recommendations based on advanced ML models and intelligent agent analysis
          </Subtitle>
        </Header>

        {renderStepIndicator()}

        {state.step === 'input' && (
          <>
            {renderError()}
            <UserInputForm 
              onSubmit={handleUserInput} 
              loading={state.loading}
            />
          </>
        )}

        {state.step === 'results' && (
          <>
            <BackButton onClick={handleBackToInput}>
              ← Back to Investment Details
            </BackButton>
            <PortfolioRecommendation 
              data={state.recommendation || undefined}
              loading={state.loading}
            />
          </>
        )}
      </ContentWrapper>
    </Container>
  );
};

export default InvestmentPlanner;