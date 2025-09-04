import React from 'react';
import styled from 'styled-components';

const RecommendationsContainer = styled.div`
  background: white;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  border: 1px solid #e2e8f0;
  margin-top: 20px;
`;

const Title = styled.h3`
  font-size: 1.25rem;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 16px;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const RecommendationsList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const RecommendationItem = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px;
  background: #f8fafc;
  border-radius: 6px;
  border-left: 4px solid #3182ce;
`;

const RecommendationIcon = styled.div`
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #3182ce;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 600;
  flex-shrink: 0;
  margin-top: 2px;
`;

const RecommendationContent = styled.div`
  flex: 1;
`;

const RecommendationTitle = styled.h4`
  font-size: 0.875rem;
  font-weight: 600;
  color: #2d3748;
  margin: 0 0 4px;
`;

const RecommendationText = styled.p`
  font-size: 0.8rem;
  color: #4a5568;
  margin: 0;
  line-height: 1.4;
`;

interface KeyRecommendationsProps {
  userInput?: {
    investment_amount: number;
    investment_tenure: number;
    risk_profile: 'low' | 'moderate' | 'high';
    investment_type: 'lump_sum' | 'sip' | 'swp';
  };
}

const KeyRecommendations: React.FC<KeyRecommendationsProps> = ({ userInput }) => {
  const getRecommendations = () => {
    if (!userInput) {
      return [
        {
          title: "Diversification Strategy",
          text: "Maintain a balanced portfolio across different asset classes to reduce risk and optimize returns."
        },
        {
          title: "Regular Rebalancing",
          text: "Review and rebalance your portfolio quarterly to maintain target allocations."
        },
        {
          title: "Long-term Perspective",
          text: "Focus on long-term growth rather than short-term market fluctuations."
        },
        {
          title: "Cost Management",
          text: "Keep investment costs low by choosing low-fee index funds and ETFs."
        }
      ];
    }

    const { risk_profile, investment_tenure, investment_type } = userInput;
    const recommendations = [];

    // Risk-based recommendations
    if (risk_profile === 'low') {
      recommendations.push({
        title: "Conservative Allocation",
        text: "Focus on bonds and stable value funds with 60-70% fixed income allocation for capital preservation."
      });
    } else if (risk_profile === 'moderate') {
      recommendations.push({
        title: "Balanced Approach",
        text: "Maintain 60% equity and 40% bonds allocation for balanced growth and stability."
      });
    } else {
      recommendations.push({
        title: "Growth-Oriented Strategy",
        text: "Allocate 80-90% to equities for maximum long-term growth potential."
      });
    }

    // Tenure-based recommendations
    if (investment_tenure >= 20) {
      recommendations.push({
        title: "Long-term Advantage",
        text: "With 20+ years, you can weather market volatility. Consider higher equity allocation."
      });
    } else if (investment_tenure >= 10) {
      recommendations.push({
        title: "Medium-term Strategy",
        text: "Balance growth and stability with gradual shift to conservative assets as you approach goals."
      });
    } else {
      recommendations.push({
        title: "Short-term Focus",
        text: "Prioritize capital preservation with higher allocation to bonds and stable investments."
      });
    }

    // Investment type recommendations
    if (investment_type === 'sip') {
      recommendations.push({
        title: "SIP Benefits",
        text: "Dollar-cost averaging through SIP reduces timing risk and builds discipline."
      });
    } else if (investment_type === 'lump_sum') {
      recommendations.push({
        title: "Lump Sum Strategy",
        text: "Consider phased investment over 6-12 months to reduce market timing risk."
      });
    }

    // General recommendation
    recommendations.push({
      title: "Regular Monitoring",
      text: "Review portfolio performance quarterly and rebalance annually or when allocations drift 5%+ from targets."
    });

    return recommendations.slice(0, 4); // Limit to 4 recommendations
  };

  const recommendations = getRecommendations();

  return (
    <RecommendationsContainer>
      <Title>
        💡 Key Recommendations
      </Title>
      <RecommendationsList>
        {recommendations.map((rec, index) => (
          <RecommendationItem key={index}>
            <RecommendationIcon>{index + 1}</RecommendationIcon>
            <RecommendationContent>
              <RecommendationTitle>{rec.title}</RecommendationTitle>
              <RecommendationText>{rec.text}</RecommendationText>
            </RecommendationContent>
          </RecommendationItem>
        ))}
      </RecommendationsList>
    </RecommendationsContainer>
  );
};

export default KeyRecommendations;