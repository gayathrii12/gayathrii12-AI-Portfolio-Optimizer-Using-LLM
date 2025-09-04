import React from 'react';
import { PortfolioDashboard } from '../components/Portfolio';

const PortfolioDashboardPage: React.FC = () => {
  // This could be populated from user input form or URL parameters
  const mockUserInput = {
    investment_amount: 100000,
    investment_tenure: 10,
    risk_profile: 'moderate' as const,
    investment_type: 'lump_sum' as const
  };

  return <PortfolioDashboard userInput={mockUserInput} />;
};

export default PortfolioDashboardPage;