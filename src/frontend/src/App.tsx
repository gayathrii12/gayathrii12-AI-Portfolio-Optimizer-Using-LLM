import React from 'react';
import { Routes, Route } from 'react-router-dom';
import styled from 'styled-components';
import Header from './components/Layout/Header';
import Sidebar from './components/Layout/Sidebar';
import RealDataDashboard from './pages/RealDataDashboard';
import AgentDashboard from './pages/AgentDashboard';
import PortfolioAnalysis from './pages/PortfolioAnalysis';
import AgentPortfolioAnalysis from './pages/AgentPortfolioAnalysis';
import PortfolioDashboardPage from './pages/PortfolioDashboardPage';
import DataQuality from './pages/DataQuality';
import InvestmentPlanner from './pages/InvestmentPlanner';

const AppContainer = styled.div`
  display: flex;
  min-height: 100vh;
  background-color: #f8fafc;
`;

const MainContent = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
`;

const ContentArea = styled.main`
  flex: 1;
  padding: 20px;
  overflow-y: auto;
`;

function App() {
  return (
    <AppContainer>
      <Sidebar />
      <MainContent>
        <Header />
        <ContentArea>
          <Routes>
            <Route path="/" element={<RealDataDashboard />} />
            <Route path="/dashboard" element={<RealDataDashboard />} />
            <Route path="/agent-dashboard" element={<AgentDashboard />} />
            <Route path="/portfolio" element={<PortfolioAnalysis />} />
            <Route path="/portfolio-dashboard" element={<PortfolioDashboardPage />} />
            <Route path="/agent-portfolio" element={<AgentPortfolioAnalysis />} />
            <Route path="/data-quality" element={<DataQuality />} />
            <Route path="/investment-planner" element={<InvestmentPlanner />} />
          </Routes>
        </ContentArea>
      </MainContent>
    </AppContainer>
  );
}

export default App;