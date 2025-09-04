# Implementation Plan

- [x] 1. Implement ML Models for Asset Returns

  - Create AssetReturnModels class with scikit-learn integration
  - Implement individual ML models for S&P 500, Small Cap, T-Bills, T-Bonds, Corporate Bonds, Real Estate, Gold
  - Create model training pipeline using 50-year historical data from Excel
  - Implement return prediction methods for each asset class
  - Add model validation and accuracy testing
  - Write unit tests for model training and prediction
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. Create Langgraph Agent Workflow

  - Install and configure Langgraph framework
  - Create ReturnPredictionAgent class with LLM integration
  - Implement PortfolioAllocationAgent class with decision logic
  - Create RebalancingAgent class for time-based allocation changes
  - Implement Langgraph workflow to coordinate agent execution
  - Add error handling and retry logic for agent failures
  - Write integration tests for complete agent workflow
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 3. Implement Portfolio Allocation Engine

  - Create PortfolioAllocationEngine class with risk-based strategies
  - Implement low risk allocation (Conservative: 70% bonds, 30% equity)
  - Implement moderate risk allocation (Balanced: 50% bonds, 50% equity)
  - Implement high risk allocation (Aggressive: 20% bonds, 80% equity)
  - Add allocation validation to ensure 100% total
  - Write unit tests for each risk profile allocation
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 4. Create Portfolio Rebalancing System

  - Extend RebalancingAgent with time-based rebalancing rules
  - Implement equity reduction logic (e.g., reduce 5% every 2 years)
  - Create rebalancing schedule calculation methods
  - Add rebalancing impact on portfolio projections
  - Implement visualization of allocation changes over time
  - Write unit tests for rebalancing scenarios
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 5. Implement Investment Options Calculators

  - Create InvestmentCalculators class with compound growth formulas
  - Implement lump sum calculator with single initial investment
  - Create SIP calculator with monthly systematic investments
  - Implement SWP calculator with systematic withdrawals
  - Add investment projection generation for each strategy
  - Write unit tests for all investment calculation scenarios
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 6. Create Basic Dashboard Frontend

  - Implement PortfolioDashboard React component
  - Create AllocationPieChart component for portfolio visualization
  - Add ReturnsLineChart component for growth over time
  - Implement PortfolioValueChart for value appreciation display
  - Create responsive design for dashboard layout
  - Write component tests and integration with backend APIs
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 7. Implement End-to-End User Flow

  - Create UserInputForm React component for investment parameters
  - Implement input validation for amount, tenure, and risk profile
  - Create API endpoints to process user inputs through agent workflow
  - Add portfolio recommendation generation and display
  - Implement return calculation and visualization pipeline
  - Create complete user journey from input to results
  - Write end-to-end integration tests for complete flow
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 8. Integrate ML Models with Agent Workflow

  - Connect trained ML models to ReturnPredictionAgent
  - Implement model prediction calls within agent execution
  - Add model results to portfolio allocation decision process
  - Create error handling for model prediction failures
  - Write integration tests for ML model and agent interaction
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4_

- [x] 9. Create Backend API Endpoints

  - Implement /api/portfolio/allocate endpoint for portfolio generation
  - Create /api/investment/calculate endpoint for investment projections
  - Add /api/rebalancing/simulate endpoint for rebalancing scenarios
  - Implement /api/models/predict endpoint for ML return predictions
  - Add proper error handling and input validation for all endpoints
  - Write API integration tests with sample data
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 10. Final Integration and Testing
  - Integrate all components into complete working system
  - Test complete user workflow from input to dashboard display
  - Validate ML model predictions against historical performance
  - Test agent workflow with various user input scenarios
  - Verify portfolio allocations and rebalancing calculations
  - Create comprehensive test suite for all core components
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4, 7.1, 7.2, 7.3, 7.4_
