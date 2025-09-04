# Requirements Document

## Introduction

This specification focuses on the 7 mandatory core components for the Financial Returns Optimizer. The system will provide ML-based asset return predictions, LLM agent-driven portfolio allocation, and a complete end-to-end investment planning workflow.

## Requirements

### Requirement 1: ML Models for Asset Returns

**User Story:** As an investor, I want ML models to predict returns for each asset class so that I can make data-driven investment decisions.

#### Acceptance Criteria

1. WHEN historical data is processed THEN the system SHALL train ML models for S&P 500, Small Cap, T-Bills, T-Bonds, Corporate Bonds, Real Estate, and Gold
2. WHEN models are trained THEN the system SHALL use 50-year historical data from the provided Excel file
3. WHEN return predictions are requested THEN the system SHALL generate expected returns for each asset class
4. WHEN models are evaluated THEN the system SHALL validate accuracy against historical performance

### Requirement 2: Langgraph + LLM Agents

**User Story:** As a system, I want LLM agents coordinated through Langgraph so that portfolio decisions are made intelligently.

#### Acceptance Criteria

1. WHEN portfolio allocation is needed THEN the system SHALL use a Portfolio Allocation Agent
2. WHEN return predictions are required THEN the system SHALL use a Return Prediction Agent
3. WHEN agents execute THEN the system SHALL coordinate them through Langgraph workflow
4. WHEN agent workflow runs THEN the system SHALL handle agent communication and data flow

### Requirement 3: Portfolio Allocation Engine

**User Story:** As an investor, I want different risk-based portfolio allocations so that I can invest according to my risk tolerance.

#### Acceptance Criteria

1. WHEN low risk is selected THEN the system SHALL provide conservative allocation (high bonds, low equity)
2. WHEN moderate risk is selected THEN the system SHALL provide balanced allocation (mixed bonds and equity)
3. WHEN high risk is selected THEN the system SHALL provide aggressive allocation (high equity, low bonds)
4. WHEN allocation is generated THEN the system SHALL ensure total allocation equals 100%

### Requirement 4: Portfolio Rebalancing

**User Story:** As an investor, I want to change my portfolio allocation over time so that I can adjust my risk as I age.

#### Acceptance Criteria

1. WHEN rebalancing rules are set THEN the system SHALL support changing allocations over time
2. WHEN time-based rebalancing is configured THEN the system SHALL reduce equity by specified percentage over years
3. WHEN rebalancing occurs THEN the system SHALL recalculate portfolio projections
4. WHEN rebalancing is simulated THEN the system SHALL show allocation changes over investment period

### Requirement 5: Investment Options

**User Story:** As an investor, I want different investment strategies so that I can choose how to invest my money.

#### Acceptance Criteria

1. WHEN lump sum is selected THEN the system SHALL calculate growth from single initial investment
2. WHEN SIP is selected THEN the system SHALL calculate growth from systematic monthly investments
3. WHEN SWP is selected THEN the system SHALL calculate portfolio value with systematic withdrawals
4. WHEN investment option is chosen THEN the system SHALL provide projections for selected strategy

### Requirement 6: Basic Dashboard

**User Story:** As an investor, I want to visualize my portfolio so that I can understand my investment plan.

#### Acceptance Criteria

1. WHEN dashboard loads THEN the system SHALL show portfolio allocation across asset classes
2. WHEN returns are calculated THEN the system SHALL plot returns over investment time period
3. WHEN projections are made THEN the system SHALL display portfolio value growth over time
4. WHEN visualizations are shown THEN the system SHALL provide clear, readable charts

### Requirement 7: End-to-End Flow

**User Story:** As an investor, I want a complete investment planning process so that I can get from inputs to recommendations.

#### Acceptance Criteria

1. WHEN user provides inputs THEN the system SHALL accept investment amount, tenure, and risk profile
2. WHEN inputs are received THEN the system SHALL process them through the agent workflow
3. WHEN agents complete processing THEN the system SHALL generate portfolio allocation recommendations
4. WHEN calculations are done THEN the system SHALL calculate and display expected returns and growth