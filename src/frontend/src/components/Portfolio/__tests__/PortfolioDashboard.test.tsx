import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import PortfolioDashboard from '../PortfolioDashboard';

// Mock the API service
jest.mock('../../../services/api', () => ({
  apiService: {
    getPortfolioAllocation: jest.fn(),
    getPieChartData: jest.fn(),
    getLineChartData: jest.fn(),
  }
}));

import { apiService } from '../../../services/api';
const mockApiService = apiService as jest.Mocked<typeof apiService>;

// Mock the chart components
jest.mock('../AllocationPieChart', () => {
  return function MockAllocationPieChart({ title, loading, error }: any) {
    if (loading) return <div data-testid="allocation-loading">Loading...</div>;
    if (error) return <div data-testid="allocation-error">{error}</div>;
    return <div data-testid="allocation-chart">{title}</div>;
  };
});

jest.mock('../ReturnsLineChart', () => {
  return function MockReturnsLineChart({ title, loading, error }: any) {
    if (loading) return <div data-testid="returns-loading">Loading...</div>;
    if (error) return <div data-testid="returns-error">{error}</div>;
    return <div data-testid="returns-chart">{title}</div>;
  };
});

jest.mock('../PortfolioValueChart', () => {
  return function MockPortfolioValueChart({ title, loading, error }: any) {
    if (loading) return <div data-testid="value-loading">Loading...</div>;
    if (error) return <div data-testid="value-error">{error}</div>;
    return <div data-testid="value-chart">{title}</div>;
  };
});

describe('PortfolioDashboard', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  const mockApiData = {
    allocation: { sp500: 45, small_cap: 20, bonds: 25, real_estate: 7.5, gold: 2.5 },
    pieData: [
      { name: 'S&P 500', value: 45, color: '#1f77b4', percentage: '45.0%' },
      { name: 'Bonds', value: 25, color: '#2ca02c', percentage: '25.0%' }
    ],
    lineData: [
      { year: 0, portfolio_value: 100000, formatted_value: '100,000' },
      { year: 1, portfolio_value: 110000, formatted_value: '110,000' }
    ]
  };

  it('renders loading state initially', () => {
    mockApiService.getPortfolioAllocation.mockImplementation(() => new Promise(() => {}));
    mockApiService.getPieChartData.mockImplementation(() => new Promise(() => {}));
    mockApiService.getLineChartData.mockImplementation(() => new Promise(() => {}));

    render(<PortfolioDashboard />);
    
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });

  it('renders dashboard with charts when data loads successfully', async () => {
    mockApiService.getPortfolioAllocation.mockResolvedValue(mockApiData.allocation);
    mockApiService.getPieChartData.mockResolvedValue(mockApiData.pieData);
    mockApiService.getLineChartData.mockResolvedValue(mockApiData.lineData);

    render(<PortfolioDashboard />);

    await waitFor(() => {
      expect(screen.getByText('Portfolio Dashboard')).toBeInTheDocument();
      expect(screen.getByTestId('allocation-chart')).toBeInTheDocument();
      expect(screen.getByTestId('returns-chart')).toBeInTheDocument();
      expect(screen.getByTestId('value-chart')).toBeInTheDocument();
    });
  });

  it('renders error state when API calls fail', async () => {
    const errorMessage = 'API Error';
    mockApiService.getPortfolioAllocation.mockRejectedValue(new Error(errorMessage));
    mockApiService.getPieChartData.mockRejectedValue(new Error(errorMessage));
    mockApiService.getLineChartData.mockRejectedValue(new Error(errorMessage));

    render(<PortfolioDashboard />);

    await waitFor(() => {
      expect(screen.getByText(/Failed to load portfolio dashboard data/)).toBeInTheDocument();
    });
  });

  it('displays user input information when provided', async () => {
    mockApiService.getPortfolioAllocation.mockResolvedValue(mockApiData.allocation);
    mockApiService.getPieChartData.mockResolvedValue(mockApiData.pieData);
    mockApiService.getLineChartData.mockResolvedValue(mockApiData.lineData);

    const userInput = {
      investment_amount: 100000,
      investment_tenure: 10,
      risk_profile: 'moderate' as const,
      investment_type: 'lump_sum' as const
    };

    render(<PortfolioDashboard userInput={userInput} />);

    await waitFor(() => {
      expect(screen.getByText(/Investment analysis for moderate risk profile/)).toBeInTheDocument();
    });
  });

  it('calls API services on mount', async () => {
    mockApiService.getPortfolioAllocation.mockResolvedValue(mockApiData.allocation);
    mockApiService.getPieChartData.mockResolvedValue(mockApiData.pieData);
    mockApiService.getLineChartData.mockResolvedValue(mockApiData.lineData);

    render(<PortfolioDashboard />);

    await waitFor(() => {
      expect(mockApiService.getPortfolioAllocation).toHaveBeenCalledTimes(1);
      expect(mockApiService.getPieChartData).toHaveBeenCalledTimes(1);
      expect(mockApiService.getLineChartData).toHaveBeenCalledTimes(2); // Called twice for different charts
    });
  });
});