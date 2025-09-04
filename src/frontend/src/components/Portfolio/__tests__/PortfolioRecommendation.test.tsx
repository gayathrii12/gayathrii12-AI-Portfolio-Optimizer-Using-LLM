import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import PortfolioRecommendation, { PortfolioRecommendationData } from '../PortfolioRecommendation';

// Mock recharts components
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  PieChart: ({ children }: any) => <div data-testid="pie-chart">{children}</div>,
  Pie: () => <div data-testid="pie" />,
  Cell: () => <div data-testid="cell" />,
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />
}));

describe('PortfolioRecommendation', () => {
  const mockData: PortfolioRecommendationData = {
    allocation: {
      sp500: 40.0,
      small_cap: 20.0,
      bonds: 25.0,
      real_estate: 10.0,
      gold: 5.0
    },
    projections: [
      {
        year: 0,
        portfolio_value: 100000,
        annual_return: 0,
        cumulative_return: 0
      },
      {
        year: 5,
        portfolio_value: 150000,
        annual_return: 8.5,
        cumulative_return: 50.0
      },
      {
        year: 10,
        portfolio_value: 220000,
        annual_return: 9.2,
        cumulative_return: 120.0
      }
    ],
    risk_metrics: {
      expected_return: 11.0,
      volatility: 12.0,
      sharpe_ratio: 0.80
    },
    summary: {
      initial_investment: 100000,
      final_value: 220000,
      total_return: 120000,
      investment_type: 'lumpsum',
      tenure_years: 10,
      risk_profile: 'Moderate'
    }
  };

  const renderComponent = (props = {}) => {
    return render(
      <PortfolioRecommendation
        data={mockData}
        {...props}
      />
    );
  };

  test('renders loading state when loading prop is true', () => {
    render(<PortfolioRecommendation data={mockData} loading={true} />);
    
    expect(screen.getByText(/generating your personalized portfolio recommendation/i)).toBeInTheDocument();
  });

  test('renders portfolio recommendation with correct title and subtitle', () => {
    renderComponent();

    expect(screen.getByText('Your Personalized Portfolio Recommendation')).toBeInTheDocument();
    expect(screen.getByText(/based on your moderate risk profile and 10-year investment horizon/i)).toBeInTheDocument();
  });

  test('displays asset allocation section', () => {
    renderComponent();

    expect(screen.getByText('Recommended Asset Allocation')).toBeInTheDocument();
    
    // Check if all asset classes are displayed
    expect(screen.getByText('S&P 500')).toBeInTheDocument();
    expect(screen.getByText('US Small Cap')).toBeInTheDocument();
    expect(screen.getByText('Bonds')).toBeInTheDocument();
    expect(screen.getByText('Real Estate')).toBeInTheDocument();
    expect(screen.getByText('Gold')).toBeInTheDocument();

    // Check if percentages are displayed correctly
    expect(screen.getByText('40.0%')).toBeInTheDocument(); // S&P 500
    expect(screen.getByText('20.0%')).toBeInTheDocument(); // Small Cap
    expect(screen.getByText('25.0%')).toBeInTheDocument(); // Bonds
    expect(screen.getByText('10.0%')).toBeInTheDocument(); // Real Estate
    expect(screen.getByText('5.0%')).toBeInTheDocument();  // Gold
  });

  test('displays portfolio growth projection section', () => {
    renderComponent();

    expect(screen.getByText('Portfolio Growth Projection')).toBeInTheDocument();
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
  });

  test('displays risk metrics correctly', () => {
    renderComponent();

    // Check if risk metrics are displayed
    expect(screen.getByText('11.0%')).toBeInTheDocument(); // Expected Return
    expect(screen.getByText('12.0%')).toBeInTheDocument(); // Volatility
    expect(screen.getByText('0.80')).toBeInTheDocument();  // Sharpe Ratio

    // Check metric labels
    expect(screen.getByText('Expected Return')).toBeInTheDocument();
    expect(screen.getByText('Volatility')).toBeInTheDocument();
    expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();
  });

  test('displays investment summary correctly', () => {
    renderComponent();

    expect(screen.getByText('Investment Summary')).toBeInTheDocument();
    
    // Check summary values
    expect(screen.getByText('Lump Sum')).toBeInTheDocument();
    expect(screen.getByText('$100,000')).toBeInTheDocument(); // Initial investment
    expect(screen.getByText('10 years')).toBeInTheDocument();
    expect(screen.getByText('Moderate')).toBeInTheDocument();
  });

  test('displays projected returns correctly', () => {
    renderComponent();

    expect(screen.getByText('Projected Returns')).toBeInTheDocument();
    
    // Check projected values
    expect(screen.getByText('$220,000')).toBeInTheDocument(); // Final value
    expect(screen.getByText('$120,000')).toBeInTheDocument(); // Total return
    expect(screen.getByText('120.0%')).toBeInTheDocument();   // Return percentage
  });

  test('formats currency values correctly', () => {
    renderComponent();

    // Test various currency formatting
    const currencyElements = screen.getAllByText(/\$[\d,]+/);
    expect(currencyElements.length).toBeGreaterThan(0);
    
    // Specific currency values
    expect(screen.getByText('$100,000')).toBeInTheDocument();
    expect(screen.getByText('$220,000')).toBeInTheDocument();
    expect(screen.getByText('$120,000')).toBeInTheDocument();
  });

  test('formats percentage values correctly', () => {
    renderComponent();

    // Test percentage formatting
    const percentageElements = screen.getAllByText(/\d+\.\d%/);
    expect(percentageElements.length).toBeGreaterThan(0);
  });

  test('renders pie chart for allocation', () => {
    renderComponent();

    expect(screen.getByTestId('pie-chart')).toBeInTheDocument();
    expect(screen.getByTestId('pie')).toBeInTheDocument();
  });

  test('renders line chart for projections', () => {
    renderComponent();

    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    expect(screen.getByTestId('line')).toBeInTheDocument();
    expect(screen.getByTestId('x-axis')).toBeInTheDocument();
    expect(screen.getByTestId('y-axis')).toBeInTheDocument();
  });

  test('handles SIP investment type correctly', () => {
    const sipData = {
      ...mockData,
      summary: {
        ...mockData.summary,
        investment_type: 'sip'
      }
    };

    render(<PortfolioRecommendation data={sipData} />);

    expect(screen.getByText('SIP')).toBeInTheDocument();
  });

  test('displays different risk profiles correctly', () => {
    // Test Low risk
    const lowRiskData = {
      ...mockData,
      summary: {
        ...mockData.summary,
        risk_profile: 'Low'
      }
    };

    const { rerender } = render(<PortfolioRecommendation data={lowRiskData} />);
    expect(screen.getByText(/based on your low risk profile/i)).toBeInTheDocument();

    // Test High risk
    const highRiskData = {
      ...mockData,
      summary: {
        ...mockData.summary,
        risk_profile: 'High'
      }
    };

    rerender(<PortfolioRecommendation data={highRiskData} />);
    expect(screen.getByText(/based on your high risk profile/i)).toBeInTheDocument();
  });

  test('calculates return percentage correctly', () => {
    renderComponent();

    // Return percentage should be (total_return / initial_investment) * 100
    // (120000 / 100000) * 100 = 120%
    expect(screen.getByText('120.0%')).toBeInTheDocument();
  });

  test('handles edge case with zero returns', () => {
    const zeroReturnData = {
      ...mockData,
      summary: {
        ...mockData.summary,
        total_return: 0,
        final_value: 100000
      }
    };

    render(<PortfolioRecommendation data={zeroReturnData} />);

    expect(screen.getByText('$0')).toBeInTheDocument(); // Total return
    expect(screen.getByText('0.0%')).toBeInTheDocument(); // Return percentage
  });

  test('handles large numbers correctly', () => {
    const largeNumberData = {
      ...mockData,
      summary: {
        ...mockData.summary,
        initial_investment: 1000000,
        final_value: 2500000,
        total_return: 1500000
      }
    };

    render(<PortfolioRecommendation data={largeNumberData} />);

    expect(screen.getByText('$1,000,000')).toBeInTheDocument();
    expect(screen.getByText('$2,500,000')).toBeInTheDocument();
    expect(screen.getByText('$1,500,000')).toBeInTheDocument();
  });

  test('displays all required sections', () => {
    renderComponent();

    // Check that all main sections are present
    expect(screen.getByText('Recommended Asset Allocation')).toBeInTheDocument();
    expect(screen.getByText('Portfolio Growth Projection')).toBeInTheDocument();
    expect(screen.getByText('Investment Summary')).toBeInTheDocument();
    expect(screen.getByText('Projected Returns')).toBeInTheDocument();
  });
});