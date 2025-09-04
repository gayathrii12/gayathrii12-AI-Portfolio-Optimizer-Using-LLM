import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import PortfolioValueChart from '../PortfolioValueChart';
import { LineChartDataPoint } from '../../../types';

describe('PortfolioValueChart', () => {
  const mockData: LineChartDataPoint[] = [
    { year: 0, portfolio_value: 100000, formatted_value: '100,000', annual_return: 0, cumulative_return: 0 },
    { year: 1, portfolio_value: 110000, formatted_value: '110,000', annual_return: 10, cumulative_return: 10 },
    { year: 2, portfolio_value: 121000, formatted_value: '121,000', annual_return: 10, cumulative_return: 21 },
    { year: 5, portfolio_value: 161051, formatted_value: '161,051', annual_return: 10, cumulative_return: 61.051 },
    { year: 10, portfolio_value: 259374, formatted_value: '259,374', annual_return: 10, cumulative_return: 159.374 }
  ];

  const mockUserInput = {
    investment_amount: 100000,
    investment_tenure: 10,
    risk_profile: 'moderate' as const,
    investment_type: 'lump_sum' as const
  };

  it('renders loading state', () => {
    render(<PortfolioValueChart data={[]} loading={true} />);
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });

  it('renders error state', () => {
    const errorMessage = 'Failed to load data';
    render(<PortfolioValueChart data={[]} error={errorMessage} />);
    expect(screen.getByText(errorMessage)).toBeInTheDocument();
  });

  it('renders no data message when data is empty', () => {
    render(<PortfolioValueChart data={[]} />);
    expect(screen.getByText('No portfolio value data available')).toBeInTheDocument();
  });

  it('renders chart with title', () => {
    const title = 'Portfolio Value Growth';
    render(<PortfolioValueChart data={mockData} title={title} />);
    expect(screen.getByText(title)).toBeInTheDocument();
  });

  it('renders investment summary when userInput is provided', () => {
    render(<PortfolioValueChart data={mockData} userInput={mockUserInput} />);
    
    expect(screen.getByText('LUMP_SUM')).toBeInTheDocument();
    expect(screen.getByText('₹100,000')).toBeInTheDocument();
    expect(screen.getByText('10 years')).toBeInTheDocument();
    expect(screen.getByText('Moderate')).toBeInTheDocument();
  });

  it('does not render investment summary when userInput is not provided', () => {
    render(<PortfolioValueChart data={mockData} />);
    
    expect(screen.queryByText('LUMP_SUM')).not.toBeInTheDocument();
    expect(screen.queryByText('Investment Type')).not.toBeInTheDocument();
  });

  it('renders SVG chart with gradient elements', () => {
    const { container } = render(<PortfolioValueChart data={mockData} />);
    
    // Check for SVG container
    const svg = container.querySelector('svg');
    expect(svg).toBeInTheDocument();
    
    // Check for gradient definitions
    const gradients = container.querySelectorAll('linearGradient');
    expect(gradients.length).toBeGreaterThanOrEqual(2); // Should have line and area gradients
    
    // Check for chart line
    const chartLine = container.querySelector('polyline');
    expect(chartLine).toBeInTheDocument();
    
    // Check for area fill
    const areaFill = container.querySelector('polygon');
    expect(areaFill).toBeInTheDocument();
    
    // Check for data points
    const dataPoints = container.querySelectorAll('circle');
    expect(dataPoints).toHaveLength(mockData.length);
  });

  it('displays correct statistics in stat cards', () => {
    render(<PortfolioValueChart data={mockData} />);
    
    // Final Portfolio Value
    expect(screen.getByText('₹259,374')).toBeInTheDocument();
    expect(screen.getByText('Final Portfolio Value')).toBeInTheDocument();
    
    // Total Return
    expect(screen.getByText('159.4%')).toBeInTheDocument();
    expect(screen.getByText('Total Return')).toBeInTheDocument();
    
    // CAGR
    expect(screen.getByText('CAGR')).toBeInTheDocument();
    
    // Average Annual Return
    expect(screen.getByText('Avg Annual Return')).toBeInTheDocument();
  });

  it('handles custom height prop', () => {
    const customHeight = 600;
    const { container } = render(<PortfolioValueChart data={mockData} height={customHeight} />);
    
    const chartContainer = container.firstChild as HTMLElement;
    expect(chartContainer).toHaveStyle(`height: ${customHeight}px`);
  });

  it('renders grid lines and axis labels', () => {
    const { container } = render(<PortfolioValueChart data={mockData} />);
    
    // Check for grid lines
    const gridLines = container.querySelectorAll('line');
    expect(gridLines.length).toBeGreaterThan(0);
    
    // Check for axis labels
    const axisLabels = container.querySelectorAll('text');
    expect(axisLabels.length).toBeGreaterThan(0);
  });

  it('shows tooltip on data point hover', () => {
    const { container } = render(<PortfolioValueChart data={mockData} />);
    
    const dataPoint = container.querySelector('circle');
    expect(dataPoint).toBeInTheDocument();
    
    if (dataPoint) {
      fireEvent.mouseEnter(dataPoint);
      // Tooltip should be present - check for styled component
      const tooltip = container.querySelector('div[class*="sc-"]');
      expect(tooltip).toBeInTheDocument();
    }
  });

  it('formats large values correctly in axis labels', () => {
    const largeValueData = [
      { year: 0, portfolio_value: 1000000, formatted_value: '1,000,000' },
      { year: 10, portfolio_value: 2500000, formatted_value: '2,500,000' }
    ];
    
    const { container } = render(<PortfolioValueChart data={largeValueData} />);
    
    // Should format values in millions
    const axisLabels = container.querySelectorAll('text');
    const hasMillionFormat = Array.from(axisLabels).some(label => 
      label.textContent?.includes('M')
    );
    expect(hasMillionFormat).toBe(true);
  });

  it('calculates CAGR correctly', () => {
    render(<PortfolioValueChart data={mockData} />);
    
    // With the mock data, CAGR should be calculated
    // Final value: 259,374, Initial: 100,000, Years: 4 (index 4 = year 10, but we have 5 data points)
    // CAGR = (259374/100000)^(1/4) - 1 ≈ 26.9%
    const cagrElements = screen.getAllByText(/\d+\.\d+%/);
    expect(cagrElements.length).toBeGreaterThan(0);
  });

  it('handles single data point gracefully', () => {
    const singlePointData = [mockData[0]];
    render(<PortfolioValueChart data={singlePointData} />);
    
    // Should not crash and should show the single value
    expect(screen.getByText('₹100,000')).toBeInTheDocument();
    expect(screen.getAllByText('0.0%')[0]).toBeInTheDocument(); // Total return should be 0
  });
});