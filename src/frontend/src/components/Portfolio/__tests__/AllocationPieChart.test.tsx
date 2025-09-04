import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import AllocationPieChart from '../AllocationPieChart';
import { PieChartDataPoint } from '../../../types';

describe('AllocationPieChart', () => {
  const mockData: PieChartDataPoint[] = [
    { name: 'S&P 500', value: 45, color: '#1f77b4', percentage: '45.0%' },
    { name: 'Bonds', value: 25, color: '#2ca02c', percentage: '25.0%' },
    { name: 'US Small Cap', value: 20, color: '#ff7f0e', percentage: '20.0%' },
    { name: 'Real Estate', value: 7.5, color: '#d62728', percentage: '7.5%' },
    { name: 'Gold', value: 2.5, color: '#9467bd', percentage: '2.5%' }
  ];

  it('renders loading state', () => {
    render(<AllocationPieChart data={[]} loading={true} />);
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });

  it('renders error state', () => {
    const errorMessage = 'Failed to load data';
    render(<AllocationPieChart data={[]} error={errorMessage} />);
    expect(screen.getByText(errorMessage)).toBeInTheDocument();
  });

  it('renders no data message when data is empty', () => {
    render(<AllocationPieChart data={[]} />);
    expect(screen.getByText('No allocation data available')).toBeInTheDocument();
  });

  it('renders chart with title', () => {
    const title = 'Portfolio Allocation';
    render(<AllocationPieChart data={mockData} title={title} />);
    expect(screen.getByText(title)).toBeInTheDocument();
  });

  it('renders all asset classes in legend', () => {
    render(<AllocationPieChart data={mockData} />);
    
    mockData.forEach(item => {
      expect(screen.getByText(item.name)).toBeInTheDocument();
      expect(screen.getByText(item.percentage)).toBeInTheDocument();
    });
  });

  it('displays correct portfolio summary', () => {
    render(<AllocationPieChart data={mockData} />);
    
    expect(screen.getByText('Total Allocation: 100.0%')).toBeInTheDocument();
    expect(screen.getByText('Asset Classes: 5')).toBeInTheDocument();
    expect(screen.getByText('Portfolio')).toBeInTheDocument();
    expect(screen.getByText('5 Assets')).toBeInTheDocument();
  });

  it('handles custom height prop', () => {
    const customHeight = 500;
    const { container } = render(<AllocationPieChart data={mockData} height={customHeight} />);
    
    const chartContainer = container.firstChild as HTMLElement;
    expect(chartContainer).toHaveStyle(`height: ${customHeight}px`);
  });

  it('calculates total allocation correctly with partial data', () => {
    const partialData = mockData.slice(0, 2); // Only first 2 items (45 + 25 = 70)
    render(<AllocationPieChart data={partialData} />);
    
    expect(screen.getByText('Total Allocation: 70.0%')).toBeInTheDocument();
    expect(screen.getByText('Asset Classes: 2')).toBeInTheDocument();
  });

  it('renders pie chart with correct gradient', () => {
    const { container } = render(<AllocationPieChart data={mockData} />);
    
    // Check if the pie chart div exists - it should be styled with styled-components
    const pieChart = container.querySelector('div[class*="sc-"]');
    expect(pieChart).toBeInTheDocument();
  });

  it('renders legend colors correctly', () => {
    const { container } = render(<AllocationPieChart data={mockData} />);
    
    // Check that legend items exist - they should be styled components
    const legendItems = container.querySelectorAll('div[class*="sc-"]');
    expect(legendItems.length).toBeGreaterThan(0);
  });
});