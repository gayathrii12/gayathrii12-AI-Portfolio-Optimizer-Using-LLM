import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import ReturnsLineChart from '../ReturnsLineChart';
import { LineChartDataPoint } from '../../../types';

describe('ReturnsLineChart', () => {
  const mockData: LineChartDataPoint[] = [
    { year: 0, portfolio_value: 100000, formatted_value: '100,000', annual_return: 0, cumulative_return: 0 },
    { year: 1, portfolio_value: 110000, formatted_value: '110,000', annual_return: 10, cumulative_return: 10 },
    { year: 2, portfolio_value: 121000, formatted_value: '121,000', annual_return: 10, cumulative_return: 21 },
    { year: 3, portfolio_value: 133100, formatted_value: '133,100', annual_return: 10, cumulative_return: 33.1 }
  ];

  it('renders loading state', () => {
    render(<ReturnsLineChart data={[]} loading={true} />);
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });

  it('renders error state', () => {
    const errorMessage = 'Failed to load data';
    render(<ReturnsLineChart data={[]} error={errorMessage} />);
    expect(screen.getByText(errorMessage)).toBeInTheDocument();
  });

  it('renders no data message when data is empty', () => {
    render(<ReturnsLineChart data={[]} />);
    expect(screen.getByText('No returns data available')).toBeInTheDocument();
  });

  it('renders chart with title', () => {
    const title = 'Expected Returns Over Time';
    render(<ReturnsLineChart data={mockData} title={title} />);
    expect(screen.getByText(title)).toBeInTheDocument();
  });

  it('renders SVG chart elements', () => {
    const { container } = render(<ReturnsLineChart data={mockData} />);
    
    // Check for SVG container
    const svg = container.querySelector('svg');
    expect(svg).toBeInTheDocument();
    
    // Check for chart line
    const chartLine = container.querySelector('polyline');
    expect(chartLine).toBeInTheDocument();
    
    // Check for data points
    const dataPoints = container.querySelectorAll('circle');
    expect(dataPoints).toHaveLength(mockData.length);
  });

  it('displays statistics correctly', () => {
    render(<ReturnsLineChart data={mockData} />);
    
    // Should show final value
    expect(screen.getByText('133,100')).toBeInTheDocument();
    
    // Should show average annual return
    expect(screen.getByText('10.0%')).toBeInTheDocument();
    
    // Should show number of years
    expect(screen.getByText('3')).toBeInTheDocument();
  });

  it('shows returns data when showReturns is true', () => {
    render(<ReturnsLineChart data={mockData} showReturns={true} />);
    
    // Should show total return instead of final value
    expect(screen.getByText('33.1%')).toBeInTheDocument();
    expect(screen.getByText('Total Return')).toBeInTheDocument();
  });

  it('shows portfolio value when showReturns is false', () => {
    render(<ReturnsLineChart data={mockData} showReturns={false} />);
    
    // Should show final value
    expect(screen.getByText('133,100')).toBeInTheDocument();
    expect(screen.getByText('Final Value')).toBeInTheDocument();
  });

  it('handles custom height prop', () => {
    const customHeight = 500;
    const { container } = render(<ReturnsLineChart data={mockData} height={customHeight} />);
    
    const chartContainer = container.firstChild as HTMLElement;
    expect(chartContainer).toHaveStyle(`height: ${customHeight}px`);
  });

  it('renders grid lines and axis labels', () => {
    const { container } = render(<ReturnsLineChart data={mockData} />);
    
    // Check for grid lines
    const gridLines = container.querySelectorAll('line');
    expect(gridLines.length).toBeGreaterThan(0);
    
    // Check for axis labels
    const axisLabels = container.querySelectorAll('text');
    expect(axisLabels.length).toBeGreaterThan(0);
  });

  it('shows tooltip on data point hover', () => {
    const { container } = render(<ReturnsLineChart data={mockData} />);
    
    const dataPoint = container.querySelector('circle');
    expect(dataPoint).toBeInTheDocument();
    
    if (dataPoint) {
      fireEvent.mouseEnter(dataPoint);
      // Tooltip should be present - check for styled component
      const tooltip = container.querySelector('div[class*="sc-"]');
      expect(tooltip).toBeInTheDocument();
    }
  });

  it('calculates statistics correctly with single data point', () => {
    const singlePointData = [mockData[0]];
    render(<ReturnsLineChart data={singlePointData} />);
    
    // Should handle single point gracefully
    expect(screen.getByText('100,000')).toBeInTheDocument();
    expect(screen.getByText('0')).toBeInTheDocument(); // Years
  });
});