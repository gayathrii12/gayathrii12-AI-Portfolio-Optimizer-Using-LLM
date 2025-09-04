import axios, { AxiosResponse } from 'axios';
import { UserInputData } from '../components/UserInput';
import { PortfolioRecommendationData, AssetAllocation, ProjectionData } from '../components/Portfolio';
import { ApiResponse } from '../types';

// Base API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds for portfolio generation
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request/Response types for the backend
interface PortfolioGenerationRequest {
  investment_amount: number;
  investment_type: 'lumpsum' | 'sip';
  tenure_years: number;
  risk_profile: 'Low' | 'Moderate' | 'High';
  return_expectation: number;
  monthly_amount?: number;
}

interface PortfolioGenerationResponse {
  allocation: AssetAllocation;
  projections: ProjectionData[];
  risk_metrics: {
    expected_return: number;
    volatility: number;
    sharpe_ratio: number;
  };
  summary: {
    initial_investment: number;
    final_value: number;
    total_return: number;
    investment_type: string;
    tenure_years: number;
    risk_profile: string;
  };
}

class InvestmentPlannerService {
  /**
   * Generate portfolio recommendation based on user input
   */
  async generatePortfolioRecommendation(userInput: UserInputData): Promise<PortfolioRecommendationData> {
    try {
      // Prepare request data
      const requestData: PortfolioGenerationRequest = {
        investment_amount: userInput.investment_amount,
        investment_type: userInput.investment_type,
        tenure_years: userInput.tenure_years,
        risk_profile: userInput.risk_profile,
        return_expectation: userInput.return_expectation,
        ...(userInput.monthly_amount && { monthly_amount: userInput.monthly_amount })
      };

      console.log('Generating portfolio recommendation with data:', requestData);

      // Call the working portfolio generation endpoint
      const response: AxiosResponse<ApiResponse<PortfolioGenerationResponse>> = await apiClient.post(
        '/api/portfolio/generate',
        requestData
      );

      console.log('Portfolio recommendation response:', response.data);

      const responseData = response.data.data;

      // Transform allocation data to match frontend expectations
      // The backend might return empty allocation, so we'll get it from the separate endpoint
      let allocation: AssetAllocation;
      
      if (responseData.allocation && Object.keys(responseData.allocation).length > 0) {
        // Use allocation from the response if available (but it's usually empty)
        allocation = {
          sp500: (responseData.allocation as any).sp500 || 0,
          small_cap: (responseData.allocation as any).small_cap || 0,
          bonds: (responseData.allocation as any).bonds || 0,
          real_estate: (responseData.allocation as any).real_estate || 0,
          gold: (responseData.allocation as any).gold || 0
        };
      } else {
        // Get allocation from the separate endpoint (this is the working approach)
        try {
          const allocationResponse: AxiosResponse<ApiResponse<any>> = await apiClient.post(
            '/api/portfolio/allocate',
            requestData
          );
          const allocationData = allocationResponse.data.data;
          
          // Aggregate the detailed bond allocations into a single bonds value
          const bondsTotal = (allocationData.allocation.t_bills || 0) + 
                           (allocationData.allocation.t_bonds || 0) + 
                           (allocationData.allocation.corporate_bonds || 0);
          
          allocation = {
            sp500: allocationData.allocation.sp500 || 0,
            small_cap: allocationData.allocation.small_cap || 0,
            bonds: bondsTotal,
            real_estate: allocationData.allocation.real_estate || 0,
            gold: allocationData.allocation.gold || 0
          };
        } catch (allocError) {
          console.warn('Could not get detailed allocation, using mock data');
          allocation = this.getMockAllocation(userInput.risk_profile);
        }
      }

      // Validate that all required data is present
      if (!responseData.projections || !responseData.risk_metrics || !responseData.summary) {
        console.warn('Incomplete data from API, using mock data');
        return this.getMockPortfolioRecommendation(userInput);
      }

      // Ensure summary has all required fields
      if (!responseData.summary.risk_profile) {
        responseData.summary.risk_profile = userInput.risk_profile;
      }
      
      // Ensure risk_metrics has all required fields
      if (!responseData.risk_metrics.expected_return) {
        responseData.risk_metrics.expected_return = userInput.return_expectation;
      }
      if (!responseData.risk_metrics.volatility) {
        responseData.risk_metrics.volatility = this.getVolatilityByRisk(userInput.risk_profile);
      }
      if (!responseData.risk_metrics.sharpe_ratio) {
        responseData.risk_metrics.sharpe_ratio = this.getSharpeRatioByRisk(userInput.risk_profile);
      }

      // Ensure projections is an array with at least one entry
      if (!Array.isArray(responseData.projections) || responseData.projections.length === 0) {
        console.warn('Invalid projections data, using mock projections');
        responseData.projections = this.getMockProjections(userInput);
      }

      const result: PortfolioRecommendationData = {
        allocation,
        projections: responseData.projections,
        risk_metrics: responseData.risk_metrics,
        summary: responseData.summary
      };

      return result;
    } catch (error) {
      console.error('Error generating portfolio recommendation:', error);
      
      // If API is not available, return mock data for development
      if (axios.isAxiosError(error) && (error.code === 'ECONNREFUSED' || error.response?.status === 404)) {
        console.warn('Portfolio API not available, returning mock recommendation');
        return this.getMockPortfolioRecommendation(userInput);
      }
      
      throw new Error(
        axios.isAxiosError(error) && error.response?.data?.message
          ? error.response.data.message
          : 'Failed to generate portfolio recommendation. Please try again.'
      );
    }
  }

  /**
   * Calculate investment projections based on user input
   */
  async calculateInvestmentProjections(userInput: UserInputData): Promise<ProjectionData[]> {
    try {
      // Use the working portfolio generation endpoint and extract projections
      const response: AxiosResponse<ApiResponse<PortfolioGenerationResponse>> = await apiClient.post(
        '/api/portfolio/generate',
        userInput
      );

      return response.data.data.projections;
    } catch (error) {
      console.error('Error calculating investment projections:', error);
      
      // Return mock projections if API is not available
      if (axios.isAxiosError(error) && (error.code === 'ECONNREFUSED' || error.response?.status === 404)) {
        console.warn('Investment calculation API not available, returning mock projections');
        return this.getMockProjections(userInput);
      }
      
      throw new Error('Failed to calculate investment projections');
    }
  }

  /**
   * Get rebalancing simulation data
   */
  async simulateRebalancing(userInput: UserInputData): Promise<any> {
    try {
      const response: AxiosResponse<ApiResponse<any>> = await apiClient.post(
        '/api/rebalancing/simulate',
        userInput
      );

      return response.data.data;
    } catch (error) {
      console.error('Error simulating rebalancing:', error);
      throw new Error('Failed to simulate rebalancing');
    }
  }

  /**
   * Get ML model predictions for asset returns
   */
  async getModelPredictions(horizon: number = 1): Promise<any> {
    try {
      const requestData = {
        horizon: horizon,
        include_confidence: true
      };

      const response: AxiosResponse<ApiResponse<any>> = await apiClient.post(
        '/api/models/predict',
        requestData
      );

      return response.data.data;
    } catch (error) {
      console.error('Error getting model predictions:', error);
      throw new Error('Failed to get model predictions');
    }
  }

  /**
   * Mock portfolio recommendation for development/fallback
   */
  private getMockPortfolioRecommendation(userInput: UserInputData): PortfolioRecommendationData {
    // Generate allocation based on risk profile
    const allocation = this.getMockAllocation(userInput.risk_profile);
    
    // Generate projections based on user input
    const projections = this.getMockProjections(userInput);
    
    // Calculate summary metrics
    const finalValue = projections[projections.length - 1].portfolio_value;
    const totalReturn = finalValue - userInput.investment_amount;
    
    return {
      allocation,
      projections,
      risk_metrics: {
        expected_return: userInput.return_expectation,
        volatility: this.getVolatilityByRisk(userInput.risk_profile),
        sharpe_ratio: this.getSharpeRatioByRisk(userInput.risk_profile)
      },
      summary: {
        initial_investment: userInput.investment_amount,
        final_value: finalValue,
        total_return: totalReturn,
        investment_type: userInput.investment_type,
        tenure_years: userInput.tenure_years,
        risk_profile: userInput.risk_profile
      }
    };
  }

  /**
   * Generate mock allocation based on risk profile
   */
  private getMockAllocation(riskProfile: string): AssetAllocation {
    switch (riskProfile) {
      case 'Low':
        return {
          sp500: 25.0,
          small_cap: 10.0,
          bonds: 50.0,
          real_estate: 10.0,
          gold: 5.0
        };
      case 'High':
        return {
          sp500: 50.0,
          small_cap: 25.0,
          bonds: 15.0,
          real_estate: 7.5,
          gold: 2.5
        };
      default: // Moderate
        return {
          sp500: 40.0,
          small_cap: 20.0,
          bonds: 25.0,
          real_estate: 10.0,
          gold: 5.0
        };
    }
  }

  /**
   * Generate mock projections based on user input
   */
  private getMockProjections(userInput: UserInputData): ProjectionData[] {
    const projections: ProjectionData[] = [];
    let currentValue = userInput.investment_amount;
    const annualReturn = userInput.return_expectation / 100;
    
    // Add some volatility to make it more realistic
    const volatility = this.getVolatilityByRisk(userInput.risk_profile) / 100;
    
    for (let year = 0; year <= userInput.tenure_years; year++) {
      if (year === 0) {
        projections.push({
          year,
          portfolio_value: currentValue,
          annual_return: 0,
          cumulative_return: 0
        });
      } else {
        // Add some random variation based on volatility
        const randomFactor = 1 + (Math.random() - 0.5) * volatility * 0.5;
        const yearlyReturn = annualReturn * randomFactor;
        
        // Handle SIP monthly contributions
        if (userInput.investment_type === 'sip' && userInput.monthly_amount) {
          // Add monthly contributions throughout the year
          const monthlyContribution = userInput.monthly_amount * 12;
          currentValue = (currentValue + monthlyContribution) * (1 + yearlyReturn);
        } else {
          currentValue = currentValue * (1 + yearlyReturn);
        }
        
        const cumulativeReturn = ((currentValue / userInput.investment_amount) - 1) * 100;
        
        projections.push({
          year,
          portfolio_value: Math.round(currentValue),
          annual_return: Math.round(yearlyReturn * 100 * 100) / 100,
          cumulative_return: Math.round(cumulativeReturn * 100) / 100
        });
      }
    }
    
    return projections;
  }

  /**
   * Get volatility based on risk profile
   */
  private getVolatilityByRisk(riskProfile: string): number {
    switch (riskProfile) {
      case 'Low':
        return 8.0; // 8% volatility for conservative
      case 'High':
        return 18.0; // 18% volatility for aggressive
      default: // Moderate
        return 12.0; // 12% volatility for balanced
    }
  }

  /**
   * Get Sharpe ratio based on risk profile
   */
  private getSharpeRatioByRisk(riskProfile: string): number {
    switch (riskProfile) {
      case 'Low':
        return 0.75; // Lower but more stable returns
      case 'High':
        return 0.85; // Higher potential returns
      default: // Moderate
        return 0.80; // Balanced risk-return
    }
  }
}

// Export singleton instance
export const investmentPlannerService = new InvestmentPlannerService();
export default investmentPlannerService;