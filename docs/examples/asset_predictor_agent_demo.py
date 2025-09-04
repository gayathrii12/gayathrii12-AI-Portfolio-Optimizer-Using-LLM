"""
Asset Predictor Agent Demo

This script demonstrates how to use the Asset Predictor Agent to generate
expected return predictions for different asset classes based on historical data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from agents.asset_predictor_agent import (
    AssetPredictorAgent,
    PredictionInput,
    MarketRegime,
    create_asset_predictor_agent
)
from utils.data_loader import HistoricalDataLoader
from models.data_models import AssetReturns


def create_sample_data():
    """Create sample historical data for demonstration."""
    sample_data = [
        AssetReturns(year=2014, sp500=0.136, small_cap=0.049, t_bills=0.001, 
                    t_bonds=0.251, corporate_bonds=0.075, real_estate=0.301, gold=-0.017),
        AssetReturns(year=2015, sp500=0.014, small_cap=-0.041, t_bills=0.001, 
                    t_bonds=0.013, corporate_bonds=-0.005, real_estate=0.024, gold=-0.103),
        AssetReturns(year=2016, sp500=0.120, small_cap=0.213, t_bills=0.003, 
                    t_bonds=0.006, corporate_bonds=0.134, real_estate=0.076, gold=0.085),
        AssetReturns(year=2017, sp500=0.217, small_cap=0.143, t_bills=0.010, 
                    t_bonds=0.024, corporate_bonds=0.064, real_estate=0.050, gold=0.133),
        AssetReturns(year=2018, sp500=-0.043, small_cap=-0.111, t_bills=0.018, 
                    t_bonds=0.001, corporate_bonds=-0.021, real_estate=-0.041, gold=-0.018),
        AssetReturns(year=2019, sp500=0.315, small_cap=0.226, t_bills=0.021, 
                    t_bonds=0.069, corporate_bonds=0.137, real_estate=0.226, gold=0.184),
        AssetReturns(year=2020, sp500=0.184, small_cap=0.199, t_bills=0.006, 
                    t_bonds=0.080, corporate_bonds=0.094, real_estate=-0.021, gold=0.249),
        AssetReturns(year=2021, sp500=0.288, small_cap=0.143, t_bills=0.001, 
                    t_bonds=-0.024, corporate_bonds=-0.010, real_estate=0.434, gold=-0.037),
        AssetReturns(year=2022, sp500=-0.181, small_cap=-0.206, t_bills=0.015, 
                    t_bonds=-0.130, corporate_bonds=-0.156, real_estate=-0.256, gold=0.001),
        AssetReturns(year=2023, sp500=0.264, small_cap=0.169, t_bills=0.046, 
                    t_bonds=-0.031, corporate_bonds=0.084, real_estate=0.111, gold=0.134)
    ]
    return sample_data


def demo_basic_prediction():
    """Demonstrate basic asset return prediction."""
    print("=== Asset Predictor Agent Demo ===\n")
    
    # Create agent
    agent = create_asset_predictor_agent()
    
    # Create sample historical data
    historical_data = create_sample_data()
    print(f"Using {len(historical_data)} years of historical data ({historical_data[0].year}-{historical_data[-1].year})")
    
    # Create prediction input
    prediction_input = PredictionInput(
        historical_data=historical_data,
        lookback_years=10,
        volatility_adjustment=True,
        market_regime_analysis=True,
        risk_free_rate=0.025
    )
    
    print(f"Prediction parameters:")
    print(f"  - Lookback period: {prediction_input.lookback_years} years")
    print(f"  - Volatility adjustment: {prediction_input.volatility_adjustment}")
    print(f"  - Market regime analysis: {prediction_input.market_regime_analysis}")
    print(f"  - Risk-free rate: {prediction_input.risk_free_rate:.1%}")
    print()
    
    # Generate predictions
    print("Generating asset return predictions...")
    result = agent.predict_returns(prediction_input)
    
    if result.success:
        print("✓ Predictions generated successfully!\n")
        
        # Display summary
        summary = agent.get_prediction_summary(result)
        print(summary)
        
        # Display detailed analysis
        print("\n=== DETAILED ANALYSIS ===")
        print(f"Market Regime Detected: {result.market_regime.value.replace('_', ' ').title()}")
        print(f"Analysis Period: {result.analysis_period['start_year']}-{result.analysis_period['end_year']}")
        print()
        
        print("Asset-by-Asset Breakdown:")
        print("-" * 80)
        
        for asset_code, prediction in result.predictions.items():
            print(f"\n{prediction.asset_name}:")
            print(f"  Historical Mean:           {prediction.historical_mean:8.1%}")
            print(f"  Volatility Adjusted:       {prediction.volatility_adjusted_return:8.1%}")
            print(f"  Regime Adjusted:           {prediction.regime_adjusted_return:8.1%}")
            print(f"  Final Expected Return:     {prediction.expected_return:8.1%}")
            print(f"  Volatility (Std Dev):      {prediction.volatility:8.1%}")
            print(f"  Sharpe Ratio:              {prediction.sharpe_ratio:8.2f}")
            
            ci_lower, ci_upper = prediction.confidence_interval
            print(f"  95% Confidence Interval:   [{ci_lower:6.1%}, {ci_upper:6.1%}]")
        
    else:
        print(f"✗ Prediction failed: {result.error_message}")


def demo_different_scenarios():
    """Demonstrate predictions under different scenarios."""
    print("\n\n=== SCENARIO ANALYSIS ===\n")
    
    agent = create_asset_predictor_agent()
    historical_data = create_sample_data()
    
    scenarios = [
        {
            "name": "Conservative (No Adjustments)",
            "params": {
                "lookback_years": 10,
                "volatility_adjustment": False,
                "market_regime_analysis": False,
                "risk_free_rate": 0.02
            }
        },
        {
            "name": "Moderate (Volatility Adjustment Only)",
            "params": {
                "lookback_years": 10,
                "volatility_adjustment": True,
                "market_regime_analysis": False,
                "risk_free_rate": 0.02
            }
        },
        {
            "name": "Aggressive (All Adjustments)",
            "params": {
                "lookback_years": 10,
                "volatility_adjustment": True,
                "market_regime_analysis": True,
                "risk_free_rate": 0.02
            }
        },
        {
            "name": "Short-term Focus (5 years)",
            "params": {
                "lookback_years": 5,
                "volatility_adjustment": True,
                "market_regime_analysis": True,
                "risk_free_rate": 0.03
            }
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"Running scenario: {scenario['name']}")
        
        prediction_input = PredictionInput(
            historical_data=historical_data,
            **scenario['params']
        )
        
        result = agent.predict_returns(prediction_input)
        results[scenario['name']] = result
        
        if result.success:
            print(f"  ✓ Market regime: {result.market_regime.value}")
        else:
            print(f"  ✗ Failed: {result.error_message}")
    
    # Compare results
    print("\n=== SCENARIO COMPARISON ===")
    print(f"{'Scenario':<30} {'S&P 500':<10} {'Small Cap':<10} {'T-Bills':<10} {'Gold':<10}")
    print("-" * 70)
    
    for scenario_name, result in results.items():
        if result.success:
            sp500_return = result.predictions['sp500'].expected_return
            small_cap_return = result.predictions['small_cap'].expected_return
            tbills_return = result.predictions['t_bills'].expected_return
            gold_return = result.predictions['gold'].expected_return
            
            print(f"{scenario_name:<30} {sp500_return:<10.1%} {small_cap_return:<10.1%} "
                  f"{tbills_return:<10.1%} {gold_return:<10.1%}")


def demo_with_real_data():
    """Demonstrate using real historical data if available."""
    print("\n\n=== REAL DATA DEMO ===\n")
    
    # Try to load real historical data
    try:
        loader = HistoricalDataLoader("histretSP.xls")
        print("Loading real historical data from histretSP.xls...")
        
        # Clean and preprocess the data
        cleaned_data = loader.clean_and_preprocess()
        asset_returns = loader.to_asset_returns_list()
        
        print(f"✓ Loaded {len(asset_returns)} years of real data ({asset_returns[0].year}-{asset_returns[-1].year})")
        
        # Create agent and run prediction
        agent = create_asset_predictor_agent()
        
        prediction_input = PredictionInput(
            historical_data=asset_returns,
            lookback_years=20,  # Use 20 years of real data
            volatility_adjustment=True,
            market_regime_analysis=True,
            risk_free_rate=0.025
        )
        
        print("Generating predictions with real historical data...")
        result = agent.predict_returns(prediction_input)
        
        if result.success:
            print("✓ Real data predictions generated successfully!\n")
            
            # Show key results
            print("Key Predictions (Real Data):")
            print("-" * 40)
            
            key_assets = ['sp500', 'small_cap', 't_bills', 'gold']
            for asset in key_assets:
                if asset in result.predictions:
                    pred = result.predictions[asset]
                    print(f"{pred.asset_name:<15}: {pred.expected_return:6.1%} "
                          f"(volatility: {pred.volatility:5.1%})")
            
            print(f"\nMarket Regime: {result.market_regime.value.replace('_', ' ').title()}")
            
        else:
            print(f"✗ Real data prediction failed: {result.error_message}")
            
    except Exception as e:
        print(f"Could not load real data: {e}")
        print("Using sample data instead...")
        demo_basic_prediction()


def demo_risk_analysis():
    """Demonstrate risk analysis features."""
    print("\n\n=== RISK ANALYSIS DEMO ===\n")
    
    agent = create_asset_predictor_agent()
    historical_data = create_sample_data()
    
    # Run prediction
    prediction_input = PredictionInput(
        historical_data=historical_data,
        lookback_years=10,
        volatility_adjustment=True,
        market_regime_analysis=True,
        risk_free_rate=0.025
    )
    
    result = agent.predict_returns(prediction_input)
    
    if result.success:
        print("Risk-Return Analysis:")
        print("-" * 60)
        print(f"{'Asset':<20} {'Return':<8} {'Risk':<8} {'Sharpe':<8} {'Risk/Return':<12}")
        print("-" * 60)
        
        for asset_code, prediction in result.predictions.items():
            risk_return_ratio = prediction.volatility / prediction.expected_return if prediction.expected_return > 0 else float('inf')
            
            print(f"{prediction.asset_name:<20} "
                  f"{prediction.expected_return:<8.1%} "
                  f"{prediction.volatility:<8.1%} "
                  f"{prediction.sharpe_ratio:<8.2f} "
                  f"{risk_return_ratio:<12.2f}")
        
        # Find best risk-adjusted returns
        print("\nRisk-Adjusted Rankings:")
        print("-" * 30)
        
        sorted_by_sharpe = sorted(
            result.predictions.items(),
            key=lambda x: x[1].sharpe_ratio,
            reverse=True
        )
        
        for i, (asset_code, prediction) in enumerate(sorted_by_sharpe, 1):
            print(f"{i}. {prediction.asset_name}: {prediction.sharpe_ratio:.2f}")


if __name__ == "__main__":
    try:
        demo_basic_prediction()
        demo_different_scenarios()
        demo_with_real_data()
        demo_risk_analysis()
        
        print("\n=== Demo Complete ===")
        print("The Asset Predictor Agent successfully generated return predictions!")
        print("You can now use this agent in your portfolio optimization pipeline.")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()