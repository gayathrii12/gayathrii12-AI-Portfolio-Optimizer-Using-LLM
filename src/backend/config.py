"""Configuration settings for Financial Returns Optimizer."""

import os
from pathlib import Path
from typing import Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data file paths
DATA_DIR = PROJECT_ROOT / "data"
HISTORICAL_DATA_FILE = "../../assets/histretSP.xls"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

# Risk profile configurations
RISK_PROFILES: Dict[str, Dict[str, float]] = {
    "Low": {
        "bonds": 65.0,
        "real_estate": 17.5,
        "sp500": 12.5,
        "gold": 5.0,
        "small_cap": 0.0
    },
    "Moderate": {
        "sp500": 45.0,
        "bonds": 30.0,
        "real_estate": 12.5,
        "small_cap": 7.5,
        "gold": 5.0
    },
    "High": {
        "sp500": 55.0,
        "small_cap": 20.0,
        "real_estate": 12.5,
        "bonds": 10.0,
        "gold": 2.5
    }
}

# Asset class mappings
ASSET_CLASSES = [
    "sp500",
    "small_cap", 
    "t_bills",
    "t_bonds",
    "corporate_bonds",
    "real_estate",
    "gold"
]

# Calculation parameters
DEFAULT_RISK_FREE_RATE = 0.03  # 3% risk-free rate for Sharpe ratio
MONTE_CARLO_SIMULATIONS = 1000
REBALANCING_FREQUENCY = "annual"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)