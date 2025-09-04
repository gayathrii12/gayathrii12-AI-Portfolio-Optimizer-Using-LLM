"""
ML Models for Asset Return Prediction

This module implements machine learning models for predicting returns
of various asset classes including S&P 500, Small Cap, T-Bills, T-Bonds,
Corporate Bonds, Real Estate, and Gold.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AssetReturnModels:
    """
    ML Models for predicting asset returns using historical data.
    
    Supports training and prediction for:
    - S&P 500 (Large Cap Stocks)
    - Small Cap Stocks
    - 3-month T-Bills
    - 10-year T-Bonds
    - Corporate Bonds (Baa)
    - Real Estate
    - Gold
    """
    
    def __init__(self, data_file: str = "../../assets/histretSP.xls"):
        """
        Initialize the AssetReturnModels class.
        
        Args:
            data_file: Path to the historical data Excel file
        """
        self.data_file = data_file
        self.models: Dict[str, any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.historical_data: Optional[pd.DataFrame] = None
        self.asset_columns = {
            'sp500': 'S&P 500 (includes dividends)',
            'small_cap': 'US Small cap (bottom decile)',
            't_bills': '3-month T.Bill',
            't_bonds': 'US T. Bond (10-year)',
            'corporate_bonds': ' Baa Corporate Bond',
            'real_estate': 'Real Estate',
            'gold': 'Gold*'
        }
        
    def load_historical_data(self) -> pd.DataFrame:
        """
        Load and clean historical data from Excel file.
        
        Returns:
            DataFrame with cleaned historical returns data
        """
        try:
            # Read the Excel file
            df = pd.read_excel(self.data_file, sheet_name='Returns by year')
            
            # Find the row with 'Year' which should be our header
            year_row = None
            for i, row in df.iterrows():
                if 'Year' in str(row.iloc[0]):
                    year_row = i
                    break
            
            if year_row is None:
                raise ValueError("Could not find 'Year' header in the data")
            
            # Get the header row and data
            headers = df.iloc[year_row].values
            data_df = df.iloc[year_row+1:].copy()
            data_df.columns = headers
            
            # Clean the data
            data_df = data_df.reset_index(drop=True)
            
            # Convert Year to numeric and filter out non-numeric years
            data_df['Year'] = pd.to_numeric(data_df['Year'], errors='coerce')
            data_df = data_df.dropna(subset=['Year'])
            data_df = data_df[data_df['Year'] >= 1928]  # Start from 1928
            
            # Convert return columns to numeric
            for asset_key, col_name in self.asset_columns.items():
                if col_name in data_df.columns:
                    data_df[col_name] = pd.to_numeric(data_df[col_name], errors='coerce')
            
            # Remove rows with too many missing values
            data_df = data_df.dropna(subset=list(self.asset_columns.values()), thresh=5)
            
            self.historical_data = data_df
            logger.info(f"Loaded historical data: {len(data_df)} years from {data_df['Year'].min()} to {data_df['Year'].max()}")
            
            return data_df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise
    
    def create_features(self, data: pd.DataFrame, target_col: str, lookback_years: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create features for ML model training.
        
        Args:
            data: Historical data DataFrame
            target_col: Target column name for prediction
            lookback_years: Number of years to look back for features
            
        Returns:
            Tuple of (features, targets)
        """
        features = []
        targets = []
        
        # Sort by year
        data_sorted = data.sort_values('Year').reset_index(drop=True)
        
        for i in range(lookback_years, len(data_sorted)):
            # Create features from previous years
            feature_row = []
            
            # Add lagged returns for the target asset
            for lag in range(1, lookback_years + 1):
                if i - lag >= 0:
                    feature_row.append(data_sorted.iloc[i - lag][target_col])
                else:
                    feature_row.append(0)
            
            # Add other asset returns from previous year as features
            prev_year_idx = i - 1
            for asset_key, col_name in self.asset_columns.items():
                if col_name != target_col and col_name in data_sorted.columns:
                    if prev_year_idx >= 0:
                        feature_row.append(data_sorted.iloc[prev_year_idx][col_name])
                    else:
                        feature_row.append(0)
            
            # Add year as a feature (normalized)
            year_normalized = (data_sorted.iloc[i]['Year'] - 1928) / 100
            feature_row.append(year_normalized)
            
            features.append(feature_row)
            targets.append(data_sorted.iloc[i][target_col])
        
        return np.array(features), np.array(targets)
    
    def train_model(self, asset_class: str, model_type: str = 'random_forest') -> Dict[str, float]:
        """
        Train ML model for a specific asset class.
        
        Args:
            asset_class: Asset class key (e.g., 'sp500', 'small_cap')
            model_type: Type of model ('random_forest' or 'linear')
            
        Returns:
            Dictionary with training metrics
        """
        if self.historical_data is None:
            self.load_historical_data()
        
        if asset_class not in self.asset_columns:
            raise ValueError(f"Unknown asset class: {asset_class}")
        
        target_col = self.asset_columns[asset_class]
        
        if target_col not in self.historical_data.columns:
            raise ValueError(f"Column {target_col} not found in data")
        
        # Create features and targets
        X, y = self.create_features(self.historical_data, target_col)
        
        if len(X) == 0:
            raise ValueError(f"No valid data for training {asset_class} model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                min_samples_split=5
            )
        else:
            model = LinearRegression()
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        # Store model and scaler
        self.models[asset_class] = model
        self.scalers[asset_class] = scaler
        
        metrics = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        logger.info(f"Trained {asset_class} model - Test R²: {test_r2:.3f}, Test RMSE: {test_rmse:.3f}")
        
        return metrics
    
    def train_all_models(self, model_type: str = 'random_forest') -> Dict[str, Dict[str, float]]:
        """
        Train ML models for all asset classes.
        
        Args:
            model_type: Type of model to train ('random_forest' or 'linear')
            
        Returns:
            Dictionary with metrics for each asset class
        """
        all_metrics = {}
        
        for asset_class in self.asset_columns.keys():
            try:
                metrics = self.train_model(asset_class, model_type)
                all_metrics[asset_class] = metrics
            except Exception as e:
                logger.error(f"Failed to train model for {asset_class}: {e}")
                all_metrics[asset_class] = {'error': str(e)}
        
        return all_metrics
    
    def predict_returns(self, asset_class: str, horizon: int = 1) -> float:
        """
        Predict returns for a specific asset class.
        
        Args:
            asset_class: Asset class key (e.g., 'sp500', 'small_cap')
            horizon: Prediction horizon in years (currently supports 1 year)
            
        Returns:
            Predicted annual return as a decimal (e.g., 0.08 for 8%)
        """
        if asset_class not in self.models:
            raise ValueError(f"Model for {asset_class} not trained. Call train_model() first.")
        
        if self.historical_data is None:
            self.load_historical_data()
        
        model = self.models[asset_class]
        scaler = self.scalers[asset_class]
        target_col = self.asset_columns[asset_class]
        
        # Get recent data for features
        recent_data = self.historical_data.tail(10).copy()  # Last 10 years
        
        # Create features similar to training
        lookback_years = 3
        feature_row = []
        
        # Add lagged returns for the target asset
        for lag in range(1, lookback_years + 1):
            if len(recent_data) >= lag:
                feature_row.append(recent_data.iloc[-lag][target_col])
            else:
                feature_row.append(0)
        
        # Add other asset returns from previous year
        for asset_key, col_name in self.asset_columns.items():
            if col_name != target_col and col_name in recent_data.columns:
                if len(recent_data) >= 1:
                    feature_row.append(recent_data.iloc[-1][col_name])
                else:
                    feature_row.append(0)
        
        # Add normalized year
        current_year = recent_data.iloc[-1]['Year']
        year_normalized = (current_year + horizon - 1928) / 100
        feature_row.append(year_normalized)
        
        # Scale features and predict
        features = np.array(feature_row).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        return prediction
    
    def get_all_predictions(self, horizon: int = 1) -> Dict[str, float]:
        """
        Get return predictions for all trained asset classes.
        
        Args:
            horizon: Prediction horizon in years
            
        Returns:
            Dictionary with predictions for each asset class
        """
        predictions = {}
        
        for asset_class in self.asset_columns.keys():
            if asset_class in self.models:
                try:
                    prediction = self.predict_returns(asset_class, horizon)
                    predictions[asset_class] = prediction
                except Exception as e:
                    logger.error(f"Failed to predict returns for {asset_class}: {e}")
                    predictions[asset_class] = None
        
        return predictions
    
    def validate_model_accuracy(self, asset_class: str, test_years: int = 5) -> Dict[str, float]:
        """
        Validate model accuracy using recent historical data.
        
        Args:
            asset_class: Asset class to validate
            test_years: Number of recent years to use for validation
            
        Returns:
            Dictionary with validation metrics
        """
        if asset_class not in self.models:
            raise ValueError(f"Model for {asset_class} not trained")
        
        if self.historical_data is None:
            self.load_historical_data()
        
        # Get recent data for validation
        recent_data = self.historical_data.tail(test_years + 5).copy()
        target_col = self.asset_columns[asset_class]
        
        predictions = []
        actuals = []
        
        # Make predictions for each of the test years
        for i in range(len(recent_data) - test_years, len(recent_data)):
            # Use data up to year i-1 to predict year i
            historical_subset = recent_data.iloc[:i].copy()
            
            # Temporarily set historical_data for prediction
            original_data = self.historical_data
            self.historical_data = historical_subset
            
            try:
                pred = self.predict_returns(asset_class, 1)
                actual = recent_data.iloc[i][target_col]
                
                predictions.append(pred)
                actuals.append(actual)
            except Exception as e:
                logger.warning(f"Failed to validate year {recent_data.iloc[i]['Year']}: {e}")
            
            # Restore original data
            self.historical_data = original_data
        
        if len(predictions) == 0:
            return {'error': 'No valid predictions for validation'}
        
        # Calculate validation metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actuals - predictions))
        
        # Direction accuracy (did we predict the right direction?)
        direction_correct = np.sum(np.sign(predictions) == np.sign(actuals))
        direction_accuracy = direction_correct / len(predictions)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'direction_accuracy': direction_accuracy,
            'predictions': predictions.tolist(),
            'actuals': actuals.tolist(),
            'test_years': test_years
        }
    
    def save_models(self, model_dir: str = "saved_models") -> None:
        """
        Save trained models and scalers to disk.
        
        Args:
            model_dir: Directory to save models
        """
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        for asset_class in self.models.keys():
            # Save model
            model_file = model_path / f"{asset_class}_model.joblib"
            joblib.dump(self.models[asset_class], model_file)
            
            # Save scaler
            scaler_file = model_path / f"{asset_class}_scaler.joblib"
            joblib.dump(self.scalers[asset_class], scaler_file)
        
        logger.info(f"Saved {len(self.models)} models to {model_dir}")
    
    def load_models(self, model_dir: str = "saved_models") -> None:
        """
        Load trained models and scalers from disk.
        
        Args:
            model_dir: Directory containing saved models
        """
        model_path = Path(model_dir)
        
        if not model_path.exists():
            raise ValueError(f"Model directory {model_dir} does not exist")
        
        loaded_count = 0
        for asset_class in self.asset_columns.keys():
            model_file = model_path / f"{asset_class}_model.joblib"
            scaler_file = model_path / f"{asset_class}_scaler.joblib"
            
            if model_file.exists() and scaler_file.exists():
                self.models[asset_class] = joblib.load(model_file)
                self.scalers[asset_class] = joblib.load(scaler_file)
                loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} models from {model_dir}")
    
    def get_model_summary(self) -> Dict[str, Dict]:
        """
        Get summary information about trained models.
        
        Returns:
            Dictionary with model information
        """
        summary = {}
        
        for asset_class, model in self.models.items():
            model_info = {
                'model_type': type(model).__name__,
                'trained': True,
                'asset_name': self.asset_columns[asset_class]
            }
            
            # Add model-specific info
            if hasattr(model, 'n_estimators'):
                model_info['n_estimators'] = model.n_estimators
            if hasattr(model, 'max_depth'):
                model_info['max_depth'] = model.max_depth
            
            summary[asset_class] = model_info
        
        return summary