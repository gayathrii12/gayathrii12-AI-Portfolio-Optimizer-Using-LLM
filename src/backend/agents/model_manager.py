"""
Model Manager for ML Models Integration

This module manages the initialization, training, and loading of ML models
for the Financial Returns Optimizer agent workflow.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import os

from models.asset_return_models import AssetReturnModels

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages ML model lifecycle for the agent workflow.
    
    Handles model training, loading, saving, and provides a unified interface
    for agents to access trained models.
    """
    
    def __init__(self, data_file: str = "histretSP.xls", model_dir: str = "saved_models"):
        """
        Initialize the model manager.
        
        Args:
            data_file: Path to historical data file
            model_dir: Directory for saving/loading models
        """
        self.data_file = data_file
        self.model_dir = model_dir
        self.asset_models: Optional[AssetReturnModels] = None
        self._models_loaded = False
        
    def initialize_models(self, force_retrain: bool = False) -> bool:
        """
        Initialize ML models by loading existing models or training new ones.
        
        Args:
            force_retrain: If True, retrain models even if saved models exist
            
        Returns:
            bool: True if models were successfully initialized
        """
        try:
            logger.info("Initializing ML models...")
            
            # Create AssetReturnModels instance
            self.asset_models = AssetReturnModels(self.data_file)
            
            # Check if data file exists
            if not os.path.exists(self.data_file):
                logger.error(f"Historical data file not found: {self.data_file}")
                return False
            
            # Load historical data
            logger.info("Loading historical data...")
            self.asset_models.load_historical_data()
            
            # Try to load existing models first (unless force retrain)
            models_exist = self._check_saved_models_exist()
            
            if models_exist and not force_retrain:
                logger.info("Loading existing trained models...")
                try:
                    self.asset_models.load_models(self.model_dir)
                    self._models_loaded = True
                    logger.info("Successfully loaded existing models")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load existing models: {e}")
                    logger.info("Will train new models instead...")
            
            # Train new models
            logger.info("Training ML models for all asset classes...")
            training_results = self.asset_models.train_all_models('random_forest')
            
            # Check training results
            successful_models = []
            failed_models = []
            
            for asset_class, metrics in training_results.items():
                if 'error' in metrics:
                    failed_models.append(asset_class)
                    logger.warning(f"Failed to train model for {asset_class}: {metrics['error']}")
                else:
                    successful_models.append(asset_class)
                    logger.info(f"Successfully trained {asset_class} model (R²: {metrics.get('test_r2', 'N/A'):.3f})")
            
            if len(successful_models) == 0:
                logger.error("Failed to train any models")
                return False
            
            # Save trained models
            logger.info("Saving trained models...")
            self.asset_models.save_models(self.model_dir)
            
            self._models_loaded = True
            logger.info(f"Model initialization complete. Successfully trained {len(successful_models)} models.")
            
            return True
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False
    
    def _check_saved_models_exist(self) -> bool:
        """
        Check if saved models exist for all asset classes.
        
        Returns:
            bool: True if all model files exist
        """
        model_path = Path(self.model_dir)
        if not model_path.exists():
            return False
        
        asset_classes = ['sp500', 'small_cap', 't_bills', 't_bonds', 
                        'corporate_bonds', 'real_estate', 'gold']
        
        for asset_class in asset_classes:
            model_file = model_path / f"{asset_class}_model.joblib"
            scaler_file = model_path / f"{asset_class}_scaler.joblib"
            
            if not (model_file.exists() and scaler_file.exists()):
                return False
        
        return True
    
    def get_asset_models(self) -> Optional[AssetReturnModels]:
        """
        Get the initialized AssetReturnModels instance.
        
        Returns:
            AssetReturnModels instance or None if not initialized
        """
        if not self._models_loaded:
            logger.warning("Models not loaded. Call initialize_models() first.")
            return None
        
        return self.asset_models
    
    def validate_models(self) -> Dict[str, Any]:
        """
        Validate all trained models and return validation metrics.
        
        Returns:
            Dict with validation results for each model
        """
        if not self._models_loaded or not self.asset_models:
            return {'error': 'Models not loaded'}
        
        validation_results = {}
        
        for asset_class in self.asset_models.models.keys():
            try:
                # Test prediction
                test_prediction = self.asset_models.predict_returns(asset_class, 1)
                
                # Validate model accuracy if possible
                try:
                    accuracy_metrics = self.asset_models.validate_model_accuracy(asset_class, 5)
                    validation_results[asset_class] = {
                        'status': 'valid',
                        'test_prediction': test_prediction,
                        'accuracy_metrics': accuracy_metrics
                    }
                except Exception as e:
                    validation_results[asset_class] = {
                        'status': 'valid_with_warnings',
                        'test_prediction': test_prediction,
                        'validation_warning': str(e)
                    }
                    
            except Exception as e:
                validation_results[asset_class] = {
                    'status': 'invalid',
                    'error': str(e)
                }
        
        return validation_results
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary information about loaded models.
        
        Returns:
            Dict with model summary information
        """
        if not self._models_loaded or not self.asset_models:
            return {'error': 'Models not loaded'}
        
        summary = self.asset_models.get_model_summary()
        summary['models_loaded'] = self._models_loaded
        summary['data_file'] = self.data_file
        summary['model_dir'] = self.model_dir
        summary['total_models'] = len(self.asset_models.models)
        
        return summary
    
    def retrain_model(self, asset_class: str) -> Dict[str, Any]:
        """
        Retrain a specific model.
        
        Args:
            asset_class: Asset class to retrain
            
        Returns:
            Dict with retraining results
        """
        if not self.asset_models:
            return {'error': 'Models not initialized'}
        
        try:
            logger.info(f"Retraining model for {asset_class}...")
            metrics = self.asset_models.train_model(asset_class, 'random_forest')
            
            # Save the retrained model
            self.asset_models.save_models(self.model_dir)
            
            logger.info(f"Successfully retrained {asset_class} model")
            return {
                'status': 'success',
                'asset_class': asset_class,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to retrain {asset_class} model: {e}")
            return {
                'status': 'failed',
                'asset_class': asset_class,
                'error': str(e)
            }


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """
    Get the global model manager instance.
    
    Returns:
        ModelManager: Global model manager instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def initialize_global_models(force_retrain: bool = False) -> bool:
    """
    Initialize the global model manager.
    
    Args:
        force_retrain: If True, retrain models even if saved models exist
        
    Returns:
        bool: True if initialization was successful
    """
    model_manager = get_model_manager()
    return model_manager.initialize_models(force_retrain)