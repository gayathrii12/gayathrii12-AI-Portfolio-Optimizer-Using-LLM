"""
Workflow Factory for Financial Returns Optimizer

This module provides factory functions to create properly initialized
workflows with ML models for the Financial Returns Optimizer.
"""

import logging
from typing import Optional

from agents.langgraph_workflow import FinancialPlanningWorkflow
from agents.model_manager import ModelManager, get_model_manager
from models.asset_return_models import AssetReturnModels

logger = logging.getLogger(__name__)


class WorkflowFactory:
    """
    Factory class for creating properly initialized workflows with ML models.
    """
    
    @staticmethod
    def create_workflow(force_retrain_models: bool = False) -> Optional[FinancialPlanningWorkflow]:
        """
        Create a fully initialized workflow with ML models.
        
        Args:
            force_retrain_models: If True, retrain models even if saved models exist
            
        Returns:
            FinancialPlanningWorkflow instance or None if initialization fails
        """
        try:
            logger.info("Creating workflow with ML models...")
            
            # Get model manager and initialize models
            model_manager = get_model_manager()
            
            if not model_manager.initialize_models(force_retrain_models):
                logger.error("Failed to initialize ML models for workflow")
                return None
            
            # Get initialized models
            asset_models = model_manager.get_asset_models()
            if asset_models is None:
                logger.error("Failed to get initialized asset models")
                return None
            
            # Create workflow with models
            workflow = FinancialPlanningWorkflow(asset_models)
            
            # Validate models
            validation_results = model_manager.validate_models()
            valid_models = sum(1 for result in validation_results.values() 
                             if result.get('status') == 'valid')
            total_models = len(validation_results)
            
            logger.info(f"Workflow created successfully with {valid_models}/{total_models} valid ML models")
            
            if valid_models == 0:
                logger.error("No valid ML models available")
                return None
            
            return workflow
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            return None
    
    @staticmethod
    def create_workflow_with_models(asset_models: AssetReturnModels) -> Optional[FinancialPlanningWorkflow]:
        """
        Create workflow with pre-initialized models (for testing).
        
        Args:
            asset_models: Pre-initialized AssetReturnModels instance
            
        Returns:
            FinancialPlanningWorkflow instance or None if creation fails
        """
        try:
            logger.info("Creating workflow with provided models...")
            workflow = FinancialPlanningWorkflow(asset_models)
            logger.info("Workflow created successfully with provided models")
            return workflow
            
        except Exception as e:
            logger.error(f"Failed to create workflow with provided models: {e}")
            return None
    
    @staticmethod
    def get_model_status() -> dict:
        """
        Get status of ML models.
        
        Returns:
            Dict with model status information
        """
        try:
            model_manager = get_model_manager()
            
            # Try to get models
            asset_models = model_manager.get_asset_models()
            if asset_models is None:
                return {
                    'status': 'not_initialized',
                    'message': 'ML models not initialized'
                }
            
            # Get model summary and validation
            summary = model_manager.get_model_summary()
            validation = model_manager.validate_models()
            
            valid_models = sum(1 for result in validation.values() 
                             if result.get('status') == 'valid')
            total_models = len(validation)
            
            return {
                'status': 'initialized',
                'total_models': total_models,
                'valid_models': valid_models,
                'model_summary': summary,
                'validation_results': validation
            }
            
        except Exception as e:
            logger.error(f"Failed to get model status: {e}")
            return {
                'status': 'error',
                'message': f'Failed to get model status: {str(e)}'
            }


# Convenience functions
def create_workflow(force_retrain_models: bool = False) -> Optional[FinancialPlanningWorkflow]:
    """
    Convenience function to create a workflow.
    
    Args:
        force_retrain_models: If True, retrain models even if saved models exist
        
    Returns:
        FinancialPlanningWorkflow instance or None if initialization fails
    """
    return WorkflowFactory.create_workflow(force_retrain_models)


def get_model_status() -> dict:
    """
    Convenience function to get model status.
    
    Returns:
        Dict with model status information
    """
    return WorkflowFactory.get_model_status()