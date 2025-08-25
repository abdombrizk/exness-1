#!/usr/bin/env python3
"""
Model Management System
======================

Centralized model management for loading, saving, and managing ML models.
Handles model versioning, metadata, and performance tracking.

Author: AI Trading Systems
Version: 2.0.0
"""

import os
import json
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

from ...utils.logger import get_logger
from ...utils.config import get_config

logger = get_logger(__name__)


class ModelMetadata:
    """Model metadata container"""
    
    def __init__(self, model_name: str, model_type: str, version: str = "1.0"):
        self.model_name = model_name
        self.model_type = model_type
        self.version = version
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.accuracy = 0.0
        self.feature_count = 0
        self.feature_names = []
        self.hyperparameters = {}
        self.training_samples = 0
        self.validation_samples = 0
        self.target_column = ""
        self.performance_metrics = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'accuracy': self.accuracy,
            'feature_count': self.feature_count,
            'feature_names': self.feature_names,
            'hyperparameters': self.hyperparameters,
            'training_samples': self.training_samples,
            'validation_samples': self.validation_samples,
            'target_column': self.target_column,
            'performance_metrics': self.performance_metrics
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary"""
        metadata = cls(
            model_name=data['model_name'],
            model_type=data['model_type'],
            version=data.get('version', '1.0')
        )
        
        metadata.created_at = datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
        metadata.updated_at = datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat()))
        metadata.accuracy = data.get('accuracy', 0.0)
        metadata.feature_count = data.get('feature_count', 0)
        metadata.feature_names = data.get('feature_names', [])
        metadata.hyperparameters = data.get('hyperparameters', {})
        metadata.training_samples = data.get('training_samples', 0)
        metadata.validation_samples = data.get('validation_samples', 0)
        metadata.target_column = data.get('target_column', '')
        metadata.performance_metrics = data.get('performance_metrics', {})
        
        return metadata


class ModelManager:
    """Centralized model management system"""
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize model manager
        
        Args:
            models_dir: Directory to store models (default: from config)
        """
        self.models_dir = Path(models_dir or get_config('models.directory', 'data/models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.models_dir / 'models_metadata.json'
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, ModelMetadata] = {}
        
        self._load_metadata()
        
    def _load_metadata(self):
        """Load model metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    
                for model_id, data in metadata_dict.items():
                    self.model_metadata[model_id] = ModelMetadata.from_dict(data)
                    
                logger.info(f"Loaded metadata for {len(self.model_metadata)} models")
            else:
                logger.info("No existing model metadata found")
                
        except Exception as e:
            logger.error(f"Error loading model metadata: {e}")
            
    def _save_metadata(self):
        """Save model metadata to file"""
        try:
            metadata_dict = {}
            for model_id, metadata in self.model_metadata.items():
                metadata_dict[model_id] = metadata.to_dict()
                
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
                
            logger.debug("Model metadata saved")
            
        except Exception as e:
            logger.error(f"Error saving model metadata: {e}")
            
    def save_model(self, model: Any, model_name: str, model_type: str, 
                   scaler: Any = None, feature_names: List[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save model with metadata
        
        Args:
            model: Trained model object
            model_name: Name of the model
            model_type: Type of model (e.g., 'RandomForest', 'XGBoost')
            scaler: Feature scaler object
            feature_names: List of feature names
            metadata: Additional metadata
            
        Returns:
            Model ID for future reference
        """
        try:
            # Generate model ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"{model_name}_{model_type}_{timestamp}"
            
            # Create model directory
            model_dir = self.models_dir / model_id
            model_dir.mkdir(exist_ok=True)
            
            # Save model
            model_file = model_dir / 'model.joblib'
            joblib.dump(model, model_file)
            
            # Save scaler if provided
            if scaler is not None:
                scaler_file = model_dir / 'scaler.joblib'
                joblib.dump(scaler, scaler_file)
                
            # Create and save metadata
            model_metadata = ModelMetadata(model_name, model_type)
            
            if feature_names:
                model_metadata.feature_names = feature_names
                model_metadata.feature_count = len(feature_names)
                
            if metadata:
                for key, value in metadata.items():
                    if hasattr(model_metadata, key):
                        setattr(model_metadata, key, value)
                        
            # Save metadata to model directory
            metadata_file = model_dir / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(model_metadata.to_dict(), f, indent=2)
                
            # Update global metadata
            self.model_metadata[model_id] = model_metadata
            self._save_metadata()
            
            logger.info(f"Model saved: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
            
    def load_model(self, model_id: str) -> Tuple[Any, Any, ModelMetadata]:
        """
        Load model by ID
        
        Args:
            model_id: Model identifier
            
        Returns:
            Tuple of (model, scaler, metadata)
        """
        try:
            if model_id in self.loaded_models:
                logger.debug(f"Model {model_id} already loaded")
                return self.loaded_models[model_id]
                
            model_dir = self.models_dir / model_id
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
                
            # Load model
            model_file = model_dir / 'model.joblib'
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
                
            model = joblib.load(model_file)
            
            # Load scaler if exists
            scaler_file = model_dir / 'scaler.joblib'
            scaler = None
            if scaler_file.exists():
                scaler = joblib.load(scaler_file)
                
            # Load metadata
            metadata_file = model_dir / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                metadata = ModelMetadata.from_dict(metadata_dict)
            else:
                metadata = self.model_metadata.get(model_id)
                
            # Cache loaded model
            self.loaded_models[model_id] = (model, scaler, metadata)
            
            logger.info(f"Model loaded: {model_id}")
            return model, scaler, metadata
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
            
    def get_best_model(self, model_type: Optional[str] = None) -> Optional[str]:
        """
        Get the best model ID based on accuracy
        
        Args:
            model_type: Filter by model type (optional)
            
        Returns:
            Best model ID or None if no models found
        """
        try:
            best_model_id = None
            best_accuracy = 0.0
            
            for model_id, metadata in self.model_metadata.items():
                if model_type and metadata.model_type != model_type:
                    continue
                    
                if metadata.accuracy > best_accuracy:
                    best_accuracy = metadata.accuracy
                    best_model_id = model_id
                    
            return best_model_id
            
        except Exception as e:
            logger.error(f"Error finding best model: {e}")
            return None
            
    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available models
        
        Args:
            model_type: Filter by model type (optional)
            
        Returns:
            List of model information dictionaries
        """
        try:
            models = []
            
            for model_id, metadata in self.model_metadata.items():
                if model_type and metadata.model_type != model_type:
                    continue
                    
                model_info = {
                    'model_id': model_id,
                    'model_name': metadata.model_name,
                    'model_type': metadata.model_type,
                    'version': metadata.version,
                    'accuracy': metadata.accuracy,
                    'created_at': metadata.created_at,
                    'feature_count': metadata.feature_count
                }
                
                models.append(model_info)
                
            # Sort by accuracy (descending)
            models.sort(key=lambda x: x['accuracy'], reverse=True)
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
            
    def delete_model(self, model_id: str) -> bool:
        """
        Delete model and its files
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from loaded models cache
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
                
            # Remove from metadata
            if model_id in self.model_metadata:
                del self.model_metadata[model_id]
                
            # Remove model directory
            model_dir = self.models_dir / model_id
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
                
            # Save updated metadata
            self._save_metadata()
            
            logger.info(f"Model deleted: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            return False
            
    def update_model_performance(self, model_id: str, 
                               performance_metrics: Dict[str, float]) -> bool:
        """
        Update model performance metrics
        
        Args:
            model_id: Model identifier
            performance_metrics: Dictionary of performance metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_id not in self.model_metadata:
                logger.error(f"Model not found: {model_id}")
                return False
                
            metadata = self.model_metadata[model_id]
            metadata.performance_metrics.update(performance_metrics)
            metadata.updated_at = datetime.now()
            
            # Update accuracy if provided
            if 'accuracy' in performance_metrics:
                metadata.accuracy = performance_metrics['accuracy']
                
            # Save updated metadata
            self._save_metadata()
            
            # Update model directory metadata
            model_dir = self.models_dir / model_id
            metadata_file = model_dir / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'w') as f:
                    json.dump(metadata.to_dict(), f, indent=2)
                    
            logger.info(f"Model performance updated: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
            return False
            
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed model information
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model information dictionary or None if not found
        """
        try:
            if model_id not in self.model_metadata:
                return None
                
            metadata = self.model_metadata[model_id]
            
            # Check if model files exist
            model_dir = self.models_dir / model_id
            model_file = model_dir / 'model.joblib'
            scaler_file = model_dir / 'scaler.joblib'
            
            model_info = metadata.to_dict()
            model_info.update({
                'model_id': model_id,
                'model_file_exists': model_file.exists(),
                'scaler_file_exists': scaler_file.exists(),
                'model_size_mb': model_file.stat().st_size / (1024 * 1024) if model_file.exists() else 0,
                'is_loaded': model_id in self.loaded_models
            })
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return None
            
    def cleanup_old_models(self, keep_count: int = 5) -> int:
        """
        Clean up old models, keeping only the best ones
        
        Args:
            keep_count: Number of models to keep per type
            
        Returns:
            Number of models deleted
        """
        try:
            deleted_count = 0
            
            # Group models by type
            models_by_type = {}
            for model_id, metadata in self.model_metadata.items():
                model_type = metadata.model_type
                if model_type not in models_by_type:
                    models_by_type[model_type] = []
                models_by_type[model_type].append((model_id, metadata))
                
            # Keep only best models for each type
            for model_type, models in models_by_type.items():
                # Sort by accuracy (descending)
                models.sort(key=lambda x: x[1].accuracy, reverse=True)
                
                # Delete excess models
                for model_id, metadata in models[keep_count:]:
                    if self.delete_model(model_id):
                        deleted_count += 1
                        
            logger.info(f"Cleaned up {deleted_count} old models")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up models: {e}")
            return 0
