#!/usr/bin/env python3
"""
Trained Model Loader for Gold Trading AI
Loads and manages trained models for production use

Author: AI Trading Systems
Version: 1.0.0
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_aligner import FeatureAligner


class TrainedModelLoader:
    """Loads and manages trained models for production use"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.metadata = {}
        self.is_loaded = False
        self.feature_aligner = FeatureAligner()

        print("📦 Trained Model Loader initialized")
        
    def load_models(self):
        """Load all available trained models"""
        print("\n📦 Loading trained models...")
        
        model_dir = "models/trained_models"
        if not os.path.exists(model_dir):
            print("   ❌ No trained models directory found")
            return False
            
        loaded_count = 0
        
        # Try to load compatible models first, then enhanced, then quick
        model_priorities = [
            [("random_forest_compatible.joblib", "Random Forest"),
             ("xgboost_compatible.joblib", "XGBoost"),
             ("lightgbm_compatible.joblib", "LightGBM")],
            [("random_forest_enhanced.joblib", "Random Forest"),
             ("xgboost_enhanced.joblib", "XGBoost"),
             ("lightgbm_enhanced.joblib", "LightGBM")],
            [("random_forest_quick.joblib", "Random Forest"),
             ("xgboost_quick.joblib", "XGBoost"),
             ("lightgbm_quick.joblib", "LightGBM")]
        ]
        
        # Try working models first, then each priority level
        working_models = [
            ("random_forest_working.joblib", "Random Forest"),
            ("xgboost_working.joblib", "XGBoost"),
            ("lightgbm_working.joblib", "LightGBM")
        ]

        # Try working models first
        for model_file, model_name in working_models:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                    loaded_count += 1
                    print(f"   ✅ {model_name} (working)")
                except Exception as e:
                    print(f"   ⚠️  Could not load {model_name}: {e}")

        model_type_loaded = "working" if loaded_count > 0 else None

        # Try other priority levels if no working models found
        if loaded_count == 0:
            for priority_level, model_list in enumerate(model_priorities):
                if loaded_count > 0:
                    break

            for model_file, model_name in model_list:
                model_path = os.path.join(model_dir, model_file)
                if os.path.exists(model_path):
                    try:
                        model = joblib.load(model_path)
                        self.models[model_name] = model
                        loaded_count += 1

                        if priority_level == 0:
                            model_type = "compatible"
                        elif priority_level == 1:
                            model_type = "enhanced"
                        else:
                            model_type = "quick"

                        if model_type_loaded is None:
                            model_type_loaded = model_type

                        print(f"   ✅ {model_name} ({model_type})")
                    except Exception as e:
                        print(f"   ⚠️  Could not load {model_name}: {e}")

        # Load appropriate scaler and metadata
        if model_type_loaded:
            scaler_files = [f"scaler_{model_type_loaded}.joblib", "scaler_enhanced.joblib", "scaler_quick.joblib"]
            metadata_files = [f"{model_type_loaded}_results.json", "enhanced_results.json", "quick_results.json"]
        else:
            scaler_files = ["scaler_working.joblib", "scaler_enhanced.joblib", "scaler_quick.joblib"]
            metadata_files = ["working_results.json", "enhanced_results.json", "quick_results.json"]

        # Load scaler
        for scaler_file in scaler_files:
            scaler_path = os.path.join(model_dir, scaler_file)
            if os.path.exists(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    print(f"   ✅ Scaler loaded ({scaler_file})")
                    break
                except Exception as e:
                    print(f"   ⚠️  Could not load scaler: {e}")

        # Load metadata
        for metadata_file in metadata_files:
            metadata_path = os.path.join(model_dir, metadata_file)
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    print(f"   ✅ Metadata loaded ({metadata_file})")
                    break
                except Exception as e:
                    print(f"   ⚠️  Could not load metadata: {e}")
                        
        if loaded_count > 0:
            self.is_loaded = True
            print(f"✅ Successfully loaded {loaded_count} models")

            # Load feature alignment schema
            if not self.feature_aligner.load_training_features():
                print("⚠️  Creating feature schema...")
                self.feature_aligner.create_feature_schema_from_enhanced_training()

            # Print model performance if available
            if self.metadata:
                print(f"\n📊 Model Performance:")
                for model_name, metrics in self.metadata.items():
                    if isinstance(metrics, dict) and 'accuracy' in metrics:
                        accuracy = metrics['accuracy']
                        status = "🎯" if accuracy >= 0.90 else "✅" if accuracy >= 0.85 else "👍"
                        print(f"   {status} {model_name}: {accuracy:.3f}")

            return True
        else:
            print("❌ No models could be loaded")
            return False
            
    def predict(self, features):
        """Make predictions using loaded models"""
        if not self.is_loaded:
            print("❌ No models loaded")
            return None
            
        try:
            # Ensure features is a DataFrame
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            elif isinstance(features, (list, np.ndarray)):
                features = pd.DataFrame(features)

            # Align features to match training schema
            aligned_features = self.feature_aligner.align_features(features)

            # Select numeric features only
            numeric_features = aligned_features.select_dtypes(include=[np.number])

            if len(numeric_features.columns) == 0:
                print("❌ No numeric features found")
                return None

            # Scale features if scaler available
            if self.scaler is not None:
                try:
                    features_scaled = self.scaler.transform(numeric_features)
                    features_to_use = features_scaled
                except Exception as e:
                    print(f"⚠️  Scaling error: {e}, using raw features")
                    features_to_use = numeric_features.values
            else:
                features_to_use = numeric_features.values
                
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for model_name, model in self.models.items():
                try:
                    # Get prediction
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(features_to_use)
                        pred = model.predict(features_to_use)
                        
                        # Store probabilities for confidence calculation
                        if pred_proba.shape[1] == 2:  # Binary classification
                            confidence = np.max(pred_proba, axis=1)
                            probabilities[model_name] = {
                                'prediction': pred[0] if len(pred) == 1 else pred,
                                'confidence': confidence[0] if len(confidence) == 1 else confidence,
                                'probabilities': pred_proba[0] if len(pred_proba) == 1 else pred_proba
                            }
                        else:
                            probabilities[model_name] = {
                                'prediction': pred[0] if len(pred) == 1 else pred,
                                'confidence': 0.5,
                                'probabilities': pred_proba[0] if len(pred_proba) == 1 else pred_proba
                            }
                    else:
                        pred = model.predict(features_to_use)
                        probabilities[model_name] = {
                            'prediction': pred[0] if len(pred) == 1 else pred,
                            'confidence': 0.5,
                            'probabilities': None
                        }
                        
                    predictions[model_name] = pred[0] if len(pred) == 1 else pred
                    
                except Exception as e:
                    print(f"⚠️  {model_name} prediction error: {e}")
                    
            if not predictions:
                print("❌ No successful predictions")
                return None
                
            # Create ensemble prediction
            pred_values = list(predictions.values())
            ensemble_pred = np.mean(pred_values)
            
            # Calculate ensemble confidence
            confidences = [prob['confidence'] for prob in probabilities.values() if 'confidence' in prob]
            ensemble_confidence = np.mean(confidences) if confidences else 0.5
            
            # Determine signal
            if ensemble_pred > 0.6:
                signal = "BUY"
                strength = "STRONG" if ensemble_confidence > 0.8 else "MODERATE"
            elif ensemble_pred < 0.4:
                signal = "SELL" 
                strength = "STRONG" if ensemble_confidence > 0.8 else "MODERATE"
            else:
                signal = "HOLD"
                strength = "NEUTRAL"
                
            result = {
                'signal': signal,
                'strength': strength,
                'confidence': ensemble_confidence,
                'ensemble_prediction': ensemble_pred,
                'individual_predictions': predictions,
                'individual_probabilities': probabilities,
                'models_used': list(self.models.keys())
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
            
    def get_model_info(self):
        """Get information about loaded models"""
        if not self.is_loaded:
            return {"loaded": False, "models": []}
            
        info = {
            "loaded": True,
            "models": list(self.models.keys()),
            "model_count": len(self.models),
            "has_scaler": self.scaler is not None,
            "metadata": self.metadata
        }
        
        return info
        
    def get_best_model(self):
        """Get the best performing model"""
        if not self.metadata:
            return list(self.models.keys())[0] if self.models else None
            
        best_model = None
        best_accuracy = 0
        
        for model_name, metrics in self.metadata.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    best_model = model_name
                    
        return best_model
        
    def predict_with_best_model(self, features):
        """Make prediction using only the best model"""
        best_model_name = self.get_best_model()
        
        if not best_model_name or best_model_name not in self.models:
            return self.predict(features)  # Fallback to ensemble
            
        try:
            # Prepare features
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            elif isinstance(features, (list, np.ndarray)):
                features = pd.DataFrame(features)

            # Align features
            aligned_features = self.feature_aligner.align_features(features)
            numeric_features = aligned_features.select_dtypes(include=[np.number])

            if self.scaler is not None:
                features_scaled = self.scaler.transform(numeric_features)
                features_to_use = features_scaled
            else:
                features_to_use = numeric_features.values
                
            model = self.models[best_model_name]
            
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(features_to_use)
                pred = model.predict(features_to_use)
                confidence = np.max(pred_proba, axis=1)[0]
            else:
                pred = model.predict(features_to_use)
                confidence = 0.5
                
            prediction = pred[0] if len(pred) == 1 else pred
            
            # Determine signal
            if prediction > 0.6:
                signal = "BUY"
            elif prediction < 0.4:
                signal = "SELL"
            else:
                signal = "HOLD"
                
            return {
                'signal': signal,
                'confidence': confidence,
                'prediction': prediction,
                'model_used': best_model_name
            }
            
        except Exception as e:
            print(f"❌ Best model prediction error: {e}")
            return None


# Global instance for easy access
trained_model_loader = TrainedModelLoader()
