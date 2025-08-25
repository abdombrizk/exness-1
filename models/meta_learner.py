#!/usr/bin/env python3
"""
Meta-Learner for Ensemble Model Coordination
Advanced meta-learning system to combine predictions from multiple base models

Author: AI Trading Systems
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score


class MetaLearner(nn.Module):
    """
    Meta-Learning model to combine predictions from multiple base models
    Uses neural network to learn optimal combination weights
    """
    
    def __init__(self, num_base_models=4, hidden_dim=64, output_dim=5, dropout=0.3):
        super(MetaLearner, self).__init__()
        
        self.num_base_models = num_base_models
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Meta-learning network
        self.meta_network = nn.Sequential(
            nn.Linear(num_base_models, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Confidence estimation network
        self.confidence_network = nn.Sequential(
            nn.Linear(num_base_models + output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output confidence between 0 and 1
        )
        
        # Attention mechanism for base model weighting
        self.attention_weights = nn.Sequential(
            nn.Linear(num_base_models, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_base_models),
            nn.Softmax(dim=1)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.confidence_criterion = nn.MSELoss()
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
    def forward(self, base_predictions):
        """
        Forward pass through meta-learner
        
        Args:
            base_predictions (torch.Tensor): Predictions from base models
                                           Shape: (batch_size, num_base_models)
        
        Returns:
            tuple: (final_predictions, confidence_scores, attention_weights)
        """
        # Calculate attention weights for base models
        attention_weights = self.attention_weights(base_predictions)
        
        # Apply attention to base predictions
        weighted_predictions = base_predictions * attention_weights
        
        # Meta-learning prediction
        meta_output = self.meta_network(weighted_predictions)
        
        # Confidence estimation
        confidence_input = torch.cat([base_predictions, meta_output], dim=1)
        confidence_scores = self.confidence_network(confidence_input)
        
        return meta_output, confidence_scores, attention_weights
        
    def train_model(self, base_predictions, true_labels, epochs=50, learning_rate=0.001, 
                   batch_size=32, validation_split=0.2):
        """
        Train the meta-learner
        
        Args:
            base_predictions (np.ndarray): Predictions from base models
            true_labels (np.ndarray): True labels
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            validation_split (float): Validation split ratio
            
        Returns:
            dict: Training results
        """
        try:
            # Setup optimizer
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            # Convert to tensors
            base_predictions = torch.FloatTensor(base_predictions).to(self.device)
            true_labels = torch.LongTensor(true_labels + 2).to(self.device)  # Shift to 0-4
            
            # Split data
            split_idx = int(len(base_predictions) * (1 - validation_split))
            
            train_predictions = base_predictions[:split_idx]
            train_labels = true_labels[:split_idx]
            val_predictions = base_predictions[split_idx:]
            val_labels = true_labels[split_idx:]
            
            # Training history
            train_losses = []
            val_losses = []
            val_accuracies = []
            
            best_val_accuracy = 0.0
            patience_counter = 0
            early_stopping_patience = 15
            
            print(f"🚀 Training Meta-Learner on {self.device}")
            print(f"   Training samples: {len(train_predictions)}")
            print(f"   Validation samples: {len(val_predictions)}")
            
            for epoch in range(epochs):
                # Training phase
                self.train()
                train_loss = 0.0
                num_batches = 0
                
                for i in range(0, len(train_predictions), batch_size):
                    batch_predictions = train_predictions[i:i+batch_size]
                    batch_labels = train_labels[i:i+batch_size]
                    
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    meta_output, confidence_scores, _ = self.forward(batch_predictions)
                    
                    # Calculate losses
                    classification_loss = self.criterion(meta_output, batch_labels)
                    
                    # Confidence target (higher for correct predictions)
                    predicted_classes = torch.argmax(meta_output, dim=1)
                    confidence_targets = (predicted_classes == batch_labels).float().unsqueeze(1)
                    confidence_loss = self.confidence_criterion(confidence_scores, confidence_targets)
                    
                    # Combined loss
                    total_loss = classification_loss + 0.1 * confidence_loss
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    train_loss += total_loss.item()
                    num_batches += 1
                
                avg_train_loss = train_loss / num_batches
                train_losses.append(avg_train_loss)
                
                # Validation phase
                self.eval()
                val_loss = 0.0
                val_predictions_list = []
                val_targets = []
                
                with torch.no_grad():
                    for i in range(0, len(val_predictions), batch_size):
                        batch_predictions = val_predictions[i:i+batch_size]
                        batch_labels = val_labels[i:i+batch_size]
                        
                        meta_output, confidence_scores, _ = self.forward(batch_predictions)
                        
                        classification_loss = self.criterion(meta_output, batch_labels)
                        predicted_classes = torch.argmax(meta_output, dim=1)
                        confidence_targets = (predicted_classes == batch_labels).float().unsqueeze(1)
                        confidence_loss = self.confidence_criterion(confidence_scores, confidence_targets)
                        
                        total_loss = classification_loss + 0.1 * confidence_loss
                        val_loss += total_loss.item()
                        
                        predictions = torch.argmax(meta_output, dim=1)
                        val_predictions_list.extend(predictions.cpu().numpy())
                        val_targets.extend(batch_labels.cpu().numpy())
                
                avg_val_loss = val_loss / (len(val_predictions) // batch_size + 1)
                val_losses.append(avg_val_loss)
                
                val_accuracy = accuracy_score(val_targets, val_predictions_list)
                val_accuracies.append(val_accuracy)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    # Save best model
                    torch.save(self.state_dict(), 'models/trained_models/meta_learner_best.pth')
                else:
                    patience_counter += 1
                
                # Print progress
                if (epoch + 1) % 10 == 0:
                    print(f"   Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, "
                          f"Val Acc: {val_accuracy:.4f}")
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"   Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model
            self.load_state_dict(torch.load('models/trained_models/meta_learner_best.pth'))
            
            results = {
                'best_val_accuracy': best_val_accuracy,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'epochs_trained': epoch + 1
            }
            
            print(f"✅ Meta-Learner training complete!")
            print(f"   Best validation accuracy: {best_val_accuracy:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Meta-Learner training error: {e}")
            raise
            
    def predict(self, base_predictions):
        """
        Make predictions using the meta-learner
        
        Args:
            base_predictions (np.ndarray): Predictions from base models
            
        Returns:
            np.ndarray: Final ensemble predictions
        """
        try:
            self.eval()
            
            if isinstance(base_predictions, np.ndarray):
                base_predictions = torch.FloatTensor(base_predictions).to(self.device)
            
            with torch.no_grad():
                meta_output, _, _ = self.forward(base_predictions)
                predictions = torch.argmax(meta_output, dim=1)
                predictions = predictions.cpu().numpy() - 2  # Shift back to -2 to 2
                
            return predictions
            
        except Exception as e:
            print(f"❌ Meta-Learner prediction error: {e}")
            raise
            
    def get_confidence(self, base_predictions):
        """
        Get confidence scores for predictions
        
        Args:
            base_predictions (np.ndarray): Predictions from base models
            
        Returns:
            np.ndarray: Confidence scores (0-100)
        """
        try:
            self.eval()
            
            if isinstance(base_predictions, np.ndarray):
                base_predictions = torch.FloatTensor(base_predictions).to(self.device)
            
            with torch.no_grad():
                _, confidence_scores, _ = self.forward(base_predictions)
                confidence_scores = confidence_scores.cpu().numpy() * 100  # Convert to percentage
                
            return confidence_scores.flatten()
            
        except Exception as e:
            print(f"❌ Error getting confidence scores: {e}")
            return np.array([50.0])  # Default confidence
            
    def get_attention_weights(self, base_predictions):
        """
        Get attention weights for base models
        
        Args:
            base_predictions (np.ndarray): Predictions from base models
            
        Returns:
            np.ndarray: Attention weights for each base model
        """
        try:
            self.eval()
            
            if isinstance(base_predictions, np.ndarray):
                base_predictions = torch.FloatTensor(base_predictions).to(self.device)
            
            with torch.no_grad():
                _, _, attention_weights = self.forward(base_predictions)
                attention_weights = attention_weights.cpu().numpy()
                
            return attention_weights
            
        except Exception as e:
            print(f"❌ Error getting attention weights: {e}")
            return np.ones((1, self.num_base_models)) / self.num_base_models  # Equal weights
