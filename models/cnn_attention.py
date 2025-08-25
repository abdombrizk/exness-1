#!/usr/bin/env python3
"""
CNN + Attention Model for Gold Trading
Advanced CNN architecture with attention mechanism for pattern recognition

Author: AI Trading Systems
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score


class AttentionModule(nn.Module):
    """Attention mechanism for CNN features"""
    
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Attention layers
        self.attention_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
    def forward(self, x):
        """
        Apply attention mechanism
        
        Args:
            x (torch.Tensor): Input features (batch_size, seq_len, feature_dim)
            
        Returns:
            tuple: (attended_features, attention_weights)
        """
        # Calculate attention scores
        attention_scores = self.attention_fc(x)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Apply attention weights
        attended_features = torch.sum(x * attention_weights, dim=1)  # (batch_size, feature_dim)
        
        return attended_features, attention_weights.squeeze(-1)


class CNNAttentionModel(nn.Module):
    """
    CNN + Attention model for gold price prediction
    Uses multiple CNN layers with different kernel sizes and attention mechanism
    """
    
    def __init__(self, input_channels=1, sequence_length=60, feature_dim=50, 
                 num_filters=64, kernel_sizes=[3, 5, 7], num_classes=5, dropout=0.2):
        super(CNNAttentionModel, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.num_classes = num_classes
        
        # Multi-scale CNN layers
        self.conv_layers = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            conv_block = nn.Sequential(
                nn.Conv1d(
                    in_channels=feature_dim,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    padding=kernel_size//2
                ),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.Conv1d(
                    in_channels=num_filters,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    padding=kernel_size//2
                ),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv_block)
        
        # Attention mechanism
        total_filters = num_filters * len(kernel_sizes)
        self.attention = AttentionModule(total_filters)
        
        # Global pooling layers
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Feature fusion
        fusion_dim = total_filters * 2  # avg + max pooling
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim + total_filters, fusion_dim),  # +total_filters for attention
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim // 2, num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim)
            
        Returns:
            torch.Tensor: Output predictions
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # Transpose for CNN: (batch_size, feature_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Multi-scale CNN processing
        conv_outputs = []
        
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x)  # (batch_size, num_filters, seq_len)
            conv_outputs.append(conv_out)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat(conv_outputs, dim=1)  # (batch_size, total_filters, seq_len)
        
        # Transpose back for attention: (batch_size, seq_len, total_filters)
        multi_scale_features = multi_scale_features.transpose(1, 2)
        
        # Apply attention mechanism
        attended_features, attention_weights = self.attention(multi_scale_features)
        
        # Global pooling on original features
        pooling_input = multi_scale_features.transpose(1, 2)  # (batch_size, total_filters, seq_len)
        
        avg_pooled = self.global_avg_pool(pooling_input).squeeze(-1)  # (batch_size, total_filters)
        max_pooled = self.global_max_pool(pooling_input).squeeze(-1)  # (batch_size, total_filters)
        
        # Combine all features
        combined_features = torch.cat([avg_pooled, max_pooled, attended_features], dim=1)
        
        # Feature fusion
        fused_features = self.feature_fusion(combined_features)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output
        
    def train_model(self, train_sequences, train_labels, val_sequences, val_labels,
                   epochs=100, learning_rate=0.001, batch_size=32):
        """
        Train the CNN+Attention model
        
        Args:
            train_sequences (np.ndarray): Training sequences
            train_labels (np.ndarray): Training labels
            val_sequences (np.ndarray): Validation sequences
            val_labels (np.ndarray): Validation labels
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            
        Returns:
            dict: Training results
        """
        try:
            # Setup optimizer
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
            
            # Convert to tensors
            train_sequences = torch.FloatTensor(train_sequences).to(self.device)
            train_labels = torch.LongTensor(train_labels + 2).to(self.device)  # Shift labels to 0-4
            val_sequences = torch.FloatTensor(val_sequences).to(self.device)
            val_labels = torch.LongTensor(val_labels + 2).to(self.device)
            
            # Training history
            train_losses = []
            val_losses = []
            val_accuracies = []
            
            best_val_accuracy = 0.0
            patience_counter = 0
            early_stopping_patience = 20
            
            print(f"🚀 Training CNN+Attention on {self.device}")
            print(f"   Training samples: {len(train_sequences)}")
            print(f"   Validation samples: {len(val_sequences)}")
            
            for epoch in range(epochs):
                # Training phase
                self.train()
                train_loss = 0.0
                num_batches = 0
                
                for i in range(0, len(train_sequences), batch_size):
                    batch_sequences = train_sequences[i:i+batch_size]
                    batch_labels = train_labels[i:i+batch_size]
                    
                    self.optimizer.zero_grad()
                    
                    outputs = self.forward(batch_sequences)
                    loss = self.criterion(outputs, batch_labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                    num_batches += 1
                
                avg_train_loss = train_loss / num_batches
                train_losses.append(avg_train_loss)
                
                # Validation phase
                self.eval()
                val_loss = 0.0
                val_predictions = []
                val_targets = []
                
                with torch.no_grad():
                    for i in range(0, len(val_sequences), batch_size):
                        batch_sequences = val_sequences[i:i+batch_size]
                        batch_labels = val_labels[i:i+batch_size]
                        
                        outputs = self.forward(batch_sequences)
                        loss = self.criterion(outputs, batch_labels)
                        
                        val_loss += loss.item()
                        
                        predictions = torch.argmax(outputs, dim=1)
                        val_predictions.extend(predictions.cpu().numpy())
                        val_targets.extend(batch_labels.cpu().numpy())
                
                avg_val_loss = val_loss / (len(val_sequences) // batch_size + 1)
                val_losses.append(avg_val_loss)
                
                val_accuracy = accuracy_score(val_targets, val_predictions)
                val_accuracies.append(val_accuracy)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    # Save best model
                    torch.save(self.state_dict(), 'models/trained_models/cnn_attention_best.pth')
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
            self.load_state_dict(torch.load('models/trained_models/cnn_attention_best.pth'))
            
            results = {
                'best_val_accuracy': best_val_accuracy,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'epochs_trained': epoch + 1
            }
            
            print(f"✅ CNN+Attention training complete!")
            print(f"   Best validation accuracy: {best_val_accuracy:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ CNN+Attention training error: {e}")
            raise
            
    def predict(self, sequences):
        """
        Make predictions on input sequences
        
        Args:
            sequences (np.ndarray): Input sequences
            
        Returns:
            np.ndarray: Predictions
        """
        try:
            self.eval()
            
            if isinstance(sequences, np.ndarray):
                sequences = torch.FloatTensor(sequences).to(self.device)
            
            with torch.no_grad():
                outputs = self.forward(sequences)
                predictions = torch.argmax(outputs, dim=1)
                predictions = predictions.cpu().numpy() - 2  # Shift back to -2 to 2
                
            return predictions
            
        except Exception as e:
            print(f"❌ CNN+Attention prediction error: {e}")
            raise
            
    def get_attention_weights(self, sequences):
        """
        Get attention weights for interpretability
        
        Args:
            sequences (np.ndarray): Input sequences
            
        Returns:
            np.ndarray: Attention weights
        """
        try:
            self.eval()
            
            if isinstance(sequences, np.ndarray):
                sequences = torch.FloatTensor(sequences).to(self.device)
            
            with torch.no_grad():
                # Forward pass to get attention weights
                x = sequences.transpose(1, 2)
                
                # Multi-scale CNN processing
                conv_outputs = []
                for conv_layer in self.conv_layers:
                    conv_out = conv_layer(x)
                    conv_outputs.append(conv_out)
                
                # Concatenate and get attention
                multi_scale_features = torch.cat(conv_outputs, dim=1)
                multi_scale_features = multi_scale_features.transpose(1, 2)
                
                _, attention_weights = self.attention(multi_scale_features)
                
            return attention_weights.cpu().numpy()
            
        except Exception as e:
            print(f"❌ Error getting attention weights: {e}")
            return None
