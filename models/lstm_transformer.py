#!/usr/bin/env python3
"""
LSTM + Transformer Model for Gold Trading
Advanced hybrid architecture combining LSTM memory with Transformer attention

Author: AI Trading Systems
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class LSTMTransformerModel(nn.Module):
    """
    Hybrid LSTM + Transformer model for gold price prediction
    Combines LSTM's memory capabilities with Transformer's attention mechanism
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, num_heads=8, 
                 dropout=0.2, num_classes=5):
        super(LSTMTransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Adjust hidden dimension for bidirectional LSTM
        lstm_output_dim = hidden_dim * 2
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(lstm_output_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_output_dim,
            nhead=num_heads,
            dim_feedforward=lstm_output_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
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
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output predictions
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply positional encoding
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        lstm_out = self.pos_encoder(lstm_out)
        lstm_out = lstm_out.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        
        # Transformer processing
        transformer_out = self.transformer_encoder(lstm_out)
        
        # Self-attention
        attn_out, attn_weights = self.attention(
            transformer_out, transformer_out, transformer_out
        )
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)  # (batch_size, hidden_dim)
        
        # Classification
        output = self.classifier(pooled)
        
        return output
        
    def train_model(self, train_sequences, train_labels, val_sequences, val_labels, 
                   epochs=100, learning_rate=0.001, batch_size=32):
        """
        Train the LSTM+Transformer model
        
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
            
            print(f"🚀 Training LSTM+Transformer on {self.device}")
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
                    torch.save(self.state_dict(), 'models/trained_models/lstm_transformer_best.pth')
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
            self.load_state_dict(torch.load('models/trained_models/lstm_transformer_best.pth'))
            
            results = {
                'best_val_accuracy': best_val_accuracy,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'epochs_trained': epoch + 1
            }
            
            print(f"✅ LSTM+Transformer training complete!")
            print(f"   Best validation accuracy: {best_val_accuracy:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ LSTM+Transformer training error: {e}")
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
            print(f"❌ LSTM+Transformer prediction error: {e}")
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
                # Forward pass through LSTM
                lstm_out, _ = self.lstm(sequences)
                
                # Apply positional encoding
                lstm_out = lstm_out.transpose(0, 1)
                lstm_out = self.pos_encoder(lstm_out)
                lstm_out = lstm_out.transpose(0, 1)
                
                # Get attention weights
                _, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
                
            return attn_weights.cpu().numpy()
            
        except Exception as e:
            print(f"❌ Error getting attention weights: {e}")
            return None
