import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Multi-task Transformer model for time prediction and activity classification
    """

    def __init__(self, args):
        super().__init__()

        args_defaults = dict(
            input_features=8,  # Changed to 8 to match your data
            seq_len=24,        # Changed to 24 to match your window_size
            batch_first=False,
            device='cuda',
            d_model=32,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_head=8,
            dropout_encoder=0.2,
            dropout_pos_enc=0.1,
            dim_feedforward_encoder=2048,
            num_patients=8
        )

        for arg, default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)

        print(f"Model config: input_features={self.input_features}, seq_len={self.seq_len}")

        # Input Embedding (Encoder)
        self.encoder_input_layer = nn.Linear(
            in_features=self.input_features,  # Should be 8
            out_features=self.d_model
        )

        # Positional Encoding
        self.positional_encoding_layer = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout_pos_enc
        )

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward_encoder,
            dropout=self.dropout_encoder,
            batch_first=self.batch_first
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.n_encoder_layers,
        )

        # Adaptive pooling for sequence to vector
        self.adaptive = nn.AdaptiveAvgPool1d(1)
        
        # Multi-task prediction heads
        # Task 1: Time prediction (regression)
        self.time_predictor = nn.Linear(self.d_model, 2)  # sin_t, cos_t
        
        # Task 2: Activity level classification (3 classes: low/medium/high)
        self.activity_predictor = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)  # 3 activity levels
        )

    def forward(self, src: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            src: Tensor of shape (batch_size, seq_len, input_features) = (16, 24, 8)
        """
        # Debug: Check input shape
        #print(f"Model input shape: {src.shape}")
        
        batch_size, seq_len, input_features = src.shape
        
        # Ensure correct input dimensions
        if input_features != self.input_features:
            raise ValueError(f"Expected {self.input_features} input features, got {input_features}")
        
        # The original code expects (seq_len, batch_size, features) for transformer
        # But our input is (batch_size, seq_len, features)
        # So we need to permute: (batch_size, seq_len, features) -> (seq_len, batch_size, features)
        src = src.permute(1, 0, 2)  # Now: (seq_len, batch_size, input_features)
        #print(f"After permute: {src.shape}")

        # Input Embedding (Encoder)
        src = self.encoder_input_layer(src)  # (seq_len, batch_size, d_model)
        #print(f"After encoder input layer: {src.shape}")

        # Positional Encoding (Encoder)
        src = self.positional_encoding_layer(src)  # (seq_len, batch_size, d_model)
        #print(f"After positional encoding: {src.shape}")

        # Encoder
        src = self.encoder(src=src)  # (seq_len, batch_size, d_model)
        #print(f"After transformer encoder: {src.shape}")

        # Get features: (seq_len, batch_size, d_model) -> (batch_size, d_model, seq_len) -> pooled
        features = src.permute(1, 2, 0).contiguous()  # (batch_size, d_model, seq_len)
        #print(f"After permute for pooling: {features.shape}")
        
        features = self.adaptive(features)  # (batch_size, d_model, 1)
        #print(f"After adaptive pooling: {features.shape}")
        
        features = features.squeeze(-1)  # (batch_size, d_model)
        #print(f"Final features shape: {features.shape}")
        
        # Multi-task predictions
        time_pred = self.time_predictor(features)  # (batch_size, 2)
        activity_pred = self.activity_predictor(features)  # (batch_size, 3)
        
        #print(f"Time prediction shape: {time_pred.shape}")
        #print(f"Activity prediction shape: {activity_pred.shape}")
        
        return {
            'time_pred': time_pred,
            'activity_pred': activity_pred,
            'features': features
        }
