import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerDualHead(nn.Module):
    """
    Transformer encoder with two heads:
    - Head 1: regress circadian time (sin_t, cos_t)
    - Head 2: regress sleep cycle (sin_onset, cos_onset, sin_wake, cos_wake)
    """

    def __init__(self, args):
        super().__init__()

        args_defaults = dict(
            input_features=10,
            seq_len=512,
            batch_first=False,
            device="cuda",
            d_model=64,
            n_encoder_layers=2,
            n_head=8,
            dropout_encoder=0.2,
            dropout_pos_enc=0.1,
            dim_feedforward_encoder=2048,
        )

        for arg, default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)

        # Input embedding
        self.encoder_input_layer = nn.Linear(
            in_features=self.input_features, out_features=self.d_model
        )

        # Positional encoding
        self.positional_encoding_layer = PositionalEncoding(
            d_model=self.d_model, dropout=self.dropout_pos_enc
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward_encoder,
            dropout=self.dropout_encoder,
            batch_first=self.batch_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=self.n_encoder_layers
        )

        # Pooling
        self.adaptive = nn.AdaptiveAvgPool1d(1)

        # --- HEAD 1: circadian (sin_t, cos_t) ---
        self.head_time = nn.Linear(self.d_model, 2)

        # --- HEAD 2: sleep cyclic features ---
        self.head_sleep = nn.Linear(self.d_model, 4)

    def forward(self, src: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            src: shape (batch, features, seq_len)

        Returns:
            features: latent embedding (batch, d_model)
            preds_time: circadian predictions (batch, 2)
            preds_sleep: sleep predictions (batch, 4)
        """
        # reshape to Transformer format
        src = src.permute(2, 0, 1)  # (seq_len, batch, features)

        # Input embedding
        src = self.encoder_input_layer(src)

        # Add positional encoding
        src = self.positional_encoding_layer(src)

        # Transformer encoder
        src = self.encoder(src=src)

        # Pool sequence
        features = src.permute(1, 2, 0).contiguous()
        features = self.adaptive(features).squeeze(-1)  # (batch, d_model)

        # Two heads
        preds_time = self.head_time(features)      # (batch, 2)
        preds_sleep = self.head_sleep(features)    # (batch, 4)

        return features, preds_time, preds_sleep
