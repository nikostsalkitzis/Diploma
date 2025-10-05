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
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EnsembleLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int,
                 weight_decay: float = 0., bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.lin_w = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.lin_b = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(x, self.lin_w)
        y = torch.add(w_times_x, self.lin_b[:, None, :])  # w times x + b
        return y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.lin_b is not None}"

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lin_w, a=math.sqrt(5))
        if self.lin_b is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.lin_w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.lin_b, -bound, bound)


class TransformerClassifier(nn.Module):
    """
    Multi-task Transformer:
      - Circadian regression head (sin_t, cos_t)
      - Relapse classification head (binary)
    """

    def __init__(self, args):
        super().__init__()

        args_defaults = dict(
            input_features=10,
            seq_len=512,
            batch_first=False,
            device="cuda",
            d_model=32,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_head=8,
            dropout_encoder=0.2,
            dropout_pos_enc=0.1,
            dim_feedforward_encoder=2048,
            num_patients=10,
        )

        for arg, default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)

        # Input Embedding (Encoder)
        self.encoder_input_layer = nn.Linear(self.input_features, self.d_model)

        # Positional Encoding
        self.positional_encoding_layer = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout_pos_enc,
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward_encoder,
            dropout=self.dropout_encoder,
            batch_first=self.batch_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.n_encoder_layers,
        )

        # Sequence pooling
        self.adaptive = nn.AdaptiveAvgPool1d(1)

        # 🔑 Multi-task heads
        self.circadian_head = nn.Linear(self.d_model, 2)   # regression: sin_t, cos_t
        self.relapse_head = nn.Linear(self.d_model, 1)     # classification: relapse yes/no

    def forward(self, src: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            src: shape (batch, features, seq_len)
        Returns:
            features: (batch, d_model)
            circadian_preds: (batch, 2)
            relapse_logits: (batch, 1)
        """
        # Reorder: (seq_len, batch, features)
        src = src.permute(2, 0, 1)

        # Input Embedding
        src = self.encoder_input_layer(src)

        # Positional Encoding
        src = self.positional_encoding_layer(src)

        # Transformer Encoder
        src = self.encoder(src=src)

        # Pool over sequence length
        features = src.permute(1, 2, 0).contiguous()   # (batch, d_model, seq_len)
        features = self.adaptive(features).squeeze(-1) # (batch, d_model)

        # Predictions
        circadian_preds = self.circadian_head(features)   # regression
        relapse_logits = self.relapse_head(features)      # classification

        return features, circadian_preds, relapse_logits
