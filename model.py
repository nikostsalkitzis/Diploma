import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # x: [seq_len, batch_size, embedding_dim]
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LSTMCNNClassifier(nn.Module):
    """
    LSTM + CNN based classifier for time series data.
    """

    def __init__(
        self,
        input_features: int = 48,        # <-- set to the real number of features
        hidden_size: int = 64,
        lstm_layers: int = 2,
        cnn_channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_patients: int = 10,
    ):
        super().__init__()

        self.input_features = input_features
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.cnn_channels = cnn_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        self.num_patients = num_patients

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_features,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # 1D CNN layer
        self.cnn = nn.Conv1d(
            in_channels=self.hidden_size * 2,  # bidirectional
            out_channels=self.cnn_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )

        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        # Fully connected classifier
        self.fc = nn.Linear(self.cnn_channels, self.num_patients)

    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, seq_len, input_features]
        returns: logits [batch_size, num_patients]
        """
        # Check shape early (optional)
        if x.size(-1) != self.input_features:
            raise ValueError(
                f"Expected last dim = {self.input_features}, got {x.size(-1)}"
            )

        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size*2]

        # CNN expects [batch_size, channels, seq_len]
        cnn_in = lstm_out.permute(0, 2, 1)
        cnn_out = F.relu(self.cnn(cnn_in))  # [batch_size, cnn_channels, seq_len]

        # Global average pooling over sequence
        pooled = cnn_out.mean(dim=2)  # [batch_size, cnn_channels]

        pooled = self.dropout(pooled)
        logits = self.fc(pooled)  # [batch_size, num_patients]

        return logits, pooled
