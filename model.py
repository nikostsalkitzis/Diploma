import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_features,
        cnn_channels=128,
        lstm_hidden=32,
        lstm_layers=4,
        window_size=48,
        num_patients=10,
        device='cuda'
    ):
        super().__init__()
        
        # input parameters
        self.input_channels = input_features
        self.num_patients = num_patients
        self.hidden_size = lstm_hidden
        self.seq_len = window_size
        self.lstm_layers = lstm_layers
        self.device = device
        
        # CNN layers (feature extraction)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.input_channels, out_channels=cnn_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels//2),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels//2, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU()
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels,           # matches CNN out_channels
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,                   # input shape: (batch, seq_len, features)
            bidirectional=False
        )
        
        # Adaptive pooling and classifier
        self.adaptive = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.hidden_size, self.num_patients)
        
    def forward(self, x):
        """
        Args:
            x: input tensor, shape: (batch_size, channels, seq_len)
        Returns:
            logits: classification logits
            features: LSTM hidden representation (for outlier detection)
        """
        # CNN expects (batch_size, channels, seq_len)
        cnn_out = self.cnn(x)  # shape: (batch, cnn_channels, seq_len)
        
        # transpose for LSTM: (batch, seq_len, features)
        lstm_in = cnn_out.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(lstm_in)
        
        # take last hidden state as features
        features = lstm_out[:, -1, :]  # shape: (batch, hidden_size)
        
        # classification
        logits = self.classifier(features)
        
        return logits, features
