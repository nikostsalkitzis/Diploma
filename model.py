import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1️⃣ EnsembleLinear — required by trainer.py
# ============================================================
class EnsembleLinear(nn.Module):
    """
    Linear layer for deep ensembles (one weight/bias per ensemble member).
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int

    def __init__(self, in_features: int, out_features: int, ensemble_size: int,
                 weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.lin_w = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.lin_b = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (ensemble_size, batch, in_features)
        w_times_x = torch.bmm(x, self.lin_w)
        y = torch.add(w_times_x, self.lin_b[:, None, :])
        return y

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lin_w, a=math.sqrt(5))
        if self.lin_b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.lin_w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.lin_b, -bound, bound)


# ============================================================
# 2️⃣ CNN + LSTM Encoder — replaces TransformerClassifier
# ============================================================
class CNNLSTMClassifier(nn.Module):
    """
    CNN + LSTM hybrid for time-series feature extraction and regression.
    Outputs:
        - features: latent vector for ensemble heads
        - preds: [sin_t, cos_t] regression output
    """

    def __init__(self, args):
        super().__init__()

        args_defaults = dict(
            input_features=10,
            seq_len=512,
            hidden_dim=128,
            num_layers=1,
            dropout=0.2,
            d_model=64,
            device='cuda'
        )
        for key, value in args_defaults.items():
            setattr(self, key, args[key] if key in args and args[key] is not None else value)

        # CNN feature extractor
        self.conv1 = nn.Conv1d(self.input_features, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout_cnn = nn.Dropout(self.dropout)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0
        )

        # Projection and regression head
        self.fc_proj = nn.Linear(self.hidden_dim, self.d_model)
        self.predictor = nn.Linear(self.d_model, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, src):
        """
        src shape: (batch, input_features, seq_len)
        returns:
            features: (batch, d_model)
            preds: (batch, 2)
        """
        # CNN layers
        x = F.relu(self.bn1(self.conv1(src)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout_cnn(x)

        # LSTM
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :]  # last hidden state

        # Projection + prediction
        features = self.fc_proj(features)
        preds = self.predictor(features)

        return features, preds
