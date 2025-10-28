import math
from typing import Tuple, Union, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor


# -------------------------------------------------------
# Positional Encoding (sin/cos)
# -------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Classic sine/cosine positional encoding from "Attention is All You Need".
    We keep seq_first convention (seq_len, batch, dim) to match nn.TransformerEncoder.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe: (max_len, 1, d_model)
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # odd dims

        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (seq_len, batch, d_model)
        returns: same shape, with positional encoding added
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)


# -------------------------------------------------------
# EnsembleLinear (unchanged)
# -------------------------------------------------------
class EnsembleLinear(nn.Module):
    """
    Same as your original code.
    This layer represents `ensemble_size` separate linear heads:
    - weight: (ensemble_size, in_features, out_features)
    - bias:   (ensemble_size, out_features)
    Forward:
    - x: (ensemble_size, batch, in_features)
    - out: (ensemble_size, batch, out_features)
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        weight_decay: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight_decay = weight_decay

        # one weight matrix per ensemble member
        self.lin_w = nn.Parameter(
            torch.Tensor(ensemble_size, in_features, out_features)
        )

        if bias:
            self.lin_b = nn.Parameter(
                torch.Tensor(ensemble_size, out_features)
            )
        else:
            self.register_parameter("lin_b", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lin_w, a=math.sqrt(5))
        if self.lin_b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.lin_w)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.lin_b, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (ensemble_size, batch, in_features)
        return: (ensemble_size, batch, out_features)
        """
        # bmm over last two dims:
        # (ensemble_size, batch, in_features) x (ensemble_size, in_features, out_features)
        w_times_x = torch.bmm(x, self.lin_w)  # (ensemble_size, batch, out_features)

        if self.lin_b is not None:
            # Add bias (broadcast batch axis)
            y = torch.add(w_times_x, self.lin_b[:, None, :])
        else:
            y = w_times_x
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.lin_b is not None}"
        )


# -------------------------------------------------------
# TransformerHeartPredictor for raw windows
# -------------------------------------------------------
class TransformerHeartPredictor(nn.Module):
    """
    Model for raw physiological windows.

    INPUT SHAPE (per batch sample):
        x: (batch, C, T_raw)
           - C is #channels, e.g. 10:
             [accX,accY,accZ, gyrX,gyrY,gyrZ, heartRate, rRInterval, sin_t, cos_t]
           - T_raw can be large (e.g. 36k samples for 2 hours @ 5 Hz)

    PIPELINE:
        1. patch_embed : Conv1d(C -> d_model, kernel_size=patch_stride, stride=patch_stride)
           - This "patchifies" the long time series into ~T_raw/patch_stride tokens.
           - Output: (batch, d_model, T_tokens)

        2. transformer encoder over tokens:
           - We permute to (T_tokens, batch, d_model)
           - Add positional encoding
           - Run n layers of self-attention + FFN

        3. temporal pooling:
           - AdaptiveAvgPool1d(1) over tokens
           - Resulting embedding: (batch, d_model)

        4. regression head:
           - Linear(d_model -> output_dim=5)
             predicts [HR_mean, RR_mean, RMSSD, SDNN, HF_power]
             for this window

    RETURNS:
        features:    (batch, d_model)  latent embedding of the window
        heart_preds: (batch, 5)        regression output

    This signature matches what your Trainer expects:
        features, preds = model(x)
    """

    def __init__(self, args: Union[Dict[str, Any], "argparse.Namespace"]) -> None:
        super().__init__()

        # ---------
        # pull args safely with defaults
        # ---------
        def _get(key: str, default):
            if isinstance(args, dict):
                return args.get(key, default)
            return getattr(args, key, default)

        # core dimensions
        self.input_features = _get("input_features", 10)          # channels in raw input after stacking
        self.output_dim = _get("output_dim", 5)                    # we predict 5 heart/HRV targets
        self.d_model = _get("d_model", 64)                         # transformer emb dim
        self.n_encoder_layers = _get("n_encoder_layers", _get("nlayers", 2))
        self.n_head = _get("n_head", _get("nhead", 8))
        self.dim_feedforward_encoder = _get("dim_feedforward_encoder", 2048)

        # temporal compression / patching
        # how many raw timesteps per "token"
        self.patch_stride = _get("patch_stride", 25)
        # dropout configs
        self.dropout_encoder = _get("dropout_encoder", 0.2)
        self.dropout_pos_enc = _get("dropout_pos_enc", 0.1)

        # transformer defaults
        self.batch_first = _get("batch_first", False)

        # not strictly needed by forward, but Trainer may reference
        self.num_patients = _get("num_patients", 10)
        self.device_name = _get("device", "cuda")

        # ---------
        # patch embedding conv
        # ---------
        # This will turn (B, C, T_raw) -> (B, d_model, T_tokens)
        # where T_tokens ~ T_raw / patch_stride
        self.patch_embed = nn.Conv1d(
            in_channels=self.input_features,
            out_channels=self.d_model,
            kernel_size=self.patch_stride,
            stride=self.patch_stride,
            padding=0,
        )

        # ---------
        # positional encoding
        # ---------
        self.positional_encoding_layer = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout_pos_enc,
            max_len=50000  # enough for long sequences of patches
        )

        # ---------
        # transformer encoder
        # ---------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward_encoder,
            dropout=self.dropout_encoder,
            batch_first=self.batch_first,  # False means expect (seq_len, batch, dim)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.n_encoder_layers,
        )

        # ---------
        # pooling + prediction head
        # ---------
        self.adaptive = nn.AdaptiveAvgPool1d(1)     # pool over token dimension
        self.heart_head = nn.Linear(self.d_model, self.output_dim)

    # ---------------------------------------------------
    # Forward
    # ---------------------------------------------------
    def forward(self, src: Tensor) -> Tuple[Tensor, Tensor]:
        """
        src:
            TRAIN BATCH: (batch, C, T_raw)
            VAL BATCH:   (batch, C, T_raw)
            (For val/test in your Loader you often have shape (M, C, T);
             that's fine, it's still (batch, C, T).)

        Returns:
            features    : (batch, d_model)
            heart_preds : (batch, 5)
        """
        # 1. patchify raw input
        #    (B, C, T_raw) -> (B, d_model, T_tokens)
        x = self.patch_embed(src)

        # 2. rearrange for transformer (seq_len, batch, d_model)
        #    x was (B, d_model, T_tokens)
        #    we want (T_tokens, B, d_model)
        x = x.permute(2, 0, 1)  # (T_tokens, B, d_model)

        # 3. positional encoding
        x = self.positional_encoding_layer(x)  # still (T_tokens, B, d_model)

        # 4. transformer encoder
        x = self.encoder(x)  # (T_tokens, B, d_model)

        # 5. temporal pooling to fixed-size embedding
        #    convert back to (B, d_model, T_tokens) for pooling
        x_feat = x.permute(1, 2, 0).contiguous()  # (B, d_model, T_tokens)
        x_feat = self.adaptive(x_feat).squeeze(-1)  # (B, d_model)

        # 6. regression head to predict heart-related summary targets
        heart_preds = self.heart_head(x_feat)  # (B, 5)

        # features is the latent embedding; trainer uses this for ensemble
        return x_feat, heart_preds
