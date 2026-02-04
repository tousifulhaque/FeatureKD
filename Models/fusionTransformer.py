"""Single-stream IMU transformer."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SqueezeExcitation(nn.Module):

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.SiLU(),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=2)
        scale = self.fc(scale).unsqueeze(2)
        return x * scale


class TemporalAttentionPooling(nn.Module):

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class SingleStreamTransformer(nn.Module):

    def __init__(
        self,
        imu_frames: int = 128,
        imu_channels: int = 6,
        num_heads: int = 4,
        num_layers: int = 2,
        embed_dim: int = 64,
        dropout: float = 0.5,
        use_se: bool = False,
        use_temporal_attention: bool = False,
        use_pos_encoding: bool = True,
        se_reduction: int = 4,
        activation: str = 'silu',
        norm_first: bool = True,

    ):
        super().__init__()

        self.imu_frames = imu_frames
        self.imu_channels = imu_channels
        self.embed_dim = embed_dim
        self.use_se = use_se
        self.use_temporal_attention = use_temporal_attention
        self.use_pos_encoding = use_pos_encoding

        act_fn = nn.SiLU() if activation.lower() == 'silu' else nn.ReLU()
        self.input_proj = nn.Sequential(
            nn.Conv1d(imu_channels, embed_dim, kernel_size=9, padding='same'),
            nn.BatchNorm1d(embed_dim),
            act_fn,
            nn.Dropout(dropout * 0.3),
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            norm_first=norm_first,
            batch_first=False,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)


        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, 1)

        self.last_attention_weights = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        acc_data: torch.Tensor,
        skl_data: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = acc_data
        x = rearrange(x, 'b t c -> b c t')
        x = self.input_proj(x)
        x = rearrange(x, 'b c t -> b t c')
        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')
        features = x

        x = x.mean(dim=1)

        x = self.dropout(x)
        logits = self.output(x)

        return logits, features
