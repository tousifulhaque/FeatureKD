import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple 4-layer CNN for time series classification."""
    
    def __init__(self, in_channels=4, num_classes=1, dropout=0.3):
        super().__init__()
        
        # 4 convolutional layers with kernel size 3
        # Channel progression: 16 -> 32 -> 32 -> 64
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.conv4 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        
        
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder after CNN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, skl_data=None):
        # x: [B, T, C]
        x = x.permute(0, 2, 1)  # [B, C, T]

        # Layer 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)

        # Layer 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)

        # Layer 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 2)

        # Layer 4
        x = F.relu(self.bn4(self.conv4(x)))  # [B, 64, T']

        # Transformer encoder with skip connection
        # x_t = x.permute(0, 2, 1)           # [B, T', 64]
        # x_t = self.transformer(x_t)         # [B, T', 64]
        # x = x + x_t.permute(0, 2, 1)       # skip connection [B, 64, T']

        features = x.mean(dim=2)

        # Global average pooling
        x = features  # [B, 64]
        out = self.fc(x)
        return out, features  # logits and features