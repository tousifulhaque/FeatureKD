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
        x = F.relu(self.bn4(self.conv4(x)))
        
        
        # Global average pooling
        x = x.mean(dim=2)  # [B, C]
        
        x = self.dropout(x)
        logits = self.fc(x)
        
        return logits, x  # logits and features