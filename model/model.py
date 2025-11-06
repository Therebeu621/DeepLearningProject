import torch.nn as nn
from .config import N_CLASSES

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((7,7))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64*7*7, n_classes))

    def forward(self, x):          # x: [B, n_mels, T]
        x = x.unsqueeze(1)         # -> [B,1,n_mels,T]
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)
