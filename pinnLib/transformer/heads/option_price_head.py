import torch
import torch.nn as nn
from pinnLib.transformer.core.head_base import BaseHead

class OptionPriceHead(BaseHead):
    """
    A simple regression head to map the transformer's encoded representation
    to a single option price output per example.
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        # We assume sequence reduction (e.g., mean pooling)
        x = x.mean(dim=1)
        return self.model(x).squeeze(-1)  # Output: (batch_size,)
