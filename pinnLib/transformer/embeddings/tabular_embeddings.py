# transformer/core/embeddings.py

import torch
import torch.nn as nn
from pinnLib.transformer.core.embeddings_base import BaseEmbedding

class TabularEmbedding(BaseEmbedding):
    """
    Embeds tabular inputs (e.g. strike, maturity, spot, volatility)
    using a linear projection or shallow MLP.
    """
    def __init__(self, input_dim: int, embed_dim: int, use_mlp: bool = False):
        super().__init__()
        self.use_mlp = use_mlp

        if use_mlp:
            self.project = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
        else:
            self.project = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.project(x)
