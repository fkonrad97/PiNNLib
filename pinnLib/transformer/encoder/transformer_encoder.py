# transformer/core/encoder.py

import torch
import torch.nn as nn
from pinnLib.transformer.core.transformer_block import TransformerBlock
from pinnLib.transformer.core.encoder_base import BaseEncoder

class TransformerEncoder(BaseEncoder):
    """
    Stacks multiple Transformer blocks.
    Suitable for encoding input sequences or tokenized financial data.
    """
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 ff_hidden_dim: int, 
                 num_layers: int = 6,
                 dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
