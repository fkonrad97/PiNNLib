# transformer/core/encoder.py

import torch
import torch.nn as nn
from pinnLib.transformer.blocks.transformer_block import TransformerBlock
from pinnLib.transformer.core.encoder_base import BaseEncoder

class TransformerEncoder(BaseEncoder):
    """
    Stacks multiple Transformer blocks.
    Accepts a custom block class that implements the BaseBlock interface.
    """
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 ff_hidden_dim: int, 
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 block_cls: type = TransformerBlock,
                 **block_kwargs):
        super().__init__()

        self.layers = nn.ModuleList([
            block_cls(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout,
                **block_kwargs
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
