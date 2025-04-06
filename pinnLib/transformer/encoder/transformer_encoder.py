# transformer/core/encoder.py

import torch
import torch.nn as nn
<<<<<<< HEAD
from pinnLib.transformer.blocks.transformer_block import TransformerBlock
=======
from pinnLib.transformer.core.transformer_block import TransformerBlock
>>>>>>> 63ab059a96d2942b5932c5099df9b6851269ae39
from pinnLib.transformer.core.encoder_base import BaseEncoder

class TransformerEncoder(BaseEncoder):
    """
    Stacks multiple Transformer blocks.
<<<<<<< HEAD
    Accepts a custom block class that implements the BaseBlock interface.
=======
    Suitable for encoding input sequences or tokenized financial data.
>>>>>>> 63ab059a96d2942b5932c5099df9b6851269ae39
    """
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 ff_hidden_dim: int, 
                 num_layers: int = 6,
<<<<<<< HEAD
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
=======
                 dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
>>>>>>> 63ab059a96d2942b5932c5099df9b6851269ae39
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
