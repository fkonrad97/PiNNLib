# transformer/core/positional_encoding.py

import torch
import torch.nn as nn
import math
from pinnLib.transformer.core.embeddings_base import BaseEmbedding

class PositionalEncoding(BaseEmbedding):
    """
    Adds sinusoidal positional encodings to input embeddings.
    Suitable for injecting order into non-recurrent Transformer models.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape: (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x
