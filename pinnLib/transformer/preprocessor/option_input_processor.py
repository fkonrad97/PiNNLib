import torch.nn as nn
from pinnLib.transformer.core.input_processor_base import BaseInputProcessor
from pinnLib.transformer.embeddings.tabular_embeddings import TabularEmbedding
from pinnLib.transformer.embeddings.positional_encoding import PositionalEncoding

class OptionInputProcessor(BaseInputProcessor):
    """
    A concrete input processor for option pricing tasks.
    Combines tabular feature embedding with optional positional encoding.
    """
    def __init__(self, input_dim: int, embed_dim: int, use_mlp: bool = False, use_positional: bool = True, max_len: int = 500):
        super().__init__()
        self.tabular = TabularEmbedding(input_dim=input_dim, embed_dim=embed_dim, use_mlp=use_mlp)
        self.use_positional = use_positional
        self.positional = PositionalEncoding(embed_dim=embed_dim, max_len=max_len) if use_positional else None

    def forward(self, x):
        # x: (B, seq_len, input_dim) or (B, input_dim)
        x = self.tabular(x)  # → (B, embed_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # → (B, 1, D)
        if self.use_positional:
            x = self.positional(x)  # Applies PE correctly
        return x

