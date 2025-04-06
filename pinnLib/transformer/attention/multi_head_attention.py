import torch
import torch.nn as nn
import torch.nn.functional as F
from pinnLib.transformer.core.attention_base import BaseAttention
<<<<<<< HEAD
from pinnLib.transformer.attention.types.scaled_dot_product_attention import ScaledDotProductAttention
=======
from pinnLib.transformer.attention.scaled_dot_product_attention import ScaledDotProductAttention
>>>>>>> 63ab059a96d2942b5932c5099df9b6851269ae39

class MultiHeadAttention(BaseAttention):
    def __init__(self, embed_dim: int, num_heads: int, attention_cls=ScaledDotProductAttention, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)

        # Pluggable attention mechanism, one per head
        self.attention_heads = nn.ModuleList([
            attention_cls(dropout) for _ in range(num_heads)
        ])

        self.linear_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections
        Q = self.linear_q(Q)
        K = self.linear_k(K)
        V = self.linear_v(V)

        # Reshape: (B, Seq, D) → (B, num_heads, Seq, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply each head separately
        outputs = []
        attn_weights = []
        for i in range(self.num_heads):
            out, w = self.attention_heads[i](Q[:, i], K[:, i], V[:, i], mask)
            outputs.append(out)
            attn_weights.append(w)

        # Concatenate heads: (B, Seq, embed_dim)
        concat = torch.cat(outputs, dim=-1)
        output = self.linear_out(concat)

        return output, attn_weights
