"""
# Transformer Attention Modules

This document explains the concrete attention mechanisms implemented in the project. These are built on top of the `BaseAttention` interface defined in the `core/` directory.

---

## 1. `ScaledDotProductAttention`

### Purpose:
This is the core attention function used in the original Transformer architecture. It computes attention weights using scaled dot-product similarity between queries and keys.

### Equation:
Given query `Q`, key `K`, and value `V`:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

### Inputs:
- Q, K, V: shape `(batch_size, num_heads, seq_len, d_k)`
- mask: optional boolean mask

### Outputs:
- Output tensor of shape `(batch_size, num_heads, seq_len, d_k)`
- Attention weights of shape `(batch_size, num_heads, seq_len, seq_len)`

### Notes:
- Implements dropout after softmax
- Fully compatible with multi-head attention wrappers

---

## 2. `MultiHeadAttention` (`attention/multi_head_attention.py`)

### Purpose:
Implements multiple parallel attention heads. Each head applies an independent attention function (e.g., `ScaledDotProductAttention`) and the results are concatenated and projected back to the original embedding size.

### Components:
- Linear projection layers for Q, K, V
- Independent attention mechanism per head
- Final projection layer to combine heads

### Inputs:
- Q, K, V: shape `(batch_size, seq_len, embed_dim)`
- `embed_dim` must be divisible by `num_heads`

### Internal Flow:
1. Project Q, K, V into `num_heads` subspaces of dimension `d_k = embed_dim / num_heads`
2. Compute attention for each head independently
3. Concatenate results and project to output dimension

### Output:
- Output tensor: `(batch_size, seq_len, embed_dim)`
- List of attention weights per head

---

## Design Philosophy

- `MultiHeadAttention` is pluggable: you can pass any `BaseAttention` subclass to define the head logic
- Attention heads are modular: you can implement learned bias attention, causal attention, or PINN-guided attention as drop-in replacements

---
"""