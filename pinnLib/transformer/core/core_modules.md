# Documentation: Transformer Core Interfaces

This document describes the foundational abstract components that define the modular architecture of your Transformer-based quant modeling framework.

These core interfaces ensure modularity, flexibility, and extensibility, allowing different components (attention mechanisms, blocks, encoders, etc.) to be easily plugged into different models for a variety of financial tasks.

---

## 1. `BaseAttention` (`core/attention_base.py`)

### Purpose:
Defines the interface for any attention mechanism used within a Transformer block. Examples include `ScaledDotProductAttention`, cross-attention, or future custom attention layers.

### Method:
```python
forward(Q, K, V, mask=None)
```

### Expected Input Shapes:
- Q, K, V: Tensors of shape `(batch_size, num_heads, seq_len, embed_dim)`
- mask: Optional mask tensor of shape `(batch_size, seq_len)` or `(batch_size, 1, seq_len, seq_len)`

### Notes:
- Must output attention scores and contextualized (batch_size, num_heads, seq_len, embed_dim) representations
- Enables modularity in `MultiHeadAttention`

---

## 2. `BaseBlock` (`core/block_base.py`)

### Purpose:
Defines the interface for a single block in a Transformer encoder or decoder. Each block typically consists of:
- Multi-head attention
- Feedforward network
- LayerNorm + residual connections

### Method:
```python
forward(x, mask=None)
```

### Notes:
- Used in `TransformerEncoder` as the core unit
- Can be extended to `TransformerBlock`, `PerceiverBlock`, `PINNConditionedBlock`, etc.

---

## 3. `BaseEncoder` (`core/encoder_base.py`)

### Purpose:
Defines the interface for an encoder stack — a sequence of Transformer blocks that encode input features into rich representations.

### Method:
```python
forward(x, mask=None)
```

### Input:
- `x`: Embedded input, shape `(batch_size, seq_len, embed_dim)`

### Output:
- Same shape, but contextually transformed

---

## 4. `BaseDecoder` (`core/decoder_base.py`)

### Purpose:
Defines the interface for decoder stacks used in autoregressive or sequence-to-sequence settings.

### Method:
```python
forward(x, encoder_output, mask=None)
```

### Inputs:
- `x`: Current decoder input
- `encoder_output`: Context from encoder
- `mask`: Optional causal or padding mask

---

## 5. `BaseEmbedding` (`core/embeddings_base.py`)

### Purpose:
Provides an interface for learnable input encoders. These modules convert raw input features (e.g., `[K, T, S, σ]`) into vectors in embedding space.

### Method:
```python
forward(x)
```

### Examples:
- `TabularEmbedding`: Projects financial features
- `PositionalEncoding`: Adds structural info (like maturity position, strike ordering)

---

## 6. `BaseHead` (`core/head_base.py`)

### Purpose:
Defines the interface for the final module that maps Transformer output to task-specific predictions.

### Method:
```python
forward(x)
```

### Use Cases:
- `OptionPriceHead`: Outputs scalar price from mean-pooled embedding
- `VolSurfaceHead`: Outputs a `num_strikes × num_maturities` IV surface

---

## Summary

These abstract base classes form the backbone of a fully modular Transformer modeling framework. By enforcing consistent interfaces, they make it easy to:

- Mix and match components
- Add new research ideas without breaking old code
- Keep each layer testable and independently improvable

Together, they enable a **flexible, extensible, and research-grade architecture** for quantitative modeling tasks like:
- Option pricing
- Volatility surface regression
- PINN-constrained PDE solving
- Multi-asset modeling
