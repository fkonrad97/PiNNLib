# Transformer Core Modules Overview

This module implements the core components of a Transformer architecture, tailored for flexible financial modeling. Below is a breakdown of the implemented classes and their purposes.

---

## 1. ScaledDotProductAttention

### Purpose:
Computes the attention scores between queries (Q) and keys (K), applies softmax, and uses the result to weight the values (V).

### Formula:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

### Key Steps:
- **Scale** by \(\sqrt{d_k}\) to stabilize softmax
- **Masking** (optional): prevent attending to padding tokens or invalid positions
- **Softmax** over keys for each query
- **Weighted sum** over V using softmax weights

---

## 2. MultiHeadAttention

### Purpose:
Executes multiple attention mechanisms (heads) in parallel. Each head learns different representations. Their outputs are concatenated and linearly transformed.

### Key Steps:
1. **Linear projections**: Q, K, V are projected into subspaces per head.
2. **Parallel attention**: Each head applies ScaledDotProductAttention independently.
3. **Concatenate** results from all heads.
4. **Final linear layer** to mix the outputs.

---

## Core Interfaces

### `BaseAttention`
- Abstract class for any attention mechanism
- Enables plugging in variants like cross-attention, PINN-conditioned attention, etc.

### `BaseEmbedding`
- Abstract interface for embedding modules

### `BaseEncoder`
- Interface for encoder blocks that transform input sequences contextually

### `BaseDecoder`
- Interface for decoder blocks (e.g., for sequence forecasting or autoregressive modeling)

### `BaseHead`
- Interface for downstream task heads (e.g., pricing, volatility prediction)

---

## Embedding Modules

### `TabularEmbedding`
- Embeds tabular financial parameters (e.g. strike, maturity, volatility)
- Supports linear projection or shallow MLP

### `PositionalEncoding`
- Adds sinusoidal positional encodings to represent temporal/spatial order

### `InputEncoder`
- Composite wrapper to combine `TabularEmbedding` + optional `PositionalEncoding`
- Not an encoder in the Transformer sense; rather a utility pipeline

---

## Encoding Modules

### `TransformerBlock`
- One self-contained block: MultiHeadAttention + Feedforward + LayerNorm + Residuals

### `TransformerEncoder`
- Stack of multiple TransformerBlocks
- Implements the `BaseEncoder` interface

---

## Decoding Modules

### `BaseDecoder`
- Interface for future decoders (e.g., for autoregressive or sequence-to-sequence tasks)
- Not yet implemented concretely

---

## Task Heads

### `OptionPriceHead`
- Takes the encoder output and produces a single scalar (option price)
- Uses mean pooling over sequence + MLP head
- Implements the `BaseHead` interface

---

## Architectural Diagram

```
         +----------------+
         |  InputEncoder  |     <- Tabular + Positional Encoding
         +----------------+
                  ↓
         +--------------------+
         | TransformerEncoder |     <- Multi-block stack
         +--------------------+
                  ↓
         +-------------------+
         |  OptionPriceHead  |     <- Predicts scalar option price
         +-------------------+
```

Optional future additions:
- Decoder (for forecasting or generation)
- VolSurfaceHead
- PINN module for physics constraints

---

This modular Transformer engine is the backbone of a future general-purpose quant modeling framework — adaptable to option pricing, IV surface generation, calibration, and more.
