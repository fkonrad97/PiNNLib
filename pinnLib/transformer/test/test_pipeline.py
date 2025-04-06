import torch
from pinnLib.transformer.preprocessor.option_input_processor import OptionInputProcessor
from pinnLib.transformer.encoder.transformer_encoder import TransformerEncoder
from pinnLib.transformer.heads.option_price_head import OptionPriceHead

# Synthetic input: batch of 8 samples, each with 10 time steps (or grid points), and 4 features (e.g., strike, maturity, spot, vol)
batch_size = 8
seq_len = 10
input_dim = 4
embed_dim = 64
x = torch.randn(batch_size, seq_len, input_dim)

# Initialize input processor
input_processor = OptionInputProcessor(
    input_dim=input_dim,
    embed_dim=embed_dim,
    use_mlp=True,
    use_positional=True,
    max_len=seq_len
)

# Initialize encoder
encoder = TransformerEncoder(
    embed_dim=embed_dim,
    num_heads=4,
    ff_hidden_dim=128,
    num_layers=2,
    dropout=0.1
)

# Initialize head
head = OptionPriceHead(embed_dim=embed_dim, hidden_dim=64)

# Forward pass
embedded = input_processor(x)         # shape: (batch, seq_len, embed_dim)
encoded = encoder(embedded)          # shape: (batch, seq_len, embed_dim)
predictions = head(encoded)          # shape: (batch,)

print("Input shape:", x.shape)
print("Embedded shape:", embedded.shape)
print("Encoded shape:", encoded.shape)
print("Predicted option prices:", predictions)