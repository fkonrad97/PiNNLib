import torch
import torch.nn as nn

class VolSurfaceHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_strikes: int, num_maturities: int):
        super().__init__()
        self.num_strikes = num_strikes
        self.num_maturities = num_maturities

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.size()
        expected_seq_len = self.num_strikes * self.num_maturities

        assert seq_len == expected_seq_len, f"Expected seq_len={expected_seq_len}, but got {seq_len}"

        # Predict volatility at each point
        x = self.ff(x)  # → (batch_size, seq_len, 1)
        x = x.squeeze(-1)  # → (batch_size, seq_len)

        # Reshape into (B, strikes, maturities)
        x = x.view(batch_size, self.num_strikes, self.num_maturities)
        return x
