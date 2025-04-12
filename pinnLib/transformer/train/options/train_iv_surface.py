import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pinnLib.transformer.model.option_models.iv_surface_transformer_model import IVSurfaceTransformerModel
from pinnLib.transformer.data.synthetic.iv_surface_dataset import IVSurfaceDataset

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_dim = 64
num_heads = 4
ff_hidden_dim = 128
num_layers = 4
head_hidden_dim = 64
num_strikes = 10
num_maturities = 10
input_dim = 2
batch_size = 1  # One surface per batch
epochs = 20
lr = 1e-3

# Dataset & Loader
dataset = IVSurfaceDataset(
    num_strikes=num_strikes,
    num_maturities=num_maturities
)
dataloader = DataLoader(dataset, batch_size=num_strikes * num_maturities)

# Model
model = IVSurfaceTransformerModel(
    input_dim=input_dim,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_hidden_dim=ff_hidden_dim,
    num_layers=num_layers,
    num_strikes=num_strikes,
    num_maturities=num_maturities,
    head_hidden_dim=head_hidden_dim
).to(device)

# Training Setup
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training Loop
model.train()
for epoch in range(1, epochs + 1):
    for X, y in dataloader:
        X = X.to(device)  # [B, seq_len, input_dim]
        y = y.to(device)

        X = X.unsqueeze(0)  # → (1, seq_len, input_dim)
        y = y.view(1, num_strikes, num_maturities)

        preds = model(X)  # → (1, num_strikes, num_maturities)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}/{epochs} - MSE Loss: {loss.item():.6f}")

torch.save(model.state_dict(), "iv_surface_model.pth")