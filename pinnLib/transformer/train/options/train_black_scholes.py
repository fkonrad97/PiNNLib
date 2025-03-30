import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from pinnLib.transformer.data.synthetic.black_scholes_dataset import BlackScholesDataset
from pinnLib.transformer.model.option_pricing.option_pricing_transformer_model import OptionPricingTransformerModel

# ---- Config ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 4
embed_dim = 64
num_heads = 4
ff_hidden_dim = 128
num_layers = 2
dropout = 0.1
batch_size = 64
epochs = 10
lr = 1e-3

# ---- Data ----
dataset = BlackScholesDataset(n_samples=10000, device=device)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---- Model ----
model = OptionPricingTransformerModel(
    input_dim=input_dim,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_hidden_dim=ff_hidden_dim,
    num_layers=num_layers,
    dropout=dropout
).to(device)

# ---- Optimizer & Loss ----
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# ---- Training Loop ----
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X, y in loader:
        optimizer.zero_grad()
        preds = model(X).view(-1)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{epochs} - MSE Loss: {avg_loss:.6f}")
