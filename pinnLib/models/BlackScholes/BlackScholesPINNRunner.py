import torch
from torch import nn
from torch.optim import AdamW

from pinnLib.PiNNetwork import StandardFeedForwardNN, BasePINN
from pinnLib.models.BlackScholes.BlackScholesPDE import BlackScholesPDE
from pinnLib.PiNNBaseLossBuilder import StandardPINNLossBuilder
from pinnLib.PiNNTrainer import StandardPINNTrainer

# ==== CONFIGURATION ====
dim = 5
strike = 100.0
weights = torch.ones(dim) / dim  # Uniform basket
sigma = 0.2
r = 0.05
T = 1.0
S_max = 200.0
rho = 0.5 * torch.ones(dim, dim) + 0.5 * torch.eye(dim)  # 0.5 correlation off-diagonal

# ==== MODEL ====
model: BasePINN = StandardFeedForwardNN(
    input_dim=dim + 1,  # +1 for time
    hidden_dim=128,
    output_dim=1,
    depth=5,
    activation=nn.Tanh()
)

# ==== PDE ====
pde = BlackScholesPDE(
    dim=dim,
    strike=strike,
    weights=weights,
    sigma=sigma,
    rho=rho,
    r=r,
    T=T,
    S_max=S_max,
    device='cuda'
)

# ==== LOSS & OPTIMIZER ====
loss_builder = StandardPINNLossBuilder(
    pde_weight=1.0,
    result_weight=1.0,
    boundary_weight=1.0
)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ==== TRAINER ====
trainer = StandardPINNTrainer(
    model=model,
    pde=pde,
    loss_builder=loss_builder,
    optimizer=optimizer,
    device='cuda'
)

# ==== TRAIN ====
trainer.train(
    epochs=3000,
    print_every=100,
    N_f=3000,
    Nr=600,
    Nb=600
)
