import torch
from torch import nn
from torch.optim import AdamW
from pinnLib.core.pinn_network import StandardFeedForwardNN
from pinnLib.core.pinn_loss_builder import StandardPINNLossBuilder
from pinnLib.core.pinn_trainer import StandardPINNTrainer
from pinnLib.models.black_scholes_model.black_scholes_pde import BlackScholesPDE
from pinnLib.models.black_scholes_model.black_scholes_monte_carlo_benchmark import BlackScholesMonteCarloBenchmark
from pinnLib.models.black_scholes_model.black_scholes_pinn_vs_monte_carlo_eval import BlackScholesPINNvsMonteCarloEvaluator

# ==== CONFIGURATION ====
dim = 2
strike = 100.0
weights = torch.ones(dim) / dim
sigma = 0.2
r = 0.05
T = 1.0
S_max = 200.0
rho = 0.5 * torch.ones(dim, dim) + 0.5 * torch.eye(dim)

device = 'cuda'
model_path = f'bs_pinn_{dim}d.pt'

# ==== MODEL ====
model = StandardFeedForwardNN(
    input_dim=dim + 1,
    hidden_dim=128,
    output_dim=1,
    depth=5,
    activation=nn.Tanh()
).to(device)

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
    device=device
)

# ==== LOSS & OPTIMIZER ====
loss_builder = StandardPINNLossBuilder()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ==== TRAIN ====
trainer = StandardPINNTrainer(
    model=model,
    pde=pde,
    loss_builder=loss_builder,
    optimizer=optimizer,
    device=device
)

trainer.train(
    epochs=10000,
    print_every=1000,
    N_f=3000,
    Nr=600,
    Nb=600
)

torch.save(model.state_dict(), model_path)
print(f"Trained model saved to {model_path}")

# ==== MONTE CARLO BENCHMARK ====
benchmark = BlackScholesMonteCarloBenchmark(
    strike=strike,
    r=r,
    sigma=sigma,
    T=T,
    weights=weights,
    n_paths=100_000,
    device=device
)

# ==== CREATE GRID OF EVAL INPUTS ====
s1 = torch.linspace(80, 120, 50)
s2 = torch.linspace(80, 120, 50)
S1, S2 = torch.meshgrid(s1, s2, indexing='ij')
S0_grid = torch.stack([S1, S2], dim=-1).reshape(-1, dim).to(device)

# ==== EVALUATE ====
evaluator = BlackScholesPINNvsMonteCarloEvaluator(
    model=model,
    pde=pde,
    benchmark=benchmark,
    device=device
)

results = evaluator.evaluate(S0_grid)
evaluator.summarize_performance(results)
evaluator.plot_comparison(results)
