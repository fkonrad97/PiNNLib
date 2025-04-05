import torch
from torch import nn
from pinnLib.core.pinn_base_pde import BasePDE
from pinnLib.models.black_scholes_model.option_payoff import BasketCallPayoff
from torch.autograd.functional import jacobian
from torch.func import jacrev, vmap

class BlackScholesPDE(BasePDE):
    def __init__(self,
                 dim: int,
                 strike: float,
                 weights: torch.Tensor,
                 sigma: float = 0.2,
                 rho: torch.Tensor = None,
                 r: float = 0.05,
                 T: float = 1.0,
                 S_max: float = 200.0,
                 device: str = 'cpu'):

        super().__init__(dim=dim, device=device)
        self.strike = strike
        self.weights = weights.to(device)
        self.sigma = sigma
        self.r = r
        self.T = T
        self.S_max = S_max
        self.payoff_function = BasketCallPayoff(weights, strike)
        if rho is None:
            self.rho = torch.eye(dim).to(device)  # default: identity (no correlation)
        else:
            self.rho = rho.to(device)

    def pde_residual(self, model: nn.Module, X: torch.Tensor) -> torch.Tensor:
        X = X.clone().detach().requires_grad_(True).to(self.device)
        t = X[:, 0:1]
        S = X[:, 1:]  # shape (N, d)

        def f(x_):
            return model(x_.unsqueeze(0)).squeeze()

        V = model(X)  # (N, 1)

        grad = torch.autograd.grad(
            outputs=V,
            inputs=X,
            grad_outputs=torch.ones_like(V),
            create_graph=True
        )[0]  # (N, d + 1)

        dV_dt = grad[:, 0:1]   # (N, 1)
        dV_dS = grad[:, 1:]    # (N, d)

        H = vmap(jacrev(jacrev(f)))(X)[:, 1:, 1:]  # (N, d, d)

        S_sigma = S * self.sigma  # (N, d)
        SS_rho = torch.einsum('bi,bj->bij', S_sigma, S_sigma) * self.rho  # (N, d, d)
        diffusion = 0.5 * torch.einsum('bij,bij->b', SS_rho, H).unsqueeze(1)  # (N, 1)

        drift = self.r * torch.sum(S * dV_dS, dim=1, keepdim=True)  # (N, 1)
        discount = -self.r * V  # (N, 1)

        residual = dV_dt + drift + diffusion + discount  # (N, 1)
        return residual

    def sample_collocation_points(self, N_f: int) -> torch.Tensor:
        S = torch.rand(N_f, self.dim) * self.S_max
        t = torch.rand(N_f, 1) * self.T
        return torch.cat([S, t], dim=1).to(self.device)

    def sample_result_points(self, Nr: int) -> torch.Tensor:
        S = torch.rand(Nr, self.dim) * self.S_max
        t = torch.ones(Nr, 1) * self.T
        return torch.cat([S, t], dim=1).to(self.device)

    def sample_boundary_points(self, Nb: int) -> torch.Tensor:
        """
        Samples boundary at S = 0 or S = S_max for any dimension.
        """
        points = []
        for i in range(self.dim):
            for val in [0.0, self.S_max]:
                S = torch.rand(Nb, self.dim) * self.S_max
                S[:, i] = val  # fix one asset at boundary
                t = torch.rand(Nb, 1) * self.T
                pt = torch.cat([S, t], dim=1)
                points.append(pt)
        return torch.cat(points, dim=0).to(self.device)

    def boundary_condition(self, X_b: torch.Tensor) -> torch.Tensor:
        # At boundaries, assume option value = max(weighted sum - strike, 0)
        S = X_b[:, :-1]
        return self.payoff_function(S)

    def result_condition(self, X_res: torch.Tensor) -> torch.Tensor:
        S = X_res[:, :-1]
        return self.payoff_function(S)
    
    def initial_condition(self, X0: torch.Tensor):
        raise NotImplementedError("Black-Scholes PDE is defined with terminal condition, not initial.")

    def sample_initial_points(self, N0: int):
        raise NotImplementedError("Black-Scholes PDE is defined with terminal condition, not initial.")

