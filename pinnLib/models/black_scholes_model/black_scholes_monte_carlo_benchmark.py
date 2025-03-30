import torch
from pinnLib.eval.base_monte_carlo import BaseMonteCarloBenchmark


class BlackScholesMonteCarloBenchmark(BaseMonteCarloBenchmark):
    def __init__(self,
                 strike: float,
                 r: float,
                 sigma: float,
                 T: float,
                 n_paths: int = 100_000,
                 device: str = 'cpu',
                 weights: torch.Tensor = None):
        super().__init__(r=r, T=T, n_paths=n_paths)
        self.strike = strike
        self.sigma = sigma
        self.device = device
        self.weights = weights  # Optional: for basket options

    def simulate_paths(self, S0: torch.Tensor) -> torch.Tensor:
        """
        Simulate geometric Brownian motion paths from S0.
        Returns S_T of shape (N, n_paths, d)
        """
        N, d = S0.shape
        dt = self.T
        S0 = S0.to(self.device)
        Z = torch.randn(N, self.n_paths, d, device=self.device)
        # Closed-form solution of GBM at time T: (Since we pricing European-style, so the payoff only depends on the value at maturity, not the path)
        S_T = S0.unsqueeze(1) * torch.exp(
            (self.r - 0.5 * self.sigma ** 2) * dt +
            self.sigma * torch.sqrt(torch.tensor(dt)) * Z
        )
        return S_T  # shape (N, n_paths, d)

    def payoff(self, S_T: torch.Tensor) -> torch.Tensor:
        """
        Default: basket call option payoff = max(w·S_T - K, 0)
        """
        if self.weights is not None:
            basket = torch.einsum('bnd,d->bn', S_T, self.weights.to(self.device))
        else:
            basket = S_T.mean(dim=-1)  # equally weighted basket
        return torch.clamp(basket - self.strike, min=0.0)