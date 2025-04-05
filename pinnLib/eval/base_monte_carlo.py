from abc import ABC, abstractmethod
import torch
from pinnLib.eval.base_method_benchmark import BaseMethodBenchmark


class BaseMonteCarloBenchmark(BaseMethodBenchmark, ABC):
    def __init__(self, r: float, T: float, n_paths: int = 100_000):
        self.r = r
        self.T = T
        self.n_paths = n_paths

    @abstractmethod
    def simulate_paths(self, S0: torch.Tensor) -> torch.Tensor:
        """
        Given initial asset states S0 of shape (N, d),
        return simulated terminal values S_T of shape (N, n_paths, d)
        """
        pass

    @abstractmethod
    def payoff(self, S_T: torch.Tensor) -> torch.Tensor:
        """
        Computes payoff given terminal values S_T (N, n_paths, d) → (N, n_paths)
        """
        pass

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns Monte Carlo estimated price: E[e^{-rT} * payoff(S_T)]
        Input X: shape (N, d) — current asset prices
        """
        S_T = self.simulate_paths(X)  # (N, n_paths, d)
        payoff_vals = self.payoff(S_T)  # (N, n_paths)
        discounted = torch.exp(torch.tensor(-self.r * self.T, device=payoff_vals.device)) * payoff_vals
        return discounted.mean(dim=1, keepdim=True)  # (N, 1)

    def compute_error(self, model_output: torch.Tensor, ground_truth: torch.Tensor) -> float:
        return torch.sqrt(torch.mean((model_output - ground_truth) ** 2)).item()
