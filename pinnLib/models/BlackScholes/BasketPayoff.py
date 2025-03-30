from pinnLib.PiNNResultCondition import ResultCondition
import torch

class BasketCallPayoff(ResultCondition):
    def __init__(self, weights: torch.Tensor, strike: float):
        self.weights = weights  # shape: (D,)
        self.strike = strike

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(X.device)  # <-- match device
        basket = X @ weights
        return torch.clamp(basket - self.strike, min=0.0).unsqueeze(-1)

