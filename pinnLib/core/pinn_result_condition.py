from abc import ABC, abstractmethod
import torch

class ResultCondition(ABC):
    """
    Represents the known solution at a specific time surface:
    - Terminal condition (e.g., payoff at t = T for backward PDEs like in finance)
    - Initial condition (e.g., temperature at t = 0 in physics)

    Allows for domain-agnostic use in PINN PDE solvers.
    """

    @abstractmethod
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the condition at given spatial inputs X.

        :param X: Tensor of shape (N, D), where D is the number of spatial dims
        :return: Tensor of shape (N, 1) with known values at this surface
        """
        pass