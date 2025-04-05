from abc import ABC, abstractmethod
import torch.nn as nn

class BasePINN(nn.Module, ABC):
    """
    Abstract base class for PINN models.
    Allows flexibility to implement different architectures (e.g. FNN, SIREN, Fourier, etc).
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass


class StandardFeedForwardNN(BasePINN):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        depth: int = 4,
        activation: nn.Module = nn.Tanh()
    ):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), activation]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)