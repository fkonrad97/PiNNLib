import torch.nn as nn
from abc import ABC, abstractmethod

class BaseHead(nn.Module, ABC):
    """
    Abstract base class for task heads.
    Converts encoded representations into task-specific outputs.
    Examples: option price, IV surface, volatility parameters, etc.
    """
    @abstractmethod
    def forward(self, x):
        pass