from abc import ABC, abstractmethod
import torch.nn as nn

class BaseEmbedding(nn.Module, ABC):
    """
    Abstract base class for all embedding modules.
    Every embedding module must inherit from this and implement forward.
    """
    @abstractmethod
    def forward(self, x):
        pass