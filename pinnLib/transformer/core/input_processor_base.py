import torch.nn as nn
from abc import ABC, abstractmethod

class BaseInputProcessor(nn.Module, ABC):
    """
    Abstract interface for any input processing pipeline that
    transforms raw inputs into embeddings suitable for the transformer.
    """
    @abstractmethod
    def forward(self, x):
        pass