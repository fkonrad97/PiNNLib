from abc import ABC, abstractmethod
import torch.nn as nn

class BaseAttention(nn.Module, ABC):
    @abstractmethod
    def forward(self, Q, K, V, mask=None):
        pass

