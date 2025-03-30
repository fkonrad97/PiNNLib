import torch.nn as nn
from abc import ABC, abstractmethod

class BaseBlock(nn.Module, ABC):
    @abstractmethod
    def forward(self, x, mask=None):
        pass


