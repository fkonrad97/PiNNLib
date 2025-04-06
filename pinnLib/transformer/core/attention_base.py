from abc import ABC, abstractmethod
import torch.nn as nn

class BaseAttention(nn.Module, ABC):
    @abstractmethod
    def forward(self, Q, K, V, mask=None):
<<<<<<< HEAD
        pass
=======
        pass

>>>>>>> 63ab059a96d2942b5932c5099df9b6851269ae39
