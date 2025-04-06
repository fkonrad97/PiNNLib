<<<<<<< HEAD
=======
# transformer/core/decoder_base.py

>>>>>>> 63ab059a96d2942b5932c5099df9b6851269ae39
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseDecoder(nn.Module, ABC):
    """
    Abstract base class for all decoder modules.
    Decoders map encoded sequences into task-specific outputs.
    Used in autoregressive forecasting, generation, or seq2seq tasks.
    """
    @abstractmethod
    def forward(self, x, encoded_context=None, mask=None):
        pass
