# transformer/core/decoder_base.py

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
