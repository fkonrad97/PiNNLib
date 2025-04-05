from abc import ABC, abstractmethod
import torch

class BaseMethodBenchmark(ABC):
    """
    Abstract base class for benchmarking PINNs against traditional methods.
    """
    @abstractmethod
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Given input points X, return the ground truth values.
        """
        pass

    @abstractmethod
    def compute_error(self, model_output: torch.Tensor, ground_truth: torch.Tensor) -> float:
        """
        Compare model output to ground truth, return error metric (e.g. RMSE).
        """
        pass