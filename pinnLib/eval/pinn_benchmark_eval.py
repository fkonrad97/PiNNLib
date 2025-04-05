from abc import ABC, abstractmethod
import torch
from torch import nn
from pinnLib.eval.base_method_benchmark import BaseMethodBenchmark
from pinnLib.core.pinn_base_pde import BasePDE


class BasePINNBenchmarkEvaluator(ABC):
    """
    Abstract base class for evaluating the accuracy and performance of a PINN model
    against a benchmark method for solving a PDE.
    """

    def __init__(self,
                 model: nn.Module,
                 pde: BasePDE,
                 benchmark: BaseMethodBenchmark,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.pde = pde
        self.benchmark = benchmark
        self.device = device

    @abstractmethod
    def evaluate(self, inputs: torch.Tensor) -> dict:
        """
        Evaluate both the model and the benchmark on a given set of input points.

        Returns a dictionary including:
            - model predictions
            - benchmark values
            - error tensor
            - summary error metrics (RMSE, MAE, etc.)
        """
        pass

    @abstractmethod
    def plot_comparison(self, results: dict) -> None:
        """
        Visualize comparison between model and benchmark predictions.
        """
        pass

    @abstractmethod
    def summarize_performance(self, results: dict) -> None:
        """
        Compute and optionally log:
            - Runtime performance (e.g. inference speed)
            - Memory usage (optional)
            - Metric breakdown (if not already printed in `evaluate`)
        """
        pass
