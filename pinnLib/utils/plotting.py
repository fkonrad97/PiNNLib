from abc import ABC, abstractmethod
from typing import List, Dict


class BasePlotter(ABC):
    @abstractmethod
    def plot(self, history: List[Dict[str, float]]) -> None:
        pass


class LossCurvePlotter(BasePlotter):
    def plot(self, history: List[Dict[str, float]]) -> None:
        import matplotlib.pyplot as plt
        if not history: return

        keys = [k for k in history[0].keys() if k != 'epoch']
        for key in keys:
            plt.plot([entry[key] for entry in history], label=key)

        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Components Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
