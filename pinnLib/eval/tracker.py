from abc import ABC, abstractmethod
from typing import Dict, List
import csv

class BaseTracker(ABC):
    """
    Abstract base tracker for logging loss components during PINN training.
    """
    @abstractmethod
    def log(self, epoch: int, loss_dict: Dict[str, float]) -> None:
        pass

    @abstractmethod
    def get_history(self) -> List[Dict[str, float]]:
        pass

    @abstractmethod
    def export_csv(self, filename: str) -> None:
        pass

class StandardLossTracker(BaseTracker):
    def __init__(self):
        self.history = []

    def log(self, epoch: int, loss_dict: Dict[str, float]) -> None:
        entry = {"epoch": epoch, **loss_dict}
        self.history.append(entry)

    def get_history(self) -> List[Dict[str, float]]:
        return self.history

    def export_csv(self, filename: str) -> None:
        if not self.history:
            return
        keys = self.history[0].keys()
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)
