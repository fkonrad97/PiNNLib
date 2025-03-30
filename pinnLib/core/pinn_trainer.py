import torch
from torch import nn
from pinnLib.core.pinn_base_pde import BasePDE
from pinnLib.core.pinn_loss_builder import BaseLossBuilder
from pinnLib.eval.tracker import StandardLossTracker
from pinnLib.utils.plotting import LossCurvePlotter

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @abstractmethod
    def train(self, *args, **kwargs):
        pass


class StandardPINNTrainer(BaseTrainer):
    def __init__(self,
                 model: nn.Module,
                 pde: BasePDE,
                 loss_builder: BaseLossBuilder,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.pde = pde
        self.loss_builder = loss_builder
        self.optimizer = optimizer
        self.device = device
        self.tracker = StandardLossTracker()
        self.plotter = LossCurvePlotter()

    def train(self,
              epochs: int = 10000,
              print_every: int = 100,
              N_f: int = 1000,
              Nr: int = 200,
              Nb: int = 200):

        self.model.train()

        for epoch in range(1, epochs + 1):
            loss_dict = self._train_epoch(epoch, N_f, Nr, Nb)

            if epoch % print_every == 0:
                self._print_log(epoch, loss_dict)

        print("Training complete.")
        self.tracker.export_csv("loss_history.csv")
        self.plotter.plot(self.tracker.get_history())

    def _train_epoch(self, epoch: int, N_f: int, Nr: int, Nb: int) -> dict:
        self.optimizer.zero_grad()
        loss_components = self.loss_builder.compute_loss_components(
            self.model, self.pde, N_f=N_f, Nr=Nr, Nb=Nb
        )
        total_loss = self.loss_builder.compute_total_loss(loss_components)
        total_loss.backward()
        self.optimizer.step()

        loss_entry = {
            'total loss': total_loss.item(),
            'pde loss': loss_components.get('pde', torch.tensor(0.0)).item(),
            'result loss': loss_components.get('result', torch.tensor(0.0)).item(),
            'boundary loss': loss_components.get('boundary', torch.tensor(0.0)).item()
        }
        self.tracker.log(epoch, loss_entry)
        return loss_entry

    def _print_log(self, epoch: int, loss_dict: dict) -> None:
        print(f"Epoch {epoch:5d} | " +
              " | ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items()))
