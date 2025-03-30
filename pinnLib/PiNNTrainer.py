import torch
from torch import nn
from pinnLib.PiNNBasePDE import BasePDE
from pinnLib.PiNNBaseLossBuilder import BaseLossBuilder

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

    def train(self,
              epochs: int = 10000,
              print_every: int = 100,
              N_f: int = 1000,
              Nr: int = 200,
              Nb: int = 200):

        self.model.train()
        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()

            loss = self.loss_builder.compute_loss(
                self.model, self.pde, N_f=N_f, Nr=Nr, Nb=Nb
            )

            loss.backward()
            self.optimizer.step()

            if epoch % print_every == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

        print("Training complete.")