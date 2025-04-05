from abc import ABC, abstractmethod
import torch
from torch import nn
from pinnLib.core.pinn_base_pde import BasePDE


class BaseLossBuilder(ABC):
    """
    Abstract base class for different types of PINN loss builders.
    Allows extension to variational, data-driven, or hybrid models.
    """

    def __init__(self, loss_fn: nn.Module = nn.MSELoss()):
        self.loss_fn = loss_fn

    @abstractmethod
    def compute_loss_components(self,
                     model: nn.Module,
                     pde: BasePDE,
                     N_f: int = 1000,
                     Nr: int = 200,
                     Nb: int = 200) -> torch.Tensor:
        """
        Compute the total loss given a model and a PDE instance. Return a dict of separate loss components.
        """
        pass

    @abstractmethod
    def compute_total_loss(self, loss_dict) -> torch.Tensor:
        """
        Aggregate the separated loss components and return a scalar in the form of torch.Tensor
        """
        pass

class StandardPINNLossBuilder(BaseLossBuilder):
    """
    Computes the total loss for training a PINN by combining:
        - PDE residual loss
        - Result condition loss (initial/terminal)
        - Boundary condition loss

    Allows custom weighting of each term.
    """

    def __init__(self, 
                 pde_weight: float = 1.0, 
                 result_weight: float = 1.0, 
                 boundary_weight: float = 1.0,
                 loss_fn: nn.Module = nn.MSELoss()):
        super().__init__(loss_fn)
        self.pde_weight = pde_weight
        self.result_weight = result_weight
        self.boundary_weight = boundary_weight

    def compute_loss_components(self, model: nn.Module, pde: BasePDE, N_f: int = 1000, Nr: int = 200, Nb: int = 200) -> torch.Tensor:
        """
        Computes weighted total loss for the given model and PDE. Return a dict of separate loss components.
        """
        losses = {}

        if self.pde_weight > 0:
            X_f = pde.sample_collocation_points(N_f=N_f)
            residual = pde.pde_residual(model, X_f)
            losses['pde'] = self.loss_fn(residual, torch.zeros_like(residual))

        if self.result_weight > 0:
            X_res = pde.sample_result_points(Nr=Nr)
            target_vals = pde.result_condition(X_res)
            pred_vals = model(X_res)
            # loss += self.result_weight * self.loss_fn(pred_vals, target_vals)
            losses['result'] = self.loss_fn(pred_vals, target_vals)

        if self.boundary_weight > 0:
            X_b = pde.sample_boundary_points(Nb=Nb)
            target_vals = pde.boundary_condition(X_b)
            pred_vals = model(X_b)
            # loss += self.boundary_weight * self.loss_fn(pred_vals, target_vals)
            losses['boundary'] = self.loss_fn(pred_vals, target_vals)

        return losses
    
    def compute_total_loss(self, loss_dict):
        return (
            self.pde_weight * loss_dict.get('pde', 0.0) +
            self.result_weight * loss_dict.get('result', 0.0) +
            self.boundary_weight * loss_dict.get('boundary', 0.0)
    )
