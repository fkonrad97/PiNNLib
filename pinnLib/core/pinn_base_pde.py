from abc import ABC, abstractmethod
import torch

class BasePDE(ABC):
    """
    Abstract base class for defining PDEs to be solved by PINNs.
    Each PDE subclass must define:
        - pde_residual
        - initial_condition
        - boundary_condition
        - sampling strategy for collocation points
    """

    def __init__(self, dim, device='cpu', N_f=1000, Nr=200, Nb=200):
        self.dim = dim
        self.device = device
        self.N_f = N_f
        self.Nr = Nr
        self.Nb = Nb


    @abstractmethod
    def pde_residual(self, model, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the PDE residual at collocation points X.
        """
        pass

    @abstractmethod
    def initial_condition(self, X0: torch.Tensor) -> torch.Tensor:
        """
        Return target values at initial condition points X0.
        """
        pass

    @abstractmethod
    def boundary_condition(self, X_b: torch.Tensor) -> torch.Tensor:
        """
        Return boundary loss or mask for boundary condition.
        """
        pass

    @abstractmethod
    def sample_collocation_points(self, N_f: int) -> torch.Tensor:
        """
        Sample collocation points in the domain.
        """
        pass

    @abstractmethod
    def sample_initial_points(self, N0: int) -> torch.Tensor:
        """
        Sample points on the initial surface.
        """
        pass

    @abstractmethod
    def sample_boundary_points(self, Nb: int) -> torch.Tensor:
        """
        Sample points on the boundary of the domain.
        """
        pass
