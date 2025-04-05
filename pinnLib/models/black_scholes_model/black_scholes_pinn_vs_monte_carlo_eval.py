import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pinnLib.eval.base_pinn_monte_carlo_eval import PINNvsMonteCarloEvaluator


class BlackScholesPINNvsMonteCarloEvaluator(PINNvsMonteCarloEvaluator):
    """
    Specialized evaluator for comparing a PINN trained on Black-Scholes PDE
    against a Monte Carlo benchmark.
    
    Assumes:
    - Time t = 0 for evaluation
    - Inputs are 2D: [S1, S2]
    """

    def plot_comparison(self, results: dict) -> None:
        """
        Plots 3D surfaces for PINN output, MC benchmark, and absolute error.
        Assumes 2D asset space (S1, S2).
        """
        inputs = results["inputs"]
        N = int(inputs.shape[0] ** 0.5)
        S1 = inputs[:, 0].reshape(N, N)
        S2 = inputs[:, 1].reshape(N, N)
        pinn = results["pinn"].reshape(N, N)
        mc = results["benchmark"].reshape(N, N)
        err = torch.abs(results["error"]).reshape(N, N)

        fig = plt.figure(figsize=(18, 5))

        titles = ["PINN Prediction", "Monte Carlo Benchmark", "Absolute Error"]
        surfaces = [pinn, mc, err]

        for i in range(3):
            ax = fig.add_subplot(1, 3, i + 1, projection='3d')
            ax.plot_surface(S1, S2, surfaces[i], cmap='viridis')
            ax.set_title(titles[i])
            ax.set_xlabel("S1")
            ax.set_ylabel("S2")
            ax.set_zlabel("Option Value")

        plt.tight_layout()
        plt.show()

