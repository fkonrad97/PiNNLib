import torch
import time
from pinnLib.eval.pinn_benchmark_eval import BasePINNBenchmarkEvaluator


class PINNvsMonteCarloEvaluator(BasePINNBenchmarkEvaluator):
    """
    Generic evaluator for comparing a PINN model against a Monte Carlo benchmark.
    Assumes inputs are space-only (e.g. S0), no time dimension unless passed manually.
    """

    def evaluate(self, inputs: torch.Tensor) -> dict:
        """
        Evaluates the PINN model and Monte Carlo benchmark on the given input points.
        inputs: Tensor of shape (N, d), e.g. S0 basket states (no time component).
        """
        # Optional: prepend time if required
        if inputs.shape[1] + 1 == self.model.model[0].in_features:
            t = torch.zeros(inputs.shape[0], 1, device=self.device)
            pinn_inputs = torch.cat([t, inputs], dim=1)
        else:
            pinn_inputs = inputs.to(self.device)

        self.model.eval()
        with torch.no_grad():
            start = time.time()
            pinn_output = self.model(pinn_inputs).squeeze().cpu()
            pinn_time = time.time() - start

            start = time.time()
            benchmark_output = self.benchmark.evaluate(inputs.to(self.device)).squeeze().cpu()
            bench_time = time.time() - start

        error = pinn_output - benchmark_output
        rmse = torch.sqrt(torch.mean(error ** 2)).item()
        mae = torch.mean(torch.abs(error)).item()

        return {
            "inputs": inputs.cpu(),
            "pinn": pinn_output,
            "benchmark": benchmark_output,
            "error": error,
            "metrics": {
                "RMSE": rmse,
                "MAE": mae,
                "PINN inference time (s)": pinn_time,
                "MC evaluation time (s)": bench_time
            }
        }

    def plot_comparison(self, results: dict) -> None:
        print("[PINNvsMonteCarloEvaluator] Plotting skipped: override in domain-specific subclass.")

    def summarize_performance(self, results: dict) -> None:
        print("=== PINN vs Monte Carlo Summary ===")
        for k, v in results["metrics"].items():
            print(f"{k}: {v:.6f}")
