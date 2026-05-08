import argparse
import json
import time

import torch

from torch_gmres import solver, solver_python

from prettytable import PrettyTable

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GMRES solver.")
    parser.add_argument("--batch_size", "-bs", type=int, default=10,
                        help="Batch size for the linear systems.")
    parser.add_argument("--matrix_size", "-ms", type=int, default=100,
                        help="Size of the square matrix A.")
    parser.add_argument("--num_trials", "-nt",  type=int, default=10,
                        help="Number of trials for benchmarking.")
    parser.add_argument("--m", "-m", type=int, default=50,
                        help="Restart cycle length (dimension of Krylov subspace).")
    parser.add_argument("--atol", "-atol", type=float, default=1e-8,
                        help="Absolute tolerance for convergence.")
    parser.add_argument("--rtol", "-rtol", type=float, default=1e-5,
                        help="Relative tolerance for convergence.")
    parser.add_argument("--simulate_infinite_reflection", "-s", action='store_true',
                        help="Simulate infinite reflection in the benchmark.")
    parser.add_argument("--norm_limit", "-nl", type=float, default=1.0,
                        help="Limit for the norm of the matrix A.")

    args = parser.parse_args()
    batch_size = args.batch_size
    matrix_size = args.matrix_size
    num_trials = args.num_trials
    m = args.m
    atol = args.atol
    rtol = args.rtol
    simulate_infinite_reflection = args.simulate_infinite_reflection
    norm_limit = args.norm_limit

    result_path = "benchmark_results.json"

    A = torch.randn(num_trials, batch_size, matrix_size, matrix_size,
                    device='cuda', dtype=torch.complex128)
    A /= torch.linalg.norm(A, dim=(-2, -1), keepdim=True) * norm_limit
    x_true = torch.randn(num_trials, batch_size, matrix_size,
                         device='cuda', dtype=torch.complex128)
    if simulate_infinite_reflection:
        identity = torch.eye(matrix_size, device='cuda',
                             dtype=torch.complex128).unsqueeze(
            0).unsqueeze(0).repeat(num_trials, batch_size, 1, 1)
        A = (identity - x_true[:, :, :, None] * A)
        b = x_true
    else:
        b = torch.einsum("nbij,nbj->nbi", A, x_true)

    print("Starting benchmark...")

    print(f"Batch size: {batch_size}, Matrix size: {matrix_size}, Trials: {num_trials}")

    print("Running GMRES with custom CUDA kernel...")
    # Warm-up to exclude one-time JIT/initialization overhead
    _ = solver.gmres(A[0], b[0], x0=None, m=m, rtol=rtol, atol=atol)
    torch.cuda.synchronize()
    # Benchmark the solver with CUDA kernel
    solver_results = []
    for i in range(num_trials):
        start_time = time.time()
        result = solver.gmres(A[i], b[i], x0=None, m=m, rtol=rtol, atol=atol)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        solver_results.append({
            "num_iterations": result.num_iterations.tolist(),
            "residuals": result.residuals.tolist(),
            "time": elapsed_time
        })

    print("Running GMRES with Python implementation...")
    _ = solver_python.gmres(A[0], b[0], x0=None, m=m, rtol=rtol, atol=atol)
    torch.cuda.synchronize()
    # Benchmark the solver with Python implementation
    solver_python_results = []
    for i in range(num_trials):
        start_time = time.time()
        result = solver_python.gmres(A[i], b[i], x0=None, m=m, rtol=rtol, atol=atol)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        solver_python_results.append({
            "num_iterations": result.num_iterations.tolist(),
            "residuals": result.residuals.tolist(),
            "time": elapsed_time
        })

    print("Benchmark completed.")

    results = {
        "solver": {
            "custom_cuda": solver_results,
            "python_implementation": solver_python_results
        }
    }

    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_path}")

    # Print results in a table format
    table = PrettyTable()
    table.field_names = ["Trial", "Solver", "Time (s)", "Num Iterations", "Residuals"]

    def format_residual(residual: float) -> str:
        return f"{residual:.3e}"

    def format_time(time: float) -> str:
        return f"{time:.3f}"

    def format_iterations(iterations: list) -> str:
        ave = sum(iterations) / len(iterations)
        return f"{ave:.2f} (max: {max(iterations)})"

    for i in range(num_trials):
        table.add_row([
            i + 1,
            "Custom CUDA",
            format_time(solver_results[i]["time"]),  # type: ignore
            format_iterations(solver_results[i]["num_iterations"]),  # type: ignore
            # Display last residual for clarity
            format_residual(solver_results[i]["residuals"][-1])  # type: ignore
        ])
        table.add_row([
            i + 1,
            "Python Implementation",
            format_time(solver_python_results[i]["time"]),  # type: ignore
            format_iterations(
                solver_python_results[i]["num_iterations"]),  # type: ignore
            format_residual(solver_python_results[i]["residuals"][-1])  # type: ignore
        ])
    print(table)
    print("Benchmarking complete.")

    cuda_time = sum(result["time"]  # type: ignore
                    for result in solver_results) / num_trials
    python_time = sum(
        result["time"] for result in solver_python_results) / num_trials  # type: ignore
    print(f"Average time for custom CUDA GMRES: {cuda_time:.3f} seconds")
    print(f"Average time for Python GMRES: {python_time:.3f} seconds")
    print(f"Speedup: {python_time / cuda_time:.2f}x")
