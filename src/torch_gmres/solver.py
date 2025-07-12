"""
Solver for Batched GMRES using a Custom CUDA Kernel
===================================================
"""
from dataclasses import dataclass

import torch

# This assumes your compiled module is named 'torch_gmres_cuda'
from . import cuda as torch_gmres_cuda


@dataclass
class GMRESResult:
    """
    Result class for the GMRES solver.
    """
    solution: torch.Tensor
    num_iterations: torch.Tensor
    residuals: torch.Tensor


@torch.no_grad()
def gmres(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    m: int = 50,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    max_restarts: int | None = None,
    verbose: bool = False,
) -> GMRESResult:
    """
    Solves a batch of linear systems Ax=b using the restarted GMRES method.

    This implementation uses a custom CUDA kernel for high performance.

    Args:
        A (torch.Tensor): The batched matrix of size (B, N, N).
        b (torch.Tensor): The batched right-hand side vector of size (B, N).
        x0 (torch.Tensor, optional): An initial guess for the solution.
            Defaults to a zero vector.
        m (int, optional): The restart cycle length (dimension of the Krylov subspace).
            Defaults to 50.
        rtol (float, optional): The relative tolerance for convergence.
            Defaults to 1e-5.
        atol (float, optional): The absolute tolerance for convergence.
            Defaults to 1e-8.
        max_restarts (int, optional): The maximum number of restarts.
            Defaults to N // m.
        verbose (bool, optional): If True, prints restart progress. Defaults to False.

    Returns:
        GMRESResult: An object containing the solution, total iteration counts,
        and final residuals for each system in the batch.
    """
    if A.ndim == 2:
        A = A.unsqueeze(0)
    if b.ndim == 1:
        b = b.unsqueeze(0)

    B, N = A.shape[:2]

    if A.dtype != b.dtype:
        raise ValueError("A and b must have the same dtype.")
    if A.device.type != "cuda" or b.device.type != "cuda":
        raise TypeError("Input tensors must be on a CUDA device.")

    real_dtype = torch.float64 if A.dtype == torch.complex128 else torch.float32
    b_norm = torch.linalg.norm(b, dim=1).to(real_dtype)

    current_x = torch.zeros_like(b) if x0 is None else x0.clone()

    if max_restarts is None:
        max_restarts = N // m

    total_iterations = torch.zeros(B, dtype=torch.int32, device=A.device)

    for i in range(max_restarts + 1):
        # r = b - A @ x for the current guess
        r = b - torch.einsum('bij,bj->bi', A, current_x)

        if verbose:
            print(f"Restart {i}, Max Iterations: {torch.max(total_iterations).item()}")

        # The CUDA kernel internally computes the solution update,
        # not the full solution. It solves Ay=r for y, where x_new = x_old + y.
        # Therefore, the initial "guess" passed to the kernel should be a zero vector,
        # and the "b" vector should be the current residual "r".
        x_update, ks, residuals = torch_gmres_cuda.run_iterations(
            A, r, torch.zeros_like(r), b_norm, m, rtol, atol
        )

        current_x += x_update
        total_iterations += ks

        # Get the residual at the end of the cycle for each batch item.
        # ks is 1-based, so subtract 1 for 0-based indexing.
        actual_indices = (ks - 1).clamp(min=0).view(-1, 1).to(torch.int64)
        actual_residuals = torch.gather(residuals, 1, actual_indices).squeeze(1)

        # Check for convergence
        tolerance = atol + rtol * b_norm
        if torch.all(actual_residuals <= tolerance):
            if verbose:
                print(
                    f"Converged after {i} restarts and {torch.max(total_iterations).item()} total iterations.")  # noqa: E501
            break

        if i == max_restarts and verbose:
            print("Warning: GMRES reached the maximum number of restarts without full convergence.")  # noqa: E501

    return GMRESResult(
        solution=current_x.view(B, N),
        num_iterations=total_iterations,
        residuals=actual_residuals,
    )
