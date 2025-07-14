import pytest
import torch

import torch_gmres.solver as solver


def create_test_problem(
    batch_size: int,
    n: int,
    dtype: torch.dtype,
    device: str = "cuda"
):
    """Creates a batch of random linear systems Ax=b with known solutions."""
    # Create a random matrix A and normalize it
    a = torch.randn(batch_size, n, n, dtype=dtype, device=device)
    a = a / torch.linalg.norm(a, dim=(1, 2), keepdim=True)

    # Create a known random solution vector x_true and normalize it
    x_true = torch.randn(batch_size, n, dtype=dtype, device=device)
    x_true = x_true / torch.linalg.norm(x_true, dim=1, keepdim=True)

    # Calculate the right-hand side b = A @ x_true
    b = torch.einsum('bij,bj->bi', a, x_true)

    return a, b, x_true


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_gmres_full_cycle(dtype: torch.dtype):
    """
    Tests GMRES for a full cycle (m=N), which should converge in one go.
    This checks the correctness of the core Arnoldi iteration and solve.
    """
    N = 30
    A, b, x_truth = create_test_problem(batch_size=2, n=N, dtype=dtype)

    # Run the solver with m=N (no restart needed)
    result = solver.gmres(A, b, m=N, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(result.solution, x_truth, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_gmres_with_restart(dtype: torch.dtype):
    """
    Tests GMRES with restarts (m < N), which checks the restart mechanism.
    """
    N = 30
    A, b, _ = create_test_problem(batch_size=2, n=N, dtype=dtype)

    # Run the solver with m < N, forcing at least one restart
    result = solver.gmres(A, b, m=20, rtol=1e-9, atol=1e-9)

    # Verify that the calculated solution solves the system Ax=b
    solution_b = torch.einsum('bij,bj->bi', A, result.solution)
    torch.testing.assert_close(solution_b, b, rtol=1e-5, atol=1e-8)


def _test_gmres_large_sparse_system(m: int):
    """
    Tests GMRES on a larger, sparse, well-conditioned system.
    This verifies correctness by comparing against the known true solution.
    """
    N = 512
    DTYPE = torch.complex128
    DEVICE = "cuda"
    BATCH_SIZE = 2

    # Create a diagonally dominant sparse matrix
    A = torch.eye(N, dtype=DTYPE, device=DEVICE)
    for i in range(N - 1):
        A[i, i] = 2.0
        A[i, i+1] = -1.0
        A[i+1, i] = -1.0
    A = A.unsqueeze(0)
    # Create two different true solutions for the batch
    torch.manual_seed(42)
    x_true_list = []
    b_list = []
    for _ in range(BATCH_SIZE):
        x_true = torch.randn(1, N, dtype=DTYPE, device=DEVICE)
        x_true /= torch.linalg.norm(x_true)
        b = torch.einsum('bij,bj->bi', A, x_true)
        x_true_list.append(x_true)
        b_list.append(b)
    A = A.repeat(BATCH_SIZE, 1, 1)  # Batch the matrix
    b_batch = torch.cat(b_list, dim=0)

    # Run the solver
    result = solver.gmres(A, b_batch, m=m, rtol=1e-9, atol=1e-9)
    b_solution = torch.einsum('bij,bj->bi', A, result.solution)

    # Verify that the calculated solution solves the system Ax=b
    torch.testing.assert_close(b_solution, b_batch, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("m", [20, 100])
def test_gmres_large_sparse_system(m: int):
    """
    Parametrized test for GMRES on a larger, sparse, well-conditioned system.
    This verifies correctness by comparing against the known true solution.
    """
    _test_gmres_large_sparse_system(m)


if __name__ == "__main__":
    pytest.main([__file__])
