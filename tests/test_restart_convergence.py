"""
Test cases specifically for GMRES convergence with restart parameter m < n.
These tests verify that the fix for the convergence issue is working correctly.
"""
import torch_gmres.solver as solver
import pytest
import torch

torch.cuda.is_available()

# Skip the whole test module when CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for these tests"
)


def create_test_problem(
    batch_size: int,
    n: int,
    dtype: torch.dtype,
    device: str = "cuda",
    norm_limit: float = 0.2
):
    """Creates a batch of random linear systems Ax=b with known solutions."""
    # Create a random matrix A and normalize it
    a = torch.randn(batch_size, n, n, dtype=dtype, device=device)
    a = a / torch.linalg.norm(a, dim=(1, 2), keepdim=True) * norm_limit

    # Create a known random solution vector x_true and normalize it
    x_true = torch.randn(batch_size, n, dtype=dtype, device=device)
    x_true = x_true / torch.linalg.norm(x_true, dim=1, keepdim=True)

    # Calculate the right-hand side b = A @ x_true
    b = torch.einsum('bij,bj->bi', a, x_true)

    return a, b, x_true


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("n,m", [(50, 10), (100, 20), (30, 5)])
def test_gmres_restart_convergence_m_less_than_n(dtype: torch.dtype, n: int, m: int):
    """
    Tests GMRES with m < n to verify that the restart mechanism works correctly.
    This is the key test case that should fail with the old buggy code.
    """
    A, b, _ = create_test_problem(batch_size=2, n=n, dtype=dtype)

    # Run the solver with m < n, which requires restarts
    result = solver.gmres(A, b, m=m, rtol=1e-5, atol=1e-8, max_restarts=50, verbose=False)

    # Verify that the calculated solution solves the system Ax=b
    solution_b = torch.einsum('bij,bj->bi', A, result.solution)
    torch.testing.assert_close(solution_b, b, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_gmres_restart_single_batch(dtype: torch.dtype):
    """
    Tests GMRES with m < n for a single batch element.
    """
    N = 40
    M = 8
    A, b, _ = create_test_problem(batch_size=1, n=N, dtype=dtype)

    result = solver.gmres(A, b, m=M, rtol=1e-5, atol=1e-8, max_restarts=20)

    # Check the solution
    solution_b = torch.einsum('bij,bj->bi', A, result.solution)
    relative_error = torch.linalg.norm(solution_b - b) / torch.linalg.norm(b)
    
    assert relative_error < 1e-3, f"Solution did not converge: relative error = {relative_error.item()}"


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_gmres_edge_case_m_equals_1(dtype: torch.dtype):
    """
    Tests GMRES with m=1, an edge case that requires many restarts.
    """
    N = 20
    M = 1
    A, b, _ = create_test_problem(batch_size=1, n=N, dtype=dtype)

    result = solver.gmres(A, b, m=M, rtol=1e-4, atol=1e-6, max_restarts=100)

    # Check the solution
    solution_b = torch.einsum('bij,bj->bi', A, result.solution)
    relative_error = torch.linalg.norm(solution_b - b) / torch.linalg.norm(b)
    
    # More lenient tolerance for this edge case
    assert relative_error < 1e-2, f"Solution did not converge: relative error = {relative_error.item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
