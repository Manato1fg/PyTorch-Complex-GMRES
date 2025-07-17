import torch

from torch_gmres import GMRESResult


class GMRESStepResult:
    """
    Result dataclass for GMRES step.

    it is not a dataclass because it is used in torch.jit.script
    """

    def __init__(
        self,
        is_converged: bool,
        residual_norm: torch.Tensor,
        V: torch.Tensor,
        H: torch.Tensor,
        c: torch.Tensor,
        s: torch.Tensor,
        e1: torch.Tensor,
    ):
        self.is_converged = is_converged
        self.residual_norm = residual_norm
        self.V = V
        self.H = H
        self.c = c
        self.s = s
        self.e1 = e1


@torch.jit.script
def calculate_givens_rotation(
    a: torch.Tensor,
    b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes coefficients for a complex-safe Givens rotation.

    Finds c and s such that the rotation matrix Q = [[c.conj(), s.conj()], [-s, c]]  # noqa: E501
    is unitary and transforms the vector [a, b] to [r, 0].
    """
    # Use torch.hypot on the absolute values to get the L2 norm,
    # which is safe against overflow/underflow.
    norm = torch.hypot(torch.abs(a), torch.abs(b))
    mask = norm == 0
    c = torch.where(mask, torch.tensor(1.0, dtype=a.dtype, device=a.device), a / norm)  # noqa: E501
    s = torch.where(mask, torch.tensor(0.0, dtype=a.dtype, device=a.device), b / norm)  # noqa: E501

    return c, s


@torch.jit.script
def apply_givens_rotation(
    H: torch.Tensor,
    c: torch.Tensor,
    s: torch.Tensor,
    k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply a complex-safe Givens rotation to the H matrix (Corrected version).
    """
    # 1. Apply previous Givens rotations to the new column H[:, k]
    for i in range(k):
        # Store original values before updating
        h_ik = H[:, i, k].clone()
        h_i1k = H[:, i + 1, k].clone()

        # Apply the unitary rotation Q_i = [[c_i.conj(), s_i.conj()], [-s_i, c_i]]  # noqa: E501
        H[:, i, k] = c[:, i].conj() * h_ik + s[:, i].conj() * h_i1k
        H[:, i + 1, k] = -s[:, i] * h_ik + c[:, i] * h_i1k

    # 2. Compute the new Givens rotation for the current column k
    c[:, k], s[:, k] = calculate_givens_rotation(H[:, k, k], H[:, k + 1, k])

    # 3. Apply the new rotation to the k-th column to zero out H[k+1, k]
    H[:, k, k] = c[:, k].conj() * H[:, k, k] + s[:, k].conj() * H[:, k + 1, k]
    H[:, k + 1, k] = torch.tensor(0.0, dtype=H.dtype, device=H.device)

    return H, c, s


@torch.jit.script
def gmres_step(
    A: torch.Tensor,
    V: torch.Tensor,
    H: torch.Tensor,
    c: torch.Tensor,
    s: torch.Tensor,
    e1: torch.Tensor,
    residual_norm: torch.Tensor,
    k: int,
    tol: torch.Tensor,
) -> GMRESStepResult:
    # if verbose:
    #     Logger.get_instance().info(
    #         f"GMRES iteration {k}/{max_iter} | residual norm: {residual_norm.max()}")  # noqa: E501
    w = torch.einsum("bij,bj->bi", A, V[:, :, k])  # (N, N) @ (B, N) = (B, N)
    # Arnoldi's method with modified Gram-Schmidt orthogonalization
    for j in range(k + 1):
        # (B, N) @ (B, N) = (B, 1)
        H[:, j, k] = torch.sum(w * torch.conj_physical(V[:, :, j]), dim=-1)
        w = w - H[:, j, k].unsqueeze(-1) * V[:, :, j]
    H[:, k + 1, k] = torch.norm(w, dim=-1)
    V[:, :, k + 1] = w / H[:, k + 1, k].unsqueeze(-1)

    # Givens rotation
    H, c, s = apply_givens_rotation(H, c, s, k)

    # Apply the same rotation to the right-hand side vector e1.
    # The k-th rotation Q_k affects the k-th and (k+1)-th elements.
    e1_k = e1[:, k].clone()
    e1_k1 = e1[:, k + 1].clone()

    e1[:, k] = c[:, k].conj() * e1_k + s[:, k].conj() * e1_k1
    e1[:, k + 1] = -s[:, k] * e1_k + c[:, k] * e1_k1

    # Update residual norm
    residual_norm = torch.abs(e1[:, k + 1])
    result = GMRESStepResult(
        is_converged=False,
        residual_norm=residual_norm,
        V=V,
        H=H,
        c=c,
        s=s,
        e1=e1
    )
    if torch.all(residual_norm < tol):
        result.is_converged = True
    return result


def _gmres(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor,
    tol: torch.Tensor,
    m: int = 50,
) -> GMRESResult:
    """
    Generalized minimal residual method (GMRES)
    solve AX = B, where B is a batch of right-hand side vectors.

    Args:
        A: (N, N) or (B, N, N)
        b: (B, N)
        tol: convergence tolerance
        x0: initial guess
        m: restart parameter, number of iterations before restart

    Returns:
        GMRESResult:
    """
    if b.ndim == 1:
        b = b.unsqueeze(0)
    if x0.ndim == 1:
        x0 = x0.unsqueeze(0)
    B = b.shape[0]
    if A.ndim == 2:
        A = A.unsqueeze(0)

    device = A.device
    dtype = A.dtype
    assert x0.shape[0] == B
    r = b - torch.einsum("bij,bj->bi", A, x0)
    beta = torch.norm(r, dim=-1, keepdim=True)
    H = torch.zeros(B, m + 1, m, dtype=dtype, device=device)
    V = torch.zeros(B, b.shape[1], m + 1, dtype=dtype, device=device)  # noqa: E501
    V[:, :, 0] = r / beta

    e1 = torch.zeros(B, m + 1, dtype=dtype, device=device)
    e1[:, 0] = beta.squeeze(-1)

    c = torch.zeros(B, m, dtype=dtype, device=device)
    s = torch.zeros(B, m, dtype=dtype, device=device)

    # Initialize residual norm
    residual_norm = beta

    # GMRES iterations
    k = 0
    for k in range(m):
        result = gmres_step(
            A, V, H, c, s, e1, residual_norm, k, tol)
        V = result.V
        H = result.H
        c = result.c
        s = result.s
        e1 = result.e1
        residual_norm = result.residual_norm
        if result.is_converged:
            break

    y = torch.linalg.solve_triangular(
        H[:, 0:k + 1, 0:k + 1],
        e1[:, 0:k + 1].unsqueeze(-1),
        upper=True
    ).squeeze(-1)

    x = x0 + torch.einsum("bij,bj->bi", V[:, :, :k + 1], y)
    return GMRESResult(
        solution=x,
        num_iterations=torch.tensor(k + 1, dtype=torch.int32, device=device).repeat(B),
        residuals=residual_norm
    )


def gmres(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    m: int = 50,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    max_restarts: int | None = None,
) -> GMRESResult:
    if b.ndim == 1:
        b = b.unsqueeze(0)
        if x0 is not None:
            x0 = x0.unsqueeze(0)
    if x0 is None:
        x0 = torch.zeros_like(b, dtype=b.dtype, device=b.device)

    tol = atol + rtol * torch.linalg.norm(b, dim=-1, keepdim=True)

    result = None
    i = 0
    while max_restarts is None or i < max_restarts:
        result = _gmres(A, b, x0, tol, m)
        if torch.all(result.residuals < tol):
            break
        x0 = result.solution
        i += 1
    if result is None:
        raise RuntimeError("Unknown error in GMRES with restart")
    result.num_iterations = i * m + result.num_iterations
    return result
