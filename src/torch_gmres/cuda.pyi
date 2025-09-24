# flake8: noqa
from typing import Tuple
import torch

# Stub for the compiled CUDA extension module torch_gmres.cuda


def run_iterations(
    _A: torch.Tensor,
    _b: torch.Tensor,
    _x0: torch.Tensor,
    _m: int,
    _rtol: float,
    _atol: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
