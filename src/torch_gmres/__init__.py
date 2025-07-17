from dataclasses import dataclass

import torch


@dataclass
class GMRESResult:
    """
    Result class for the GMRES solver.
    """
    solution: torch.Tensor
    num_iterations: torch.Tensor
    residuals: torch.Tensor
