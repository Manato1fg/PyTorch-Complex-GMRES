# PyTorch-Complex-GMRES

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A high-performance, batched, and complex-valued GMRES solver for PyTorch, accelerated with a custom CUDA kernel.**

This library provides an efficient implementation of the restarted Generalized Minimal Residual (GMRES) method for solving batches of complex-valued linear systems of equations 
$$Ax=b$$
It leverages a custom CUDA kernel for significant performance gains over native PyTorch solutions, especially for large batches of small to medium-sized matrices.

***

## ✨ Key Features

* **Batched Computations**: Solves thousands of linear systems in parallel, fully utilizing modern GPU architectures.
* **Complex Number Support**: Natively handles `torch.complex64` and `torch.complex128` data types.
* **High-Performance CUDA Kernel**: The core iterative process is implemented in a single CUDA kernel, minimizing kernel launch overhead and maximizing data locality.
* **Automatic Shared Memory Optimization**: The CUDA kernel intelligently checks the GPU's available shared memory and uses it to store intermediate matrices (Hessenberg matrix, Givens rotations, etc.) for faster access. If the problem size exceeds the shared memory capacity, it gracefully falls back to using global memory.
* **Restarted GMRES**: Implements the restarted version of GMRES (`GMRES(m)`) to manage memory consumption for large systems.
* **Easy-to-Use Python API**: A clean and simple Python interface that integrates seamlessly with existing PyTorch workflows.

***

## ⚙️ Installation

To use this library, you need a PyTorch installation with CUDA support and the CUDA toolkit.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/PyTorch-Complex-GMRES.git](https://github.com/your-username/PyTorch-Complex-GMRES.git)
    cd PyTorch-Complex-GMRES
    uv sync --extra cu126
    ```

2.  **Install the package:**
    Run the following command in the project root to build and install the CUDA extension.
    ```bash
    uv run setup.py install
    ```