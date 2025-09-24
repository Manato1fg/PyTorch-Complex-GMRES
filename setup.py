from setuptools import find_packages, setup
import os
import shutil


def _should_build_cuda_ext() -> bool:
    """Return True if we should attempt to build the CUDA extension.

    Conditions:
    - Explicitly disabled with env BUILD_CUDA_EXT=0 -> False
    - Explicitly enabled with env BUILD_CUDA_EXT=1 -> True
    - Otherwise, build only if nvcc is available (simple heuristic)
    """
    flag = os.environ.get("BUILD_CUDA_EXT")
    if flag is not None:
        return flag == "1"
    # Heuristic: nvcc must be available to build CUDA C++ extension
    return shutil.which("nvcc") is not None


ext_modules = []
cmdclass = {}
if _should_build_cuda_ext():
    # Import torch's build helpers only when we actually build the CUDA ext
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # type: ignore
    ext_modules.append(
        CUDAExtension(
            # 3. The extension name must match the package path
            # 'package_name.module_name' -> 'torch_gmres.cuda'
            name='torch_gmres.cuda',
            sources=['csrc/complex_gmres_kernel.cu']
        )
    )
    cmdclass = {'build_ext': BuildExtension}

setup(
    name='pytorch_complex_gmres',
    # 1. Tell setuptools the root of packages is the 'src' directory
    package_dir={'': 'src'},
    # 2. Find packages inside the 'src' directory
    packages=find_packages(where='src'),
    package_data={
        # ship type hints and stub files
        'torch_gmres': ['py.typed', '*.pyi'],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass
)
