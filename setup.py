from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='torch-gmres',
    # 1. Tell setuptools the root of packages is the 'src' directory
    package_dir={'': 'src'},
    # 2. Find packages inside the 'src' directory
    packages=find_packages(where='src'),
    ext_modules=[
        CUDAExtension(
            # 3. The extension name must match the package path
            # 'package_name.module_name' -> 'torch_gmres.cuda'
            name='torch_gmres.cuda',
            sources=['csrc/complex_gmres_kernel.cu']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
