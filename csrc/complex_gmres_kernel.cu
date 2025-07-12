#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/complex.h>

/**
 * @brief Performs a parallel reduction (sum) of values held by all threads in a CUDA block.
 * @tparam T The data type of the values to be reduced.
 * @param val The value held by the current thread.
 * @param shared_val A pointer to the shared memory used for reduction.
 * @return The total sum, returned by thread 0.
 */
template <typename T>
__device__ T block_reduce_sum(T val, T* shared_val) {
    int tid = threadIdx.x;
    shared_val[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_val[tid] = shared_val[tid] + shared_val[tid + s];
        }
        __syncthreads();
    }
    return shared_val[0];
}

/**
 * @brief Real-valued version of block_reduce_sum.
 */
template <typename T_real>
__device__ T_real block_reduce_sum_real(T_real val, T_real* shared_val) {
    int tid = threadIdx.x;
    shared_val[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_val[tid] = shared_val[tid] + shared_val[tid + s];
        }
        __syncthreads();
    }
    return shared_val[0];
}

/**
 * @brief Computes sqrt(|v1|^2 + |v2|^2) for complex numbers v1 and v2.
 */
template <typename T, typename T_real>
__device__ T_real hypot(const T v1, const T v2) {
    T_real a = v1.real();
    T_real b = v1.imag();
    T_real c = v2.real();
    T_real d = v2.imag();
    return sqrt(a * a + b * b + c * c + d * d);
}

/**
 * @brief Computes the squared L2 norm ||v||^2 of a complex vector.
 */
template <typename T, typename T_real>
__device__ T_real norm_sq(int n, const T* v, void* shared_mem) {
    T_real* shared_val_real = reinterpret_cast<T_real*>(shared_mem);
    T_real thread_sum = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T val = v[i];
        thread_sum += val.real() * val.real() + val.imag() * val.imag();
    }
    return block_reduce_sum_real<T_real>(thread_sum, shared_val_real);
}

/**
 * @brief Executes one cycle of iterative computations for the GMRES method.
 * @tparam T Complex type (c10::complex<float> or c10::complex<double>).
 * @tparam T_real Real type (float or double).
 */
template <typename T, typename T_real>
__global__ void gmres_iterations_kernel(
    // Inputs
    const T* A, const T* b, const T* x,
    // Workspace (Global Memory)
    T* V, T* w,
    // Outputs (Global Memory)
    T* x_out, int* k_out, T_real* residuals_out,
    // Workspace (Global Memory)
    T* H_global, T* c_global, T* s_global, T* e1_global,
    // Parameters
    int n, int m, T_real rtol, T_real atol, const T_real* b_norm_ptr, int block_size,
    // flag for shared memory usage
    bool use_shared_memory)
{
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // Offset pointers to the data for the current batch item
    A += batch_idx * n * n;
    b += batch_idx * n;
    x += batch_idx * n;
    V += batch_idx * n * (m + 1);
    w += batch_idx * n;
    x_out += batch_idx * n;
    residuals_out += batch_idx * (m + 1);

    // 共有メモリを確保 (リダクションバッファと収束フラグは常に共有メモリが望ましい)
    extern __shared__ char smem_main[];
    T* reduce_buffer    = reinterpret_cast<T*>(smem_main);
    bool* converged_flag = reinterpret_cast<bool*>(reduce_buffer + block_size);

    if (use_shared_memory) {
        // --- 共有メモリパス ---
        // 共有メモリからポインタを設定
        H  = reinterpret_cast<T*>(converged_flag + 1);
        c  = H + (m + 1) * m;
        s  = c + m;
        e1 = s + m;
    } else {
        // --- グローバルメモリパス ---
        // グローバルメモリのポインタをバッチインデックスでオフセットして設定
        const size_t h_size = (size_t)(m + 1) * m;
        const size_t c_size = m;
        const size_t s_size = m;
        const size_t e1_size = m + 1;
        
        H = H_global + batch_idx * h_size;
        c = c_global + batch_idx * c_size;
        s = s_global + batch_idx * s_size;
        e1 = e1_global + batch_idx * e1_size;
    }

    // 0. Initialize: r = b - A @ x
    for (int i = tid; i < n; i += blockDim.x) {
        T ax = T(0.0, 0.0);
        for (int j = 0; j < n; ++j) {
            ax = ax + A[i * n + j] * x[j];
        }
        w[i] = b[i] - ax;
    }
    __syncthreads();

    T_real beta = sqrt(norm_sq<T, T_real>(n, w, reduce_buffer));
    __syncthreads();

    if (beta < 1e-12) {
        if (tid == 0) {
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            k_out[batch_idx] = 0;
        }
        return;
    }

    if (tid == 0) {
        for(int i = 1; i <= m; ++i) e1[i] = T(0.0, 0.0);
        e1[0] = T(beta, 0.0);
        *converged_flag = false;
    }
    __syncthreads();
    
    if (tid == 0) residuals_out[0] = beta;

    for (int i = tid; i < n; i += blockDim.x) {
        V[i * (m + 1) + 0] = w[i] / e1[0];
    }
    __syncthreads();

    // 1. GMRES Main Loop (Arnoldi Iteration)
    int k;
    for (k = 0; k < m; ++k) {
        // Matrix-vector product: w = A @ v_k
        for (int i = tid; i < n; i += blockDim.x) {
            T aw = T(0.0, 0.0);
            for (int j = 0; j < n; ++j) {
                aw = aw + A[i * n + j] * V[j * (m + 1) + k];
            }
            w[i] = aw;
        }
        __syncthreads();

        // Modified Gram-Schmidt orthogonalization
        for (int j = 0; j <= k; ++j) {
            T thread_sum = T(0.0, 0.0);
            for (int i = tid; i < n; i += blockDim.x) {
                thread_sum = thread_sum + w[i] * std::conj(V[i * (m + 1) + j]);
            }
            T h_jk = block_reduce_sum<T>(thread_sum, reduce_buffer);
            __syncthreads();
            if (tid == 0) H[j * m + k] = h_jk;
            __syncthreads();

            for (int i = tid; i < n; i += blockDim.x) {
                w[i] = w[i] - H[j * m + k] * V[i * (m + 1) + j];
            }
            __syncthreads();
        }

        T_real h_k1_k = sqrt(norm_sq<T, T_real>(n, w, reduce_buffer));
        __syncthreads();
        
        if (tid == 0) H[(k + 1) * m + k] = T(h_k1_k, 0.0);
        T h_k1_k_complex = T(h_k1_k, 0.0);
        __syncthreads();

        for (int i = tid; i < n; i += blockDim.x) {
            if (h_k1_k > 1e-12) {
                 V[i * (m + 1) + (k + 1)] = w[i] / h_k1_k_complex;
            } else {
                 V[i * (m + 1) + (k + 1)] = T(0.0, 0.0);
            }
        }
        __syncthreads();
        
        // Apply previous Givens rotations and compute the new one
        if (tid == 0) {
            for (int j = 0; j < k; ++j) {
                T h_ik = H[j * m + k];
                T h_i1k = H[(j + 1) * m + k];
                H[j * m + k] = std::conj(c[j]) * h_ik + std::conj(s[j]) * h_i1k;
                H[(j + 1) * m + k] = -s[j] * h_ik + c[j] * h_i1k;
            }
            
            T_real norm = hypot<T, T_real>(H[k * m + k], H[(k + 1) * m + k]);
            T norm_complex = T(norm, 0.0);
            if (norm > 1e-12) {
                c[k] = H[k * m + k] / norm_complex;
                s[k] = H[(k + 1) * m + k] / norm_complex;
            } else {
                c[k] = T(1.0, 0.0);
                s[k] = T(0.0, 0.0);
            }
            H[k * m + k] = std::conj(c[k]) * H[k * m + k] + std::conj(s[k]) * H[(k + 1) * m + k];
            H[(k + 1) * m + k] = T(0.0, 0.0);

            T e1_k = e1[k];
            T e1_k1 = e1[k + 1];
            e1[k] = std::conj(c[k]) * e1_k + std::conj(s[k]) * e1_k1;
            e1[k + 1] = -s[k] * e1_k + c[k] * e1_k1;

            T_real residual = std::abs(e1[k + 1]);
            residuals_out[k + 1] = residual;

            T_real b_norm = static_cast<T_real>(b_norm_ptr[batch_idx]);
            T_real acceptance_tol = atol + rtol * b_norm;

            if (residual < acceptance_tol) {
                *converged_flag = true;
            }
        }
        
        __syncthreads();
        if (*converged_flag) {
            break;
        }
    }

    int num_iters;
    if (k < m) {
        num_iters = k + 1;
    } else {
        num_iters = m;
    }

    // 2. Solve the least-squares problem and update solution
    if (tid == 0) {
        // workspace w will be repurposed to store the solution y
        for (int j = num_iters - 1; j >= 0; --j) {
            T sum = T(0.0, 0.0);
            for (int i = j + 1; i < num_iters; ++i) {
                sum = sum + H[j * m + i] * w[i];
            }
            if (std::abs(H[j * m + j]) > 1e-12) {
                w[j] = (e1[j] - sum) / H[j * m + j];
            } else {
                w[j] = T(0.0, 0.0);
            }
        }
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        T sum = T(0.0, 0.0);
        for (int j = 0; j < num_iters; ++j) {
            sum = sum + w[j] * V[i * (m + 1) + j];
        }
        x_out[i] = x[i] + sum;
    }
    __syncthreads();

    if (tid == 0) {
        k_out[batch_idx] = num_iters;
    }
}

std::vector<torch::Tensor> gmres_launcher(
    torch::Tensor A, torch::Tensor b, torch::Tensor x0, torch::Tensor b_norm,
    int m, double rtol, double atol)
{
    const auto B = A.size(0);
    const auto N = A.size(1);
    const auto options = A.options();

    auto V = torch::empty({B, N, m + 1}, options);
    auto w = torch::empty({B, N}, options);
    auto x_out = torch::empty({B, N}, options);
    auto k_out = torch::empty({B}, options.dtype(torch::kInt));
    auto residuals_out = torch::empty({B, m + 1}, options.dtype(A.dtype() == torch::kComplexFloat ? torch::kFloat : torch::kDouble));

    // get the device properties to determine shared memory limits
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, A.device().index());
    const size_t max_smem_per_block = props.sharedMemPerBlock;

    // calculate the size of shared memory needed for the kernel
    const int block_size = 256;
    size_t smem_reduce_size = block_size * A.element_size();
    size_t converged_flag_size = sizeof(bool);
    
    // Calculate sizes for H, c, s, e1
    size_t smem_H_size = (size_t)(m + 1) * m * A.element_size();
    size_t smem_c_s_e1_size = (size_t)(m + m + m + 1) * A.element_size();
    size_t required_full_smem = smem_reduce_size + converged_flag_size + smem_H_size + smem_c_s_e1_size;

    // Check if we can use shared memory
    bool use_shared_memory = (required_full_smem <= max_smem_per_block);
    
    size_t launch_smem_size;
    torch::Tensor H_global, c_global, s_global, e1_global;

    if (use_shared_memory) {
        // shared memory path
        launch_smem_size = required_full_smem;
    } else {
        // global memory path
        launch_smem_size = smem_reduce_size + converged_flag_size;
        
        // Allocate global memory for H, c, s, e1
        H_global = torch::empty({B, (long long)(m + 1) * m}, options);
        c_global = torch::empty({B, m}, options);
        s_global = torch::empty({B, m}, options);
        e1_global = torch::empty({B, m + 1}, options);
    }

    // --- Kernel Launch Configuration ---
    const int grid_size = B;

    AT_DISPATCH_COMPLEX_TYPES(A.scalar_type(), "gmres_launcher", [&] {
        using real_t = typename scalar_t::value_type;

        gmres_iterations_kernel<scalar_t, real_t><<<grid_size, block_size, launch_smem_size>>>(
            A.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), x0.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
            x_out.data_ptr<scalar_t>(), k_out.data_ptr<int>(), residuals_out.data_ptr<real_t>(),
            use_shared_memory ? nullptr : H_global.data_ptr<scalar_t>(),
            use_shared_memory ? nullptr : c_global.data_ptr<scalar_t>(),
            use_shared_memory ? nullptr : s_global.data_ptr<scalar_t>(),
            use_shared_memory ? nullptr : e1_global.data_ptr<scalar_t>(),
            N, m, static_cast<real_t>(rtol), static_cast<real_t>(atol),
            b_norm.data_ptr<real_t>(),
            block_size,
            use_shared_memory,
        );
    });

    return {x_out, k_out, residuals_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_iterations", &gmres_launcher, "Run one cycle of GMRES iterations on CUDA");
}