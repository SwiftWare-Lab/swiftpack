import os
import sys
import time
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np

# Set single-threaded execution for external BLAS libraries to ensure reproducible benchmarks
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Default Matrix Dimensions & Block Sizes
N = 512
BLOCK_SIZE = 32
BLOCK_SIZE_L2 = 128
BLOCK_SIZE_L1 = 32

# ---------------------------------------------------------
# Baseline 0: Pure Python Naive Matmul
# ---------------------------------------------------------
def matmul_python_0(A, B, C):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            C[i, j] = 0.0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

# ---------------------------------------------------------
# Baseline 1: Naive Matmul in Numba (IJK)
# ---------------------------------------------------------
@njit
def matmul_numba_1(A, B, C):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            C[i, j] = 0.0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

# ---------------------------------------------------------
# Baseline 2: Loop Order Permutations (6 explicit functions)
# ---------------------------------------------------------
@njit
def matmul_2_ijk(A, B, C):
    n = A.shape[0]
    C.fill(0.0)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

@njit
def matmul_2_ikj(A, B, C):
    n = A.shape[0]
    C.fill(0.0)
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i, j] += A[i, k] * B[k, j]

@njit
def matmul_2_jik(A, B, C):
    n = A.shape[0]
    C.fill(0.0)
    for j in range(n):
        for i in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

@njit
def matmul_2_jki(A, B, C):
    n = A.shape[0]
    C.fill(0.0)
    for j in range(n):
        for k in range(n):
            for i in range(n):
                C[i, j] += A[i, k] * B[k, j]

@njit
def matmul_2_kij(A, B, C):
    n = A.shape[0]
    C.fill(0.0)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                C[i, j] += A[i, k] * B[k, j]

@njit
def matmul_2_kji(A, B, C):
    n = A.shape[0]
    C.fill(0.0)
    for k in range(n):
        for j in range(n):
            for i in range(n):
                C[i, j] += A[i, k] * B[k, j]

# ---------------------------------------------------------
# Baseline 3: Optimization Flags for Backend (IKJ order)
# ---------------------------------------------------------
@njit(fastmath=True)
def matmul_opt_flags_3(A, B, C):
    n = A.shape[0]
    C.fill(0.0)
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i, j] += A[i, k] * B[k, j]

# ---------------------------------------------------------
# Baseline 4: Parallel Loop Versions
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def matmul_parallel_i_4(A, B, C):
    n = A.shape[0]
    C.fill(0.0)
    for i in prange(n):
        for k in range(n):
            for j in range(n):
                C[i, j] += A[i, k] * B[k, j]

@njit(parallel=True, fastmath=True)
def matmul_parallel_k_4(A, B, C):
    n = A.shape[0]
    C.fill(0.0)
    for k in range(n):
        for i in prange(n):
            for j in range(n):
                C[i, j] += A[i, k] * B[k, j]

@njit(parallel=True, fastmath=True)
def matmul_parallel_j_4(A, B, C):
    n = A.shape[0]
    C.fill(0.0)
    for j in prange(n):
        for i in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

# ---------------------------------------------------------
# Baseline 5: Blocked (Tiled) Parallel I Code
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def matmul_blocked_parallel_5(A, B, C):
    n = A.shape[0]
    bs = BLOCK_SIZE
    C.fill(0.0)

    num_i_blocks = (n + bs - 1) // bs

    for b in prange(num_i_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)

        for k_block in range(0, n, bs):
            for j_block in range(0, n, bs):
                k_end = min(k_block + bs, n)
                j_end = min(j_block + bs, n)

                for i in range(i_block, i_end):
                    for k in range(k_block, k_end):
                        for j in range(j_block, j_end):
                            C[i, j] += A[i, k] * B[k, j]

# ---------------------------------------------------------
# Baseline 6: Blocked Parallel I using np.dot for Sub-blocks
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def matmul_blocked_np_dot_7(A, B, C):
    n = A.shape[0]
    bs = BLOCK_SIZE
    C.fill(0.0)

    num_i_blocks = (n + bs - 1) // bs

    for b in prange(num_i_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)

        for k_block in range(0, n, bs):
            for j_block in range(0, n, bs):
                k_end = min(k_block + bs, n)
                j_end = min(j_block + bs, n)

                C[i_block:i_end, j_block:j_end] += np.dot(
                    A[i_block:i_end, k_block:k_end],
                    B[k_block:k_end, j_block:j_end]
                )

# ---------------------------------------------------------
# Baseline 7: Blocked Temp Copy-In / Copy-Out
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def matmul_blocked_temp_copy_8(A, B, C):
    n = A.shape[0]
    bs = BLOCK_SIZE
    C.fill(0.0)

    num_i_blocks = (n + bs - 1) // bs

    for b in prange(num_i_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)

        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)

            temp = C[i_block:i_end, j_block:j_end].copy()

            for k_block in range(0, n, bs):
                k_end = min(k_block + bs, n)

                temp += np.dot(
                    A[i_block:i_end, k_block:k_end],
                    B[k_block:k_end, j_block:j_end]
                )

            C[i_block:i_end, j_block:j_end] = temp

# ---------------------------------------------------------
# Baseline 8: Two-Level Blocked Parallel I with Temp Copy & np.dot
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def matmul_two_level_blocked_temp_np_dot_9(A, B, C):
    n = A.shape[0]
    l2 = BLOCK_SIZE_L2
    l1 = BLOCK_SIZE_L1
    C.fill(0.0)

    num_l2_i_blocks = (n + l2 - 1) // l2

    for b2 in prange(num_l2_i_blocks):
        i2_start = b2 * l2
        i2_end = min(i2_start + l2, n)

        for j2_start in range(0, n, l2):
            j2_end = min(j2_start + l2, n)

            temp_l2 = C[i2_start:i2_end, j2_start:j2_end].copy()

            for k2_start in range(0, n, l2):
                k2_end = min(k2_start + l2, n)

                for i1_start in range(i2_start, i2_end, l1):
                    i1_end = min(i1_start + l1, i2_end)
                    i1_rel_start = i1_start - i2_start
                    i1_rel_end = i1_end - i2_start

                    for j1_start in range(j2_start, j2_end, l1):
                        j1_end = min(j1_start + l1, j2_end)
                        j1_rel_start = j1_start - j2_start
                        j1_rel_end = j1_end - j2_start

                        temp_l1 = temp_l2[i1_rel_start:i1_rel_end, j1_rel_start:j1_rel_end].copy()

                        for k1_start in range(k2_start, k2_end, l1):
                            k1_end = min(k1_start + l1, k2_end)

                            temp_l1 += np.dot(
                                A[i1_start:i1_end, k1_start:k1_end],
                                B[k1_start:k1_end, j1_start:j1_end]
                            )

                        temp_l2[i1_rel_start:i1_rel_end, j1_rel_start:j1_rel_end] = temp_l1

            C[i2_start:i2_end, j2_start:j2_end] = temp_l2

# ---------------------------------------------------------
# Baseline 9: Zero Allocation Blocked Matmul
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def matmul_blocked_zero_alloc_8(A, B, C):
    n = A.shape[0]
    bs = BLOCK_SIZE

    num_i_blocks = (n + bs - 1) // bs

    for b in prange(num_i_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        h = i_end - i_block

        temp_tile = np.empty((bs, bs), dtype=A.dtype)
        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            w = j_end - j_block

            temp = temp_tile[:h, :w]
            temp.fill(0.0)

            for k_block in range(0, n, bs):
                k_end = min(k_block + bs, n)

                temp += np.dot(
                    A[i_block:i_end, k_block:k_end],
                    B[k_block:k_end, j_block:j_end]
                )

            C[i_block:i_end, j_block:j_end] = temp

# ---------------------------------------------------------
# Baseline 10: Reference NumPy dot
# ---------------------------------------------------------
def matmul_np_dot(A, B, C):
    np.copyto(C, np.dot(A, B))

# ---------------------------------------------------------
# Execution & Benchmarking Routine
# ---------------------------------------------------------
def run_benchmark(matrix_size=512):
    global N
    N = matrix_size

    np.random.seed(42)
    A = np.random.randn(N, N).astype(np.float64)
    B = np.random.randn(N, N).astype(np.float64)
    C = np.zeros((N, N), dtype=np.float64)

    # Compute ground truth reference solution for correctness verification
    C_expected = np.dot(A, B)
    total_flops = 2.0 * (N ** 3)

    def measure(fn, warmup=True, reps=9):
        C.fill(0.0)
        if warmup:
            fn(A, B, C)

        # Correctness check against reference matrix C_expected
        C.fill(0.0)
        fn(A, B, C)
        is_correct = np.allclose(C, C_expected, rtol=1e-5, atol=1e-5)

        start = time.perf_counter()
        for _ in range(reps):
            C.fill(0.0)
            fn(A, B, C)
        elapsed = (time.perf_counter() - start) / reps
        gflops = (total_flops / elapsed) / 1e9

        return gflops, elapsed, is_correct

    results = []

    # 0. Pure Python (Run on a smaller dimension if N is large to prevent lockups)
    N_py = min(N, 128)
    A_py, B_py, C_py = A[:N_py, :N_py].copy(), B[:N_py, :N_py].copy(), np.zeros((N_py, N_py), dtype=np.float64)
    C_py_expected = np.dot(A_py, B_py)

    t0 = time.perf_counter()
    matmul_python_0(A_py, B_py, C_py)
    py_elapsed = time.perf_counter() - t0
    py_gflops = (2.0 * (N_py ** 3) / py_elapsed) / 1e9
    py_correct = np.allclose(C_py, C_py_expected, rtol=1e-5, atol=1e-5)

    results.append(("0_python_naive", py_gflops, py_elapsed, py_correct))

    # Baseline functions list
    functions = [
        ("1_numba_naive", matmul_numba_1),
        ("2_order_ijk", matmul_2_ijk),
        ("2_order_ikj", matmul_2_ikj),
        ("2_order_jik", matmul_2_jik),
        ("2_order_jki", matmul_2_jki),
        ("2_order_kij", matmul_2_kij),
        ("2_order_kji", matmul_2_kji),
        ("3_fastmath_ikj", matmul_opt_flags_3),
        ("4_parallel_i", matmul_parallel_i_4),
        ("4_parallel_k", matmul_parallel_k_4),
        ("4_parallel_j", matmul_parallel_j_4),
        ("5_blocked_parallel_i", matmul_blocked_parallel_5),
        ("6_blocked_np_dot", matmul_blocked_np_dot_7),
        ("7_blocked_temp_copy", matmul_blocked_temp_copy_8),
        ("8_two_level_blocked_temp_np_dot", matmul_two_level_blocked_temp_np_dot_9),
        ("9_blocked_zero_alloc", matmul_blocked_zero_alloc_8),
        ("10_np_dot", matmul_np_dot),
    ]

    for name, fn in functions:
        gflops, elapsed, is_correct = measure(fn)
        results.append((name, gflops, elapsed, is_correct))

    return results

# ---------------------------------------------------------
# Formatting and Output Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Get user input for matrix dimension N
    if len(sys.argv) > 1:
        try:
            N_input = int(sys.argv[1])
        except ValueError:
            N_input = 512
    else:
        N_input = 512

    print(f"\nRunning Matrix Multiplication Benchmarks for Matrix Size {N_input}x{N_input}...\n")
    benchmark_data = run_benchmark(N_input)

    py_elapsed = benchmark_data[0][2]  # Reference execution time for pure Python naive

    # Table Header Formatting
    header = f"| {'Baseline Implementation':<33} | {'GFLOP/s':<10} | {'Abs Speedup':<12} | {'Rel Speedup':<12} | {'Correct':<8} |"
    divider = "-" * len(header)

    print(divider)
    print(header)
    print(divider)

    prev_elapsed = None

    for name, gflops, elapsed, is_correct in benchmark_data:
        abs_speedup = py_elapsed / elapsed
        rel_speedup = (prev_elapsed / elapsed) if prev_elapsed is not None else 1.0
        status = "PASS" if is_correct else "FAIL"

        print(f"| {name:<33} | {gflops:10.3f} | {abs_speedup:12.2f}x | {rel_speedup:12.2f}x | {status:<8} |")
        prev_elapsed = elapsed

    print(divider)

    # Plotting Output
    names = [row[0] for row in benchmark_data]
    gflops_vals = [row[1] for row in benchmark_data]

    plt.figure(figsize=(16, 6))
    bars = plt.bar(names, gflops_vals, color="skyblue", edgecolor="navy")

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (0.02 * max(gflops_vals)),
            f"{yval:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.ylabel("GFLOP/s (Higher is better)")
    plt.title(f"Numba Matrix Multiplication Benchmark Performance (N={N_input})")
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()