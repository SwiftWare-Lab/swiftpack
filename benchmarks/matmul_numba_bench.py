import time
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np

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

                # Delegate the inner 32x32 block multiply directly to np.dot (BLAS)
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

            # Copy block of C into a temporary contiguous array
            temp = C[i_block:i_end, j_block:j_end].copy()

            for k_block in range(0, n, bs):
                k_end = min(k_block + bs, n)

                # Accumulate partial tile matrix multiplications into temp
                temp += np.dot(
                    A[i_block:i_end, k_block:k_end],
                    B[k_block:k_end, j_block:j_end]
                )

            # Copy accumulated result from temp back to C matrix
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

            # Local accumulator temp array for the L2 tile of C
            temp_l2 = C[i2_start:i2_end, j2_start:j2_end].copy()

            for k2_start in range(0, n, l2):
                k2_end = min(k2_start + l2, n)

                # Inner L1 blocking level using np.dot on sub-tiles into temp_l2
                for i1_start in range(i2_start, i2_end, l1):
                    i1_end = min(i1_start + l1, i2_end)
                    i1_rel_start = i1_start - i2_start
                    i1_rel_end = i1_end - i2_start

                    for j1_start in range(j2_start, j2_end, l1):
                        j1_end = min(j1_start + l1, j2_end)
                        j1_rel_start = j1_start - j2_start
                        j1_rel_end = j1_end - j2_start

                        for k1_start in range(k2_start, k2_end, l1):
                            k1_end = min(k1_start + l1, k2_end)

                            temp_l2[i1_rel_start:i1_rel_end, j1_rel_start:j1_rel_end] += np.dot(
                                A[i1_start:i1_end, k1_start:k1_end],
                                B[k1_start:k1_end, j1_start:j1_end]
                            )

            # Copy fully accumulated L2 tile back into destination matrix C
            C[i2_start:i2_end, j2_start:j2_end] = temp_l2


def matmul_np_dot(A, B, C):
    C.fill(0.0)
    C = np.dot(A, B)


# ---------------------------------------------------------
# Execution & Benchmarking
# ---------------------------------------------------------
def run_benchmark():
    np.random.seed(42)
    A = np.random.randn(N, N).astype(np.float64)
    B = np.random.randn(N, N).astype(np.float64)
    C = np.zeros((N, N), dtype=np.float64)

    total_flops = 2.0 * (N ** 3)

    def measure(fn, warmup=True, reps=9):
        if warmup:
            fn(A, B, C)
        start = time.perf_counter()
        for _ in range(reps):
            fn(A, B, C)
        elapsed = (time.perf_counter() - start) / reps
        return (total_flops / elapsed) / 1e9

    benchmarks = {}

    # 0. Pure Python
    N_py = 128
    A_py, B_py, C_py = A[:N_py, :N_py].copy(), B[:N_py, :N_py].copy(), C[:N_py, :N_py].copy()
    t0 = time.perf_counter()
    matmul_python_0(A_py, B_py, C_py)
    py_time = time.perf_counter() - t0
    benchmarks["0_python_naive"] = (2.0 * (N_py ** 3) / py_time) / 1e9

    # 1. Numba Naive
    benchmarks["1_numba_naive"] = measure(matmul_numba_1)

    # 2. Loop Exchanges
    benchmarks["2_order_ijk"] = measure(matmul_2_ijk)
    benchmarks["2_order_ikj"] = measure(matmul_2_ikj)
    benchmarks["2_order_jik"] = measure(matmul_2_jik)
    benchmarks["2_order_jki"] = measure(matmul_2_jki)
    benchmarks["2_order_kij"] = measure(matmul_2_kij)
    benchmarks["2_order_kji"] = measure(matmul_2_kji)

    # 3. Fastmath IKJ
    benchmarks["3_fastmath_ikj"] = measure(matmul_opt_flags_3)

    # 4. Parallel Loop Variants
    benchmarks["4_parallel_i"] = measure(matmul_parallel_i_4)
    benchmarks["4_parallel_k"] = measure(matmul_parallel_k_4)
    benchmarks["4_parallel_j"] = measure(matmul_parallel_j_4)

    # 5. Blocked Parallel I
    benchmarks["5_blocked_parallel_i"] = measure(matmul_blocked_parallel_5)

    # 6. Blocked Parallel I + np.dot
    benchmarks["6_blocked_np_dot"] = measure(matmul_blocked_np_dot_7)

    # 7. Blocked Parallel I + np.dot + temp copy
    benchmarks["7_blocked_temp_copy"] = measure(matmul_blocked_temp_copy_8)

    #8. Two-Level Blocked Parallel I + np.dot + temp copy
    benchmarks["8_two_level_blocked_temp_np_dot"] = measure(matmul_two_level_blocked_temp_np_dot_9)

    # np dot baseline
    benchmarks["np dot"] = measure(matmul_np_dot)


    return benchmarks

# ---------------------------------------------------------
# Plotting Results
# ---------------------------------------------------------
if __name__ == "__main__":
    results = run_benchmark()

    print("\n--- RESULTS (GFLOP/s) ---")
    for k, v in results.items():
        print(f"{k:25s}: {v:8.3f} GFLOP/s")

    names = list(results.keys())
    gflops = list(results.values())

    plt.figure(figsize=(16, 6))
    bars = plt.bar(names, gflops, color="skyblue", edgecolor="navy")

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.05 * max(gflops),
            f"{yval:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.ylabel("GFLOP/s (Higher is better)")
    plt.title(f"Numba Matrix Multiplication Benchmark (N={N})")
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()