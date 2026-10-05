# Matrix Multiplication example

This tutorial has two parts: the first part is code optimization where
techniques to optimize the matrix
multiplication code are discussed, and the second part discusses how
those techniques are automated by
using the `swiftpack` decorator.

## Code optimization

The optimization story behind `benchmarks/matmul_numba_bench.py` is a
classic GEMM workflow: start with a
slow reference kernel, then progressively improve locality, backend
compilation, and parallelism until the
execution pattern matches the hardware more closely.

### 0- naive baseline

The simplest implementation is the direct triple loop:

```python
def matmul_python_0(A, B, C):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            C[i, j] = 0.0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
```

This version is easy to verify and matches the mathematical definition,
but it is dominated by Python-level
loop overhead and poor memory access patterns. Each iteration performs
Python object dispatch and repeated
scalar updates, so it is orders of magnitude slower than optimized
numerical kernels.

This baseline is useful because it establishes the reference behavior:
the result is correct, but the
implementation is far from efficient.

### 1- numba optimized

The first major improvement is to move the inner computation into Numba
JIT code:

```python
@njit
def matmul_numba_1(A, B, C):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            C[i, j] = 0.0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
```

Numba compiles the loop nest to machine code, eliminating much of the
Python overhead. The algorithm is still
the same, but the kernel now runs efficiently on the CPU. This is the
point at which we see a large jump in
throughput without changing the mathematical structure.

### 2- loop exchange

The next optimization is loop reordering. The naive kernel uses
`i, j, k` order; the benchmark also compares
other permutations such as `i, k, j` and `j, i, k`.

```python
@njit
def matmul_2_ikj(A, B, C):
    n = A.shape[0]
    C.fill(0.0)
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i, j] += A[i, k] * B[k, j]
```

The benefit comes from locality and reuse. With the `i, k, j` order, the
code keeps a row of `A` and updates
multiple `C[i, j]` entries as `k` changes, while `B[k, j]` is streamed
in a way that better matches cache usage.
This reduces the penalty from poor memory locality and often improves
the compiled kernel substantially.

As a general rule, loop order matters because the innermost loop
determines the stride pattern of memory
access. Reordering loops is often the simplest way to improve a
cache-friendly GEMM kernel.

### 3- compiler flags

Once the loop nest is structurally good, the next step is to let the
backend aggressively optimize it:

```python
@njit(fastmath=True)
def matmul_opt_flags_3(A, B, C):
    n = A.shape[0]
    C.fill(0.0)
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i, j] += A[i, k] * B[k, j]
```

`fastmath=True` enables mathematically aggressive optimizations such as
reassociation and vectorization-friendly
transformations. These can increase throughput for floating-point
kernels, especially when the code is
numerically well-behaved and the result tolerances are not overly
strict.

This optimization does not change the algorithm; it changes how the
compiler is allowed to transform the kernel.
It is a pure backend optimization, not a rewrite of the matrix formula.

### 4- parallelism

A high-level parallelism improvement is to parallelize the outer loop
with `prange`:

```python
@njit(parallel=True, fastmath=True)
def matmul_parallel_i_4(A, B, C):
    n = A.shape[0]
    C.fill(0.0)
    for i in prange(n):
        for k in range(n):
            for j in range(n):
                C[i, j] += A[i, k] * B[k, j]
```

This is a safe and effective parallel strategy because different values
of `i` write to disjoint rows of `C`.
Each thread can compute one output row independently, so the work is
naturally parallelized.

The benchmark also compares other parallel choices such as `prange` over
`j` or `k`, but the `i`-parallel form is
especially natural for row-wise matrix multiplication. Parallelization
can give an additional large gain once the
single-threaded implementation is already efficient.

### 5- loop tiling

The final major optimization is loop tiling or blocking. Instead of
processing the entire matrix in one pass, we
work on sub-blocks that fit better into cache:

```python
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
```

The key idea is to split the computation into tiles of size
`BLOCK_SIZE`, so each block of `A`, `B`, and `C`
fits in cache instead of streaming huge sections of the matrices through
memory. This reduces cache misses and
keeps the arithmetic core fed with data more efficiently.

The benchmark also explores blocked variants that use `np.dot` on
submatrices and temporary buffers. These are
closer to a high-performance BLAS-style GEMM and can be even faster when
the tile size is chosen well.

The overall progression is shown 
in ![Figure 1](matmul_numba_bench_512.png) 
and can be summarized as:

1. correct but slow Python triple loop,
2. JIT compilation to remove interpreter overhead,
3. loop reordering to improve locality,
4. backend compiler flags for vectorization and reassociation,
5. parallel outer loops with `prange`, and
6. cache-aware tiling to reduce memory traffic.

This progression is the core optimization story behind matrix
multiplication in Numba and is exactly the pattern
that `swiftpack` tries to automate.

## Automation
