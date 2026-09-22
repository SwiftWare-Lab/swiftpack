# Swift-Pack

Swift-Pack explores compact tensor and matrix transformations in Python, with a focus 
on Numba-based scheduling, tiling, and parallel execution.

## Requirements

- Python 3.10+
- `numpy`
- `numba`
- `matplotlib` for the benchmark plots

## Installation

The project can be installed in editable mode from the repository root:

```bash
python -m pip install -e .
```

For running the benchmark and demo exactly as shipped in this repo, install the extra runtime packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

That file adds `scipy` and `matplotlib` on top of the core package dependencies from `pyproject.toml`.

## Running the benchmark

The main benchmark compares several matrix-multiplication implementations, including pure Python, Numba, blocked versions, and `numpy.dot`.

Run it from the project root:

```bash
python benchmarks/matmul_numba_bench.py
```

You can optionally pass the matrix size as the first argument. If omitted, the script defaults to `512`:

```bash
python benchmarks/matmul_numba_bench.py 256
```

The benchmark prints a performance table and opens a plot window at the end. If you are on a headless machine, you may need to configure a non-interactive Matplotlib backend before running it.

## Running the example demo

The demo shows how to use the `swiftpack` decorator with a scheduler strategy from `swiftpack.scheduler`:

```bash
python demos/matmul_demo.py
```

The script:

1. creates random matrices,
2. decorates a naive matrix multiplication function with `@swiftpack(...)`,
3. compiles and runs the transformed function, and
4. verifies the result with `numpy.dot`.

If the demo succeeds, it prints:

```text
Result verified successfully against NumPy!
```

## Quick usage example

You can also import the package directly in your own code:

```python
import numpy as np
from swiftpack import swiftpack, AutoTileAndParallelizeScheduler


@swiftpack(scheduler=AutoTileAndParallelizeScheduler(), parallel=True, fastmath=True)
def matmul(A, B, C):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]


N = 256
A = np.random.randn(N, N).astype(np.float64)
B = np.random.randn(N, N).astype(np.float64)
C = np.zeros((N, N), dtype=np.float64)

matmul(A, B, C)
```

## Task list

- [ ] The current GEMM is ~2 times slower than `np.dot`
  - [ ] micro-kernel in Numba
  - [ ] L2/L3 caching and tuning
- [ ] Add other benchmarks
  - [ ] stencil
  - [ ] others
