# Swiftpack

Swiftpack explores compact tensor and matrix transformations in 
Python, the project is in its early stages of development.

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

For details, see [documentation](https://swiftware-lab.github.io/swiftpack/).
