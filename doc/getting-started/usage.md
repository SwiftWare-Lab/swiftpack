# Getting Started






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

If the demo succeeds, it prints:

```text
Result verified successfully against NumPy!
```

