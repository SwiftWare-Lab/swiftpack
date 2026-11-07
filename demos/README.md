# Roofline demo

Two scripts that place code on a **roofline**: a chart that shows how fast a computation runs compared with the fastest this machine allows, and whether memory or arithmetic is holding it back.

- **`roofline.py`** is a standalone calculator. Give it one computation (its FLOPs, the bytes it moves and its run time, or a Python function to time). It measures this machine's roofs and writes one chart (PNG) and one table (CSV).
- **`run_roofline.py`** runs every `benchmarks/*_bench.py`, calls `roofline.py` for every baseline, and draws the class chart, one chart per kernel, a summary table and an optional web page.

Only the standard library, numpy, numba and matplotlib are used. (`matmul_demo.py` in this folder is the swiftpack decorator demo, described in the main README.)

## How the roofline is calculated

### The model

A core has two speed limits:

- **Peak compute, π** (GFLOP/s): the most floating-point operations it can do per second.
- **Memory bandwidth, β** (GB/s): the most bytes it can move between memory and the core per second.

A computation that does `F` FLOPs, moves `B` bytes and takes `t` seconds has:

```
performance            P  = F / t                    (GFLOP/s)
arithmetic intensity   AI = F / B                    (FLOP per byte)
attainable             roof(AI) = min(π, β × AI)
ridge point            ridge = π / β
% of roof              P / roof(AI)
```

- **Left of the ridge** (`AI < π / β`), the slanted line `β × AI` is the lower limit: the computation is **memory-bound**. Doing more FLOPs per byte (fusion, blocking, reuse) moves its dot right, under a higher roof.
- **Right of the ridge**, the flat line `π` is the limit: **compute-bound**. Better vectorization and more cores raise its dot.
- On log-log axes the bandwidth limit is a straight slanted line and the compute limit is flat; together they look like a roof (Williams, Waterman and Patterson, 2009).

### Worked example: GEMVER at N = 4096

Numbers from my GEMVER benchmark (PolyBench), one core:

| Quantity | Value |
|---|---|
| FLOPs, `F = 10N² + N` | 167,776,256 |
| Bytes, fused version: 3 passes over the 134 MB matrix A, vectors ignored, `B = 3 × 8N²` | 402,653,184 |
| Time of `17_fused_unroll_jam` | 0.013507 s |
| Peak π: 4.55 GHz × 4 doubles (AVX2) × 2 FMA units × 2 FLOPs | 72.8 GFLOP/s |
| Bandwidth β: measured with the `a = a + s*b` stream | 37.2 GB/s |

```
P         = 167,776,256 / 0.013507 s     = 12.42 GFLOP/s
AI        = 167,776,256 / 402,653,184    = 0.42 FLOP/byte
ridge     = 72.8 / 37.2                  = 1.96 FLOP/byte   -> 0.42 is left of it: memory-bound
roof      = min(72.8, 37.2 × 0.42)       = 15.5 GFLOP/s
% of roof = 12.42 / 15.5                 = 80%
```

- The unfused versions read A four times: `B = 32N²`, AI = 0.31, roof = 11.6 GFLOP/s. Fusing raises the roof itself.
- Under the memory roof the FLOP count cancels: `P / (β × AI) = (F/t) / (β × F/B) = B / (t × β)`. So "memory-bound, 80% of roof" depends only on bytes and time, not on how the FLOPs were counted.

`roofline.py` reproduces this from the numbers alone, no bench file needed:

```bash
python demos/roofline.py --name 17_fused_unroll_jam --flops 167776256 --bytes 402653184 --time 0.013507 --peak-gflops 72.8 --bandwidth 37.2
```

### Measuring the peak, π

**Theoretical peak** per core = clock (GHz) × float64 lanes per SIMD register × 2 FMA units × 2 FLOPs per FMA. On a Ryzen 9 5900X: 4.575 × 4 (AVX2) × 2 × 2 = 73.2 GFLOP/s per core, 878 for 12 cores. It is drawn as a faint dashed line.

**Measured peak** (the roof that is used) comes from this Numba kernel (`_peak_kernels` in `roofline.py`):

```python
@njit(fastmath=True)
def peak_serial(x, a, b, passes):      # x: 256 doubles, a = 0.999999, b = 1e-7
    n = x.shape[0]
    for _ in range(passes):
        for i in range(n):
            v = x[i]
            v = v * a + b              # written out 8 times in a row
            ...
            x[i] = v
    s = 0.0
    for i in range(n):
        s += x[i]
    return s                           # checksum
```

Why each part:

- **`v * a + b`** is one fused multiply-add (FMA): a multiply and an add in one instruction, 2 FLOPs. A CPU's peak FLOP/s counts FMAs (2 FMA units in the formula above); `fastmath=True` lets LLVM fuse the multiply and the add into one `vfmadd` instruction.
- **Chained**: each step uses the previous `v`, so the compiler cannot merge the 8 steps or skip any. Different elements `i` are independent, so the CPU can still overlap the chains of different elements.
- **8 per element**: measured, see the table below.
- **`a` and `b` are arguments, not constants.** With constants, `fastmath` folds the 8 steps at compile time and the kernel "reaches" 170–195 GFLOP/s, over 230% of what the chip can do. The script rejects any measured peak above 110% of theoretical (it uses the theoretical one instead) and warns below 40%.
- **`a = 0.999999`, `b = 1e-7`**: `x` settles at `b / (1 - a) = 0.1`, so it never overflows or turns into a slow denormal number.
- **256 doubles** = 2 KB, which stays in the 32 KB L1 cache, so memory is never the limit.
- **The checksum** is returned so the compiler cannot drop the work as unused.
- **All cores**: the parallel version gives each thread its own row of 256 doubles with `prange`.

**Why 8 FMAs per element?** Measured on the 5900X, one core:

| FMAs per element | 1 | 2 | 4 | **8** | 16 | 32 |
|---|---|---|---|---|---|---|
| GFLOP/s | 23.9 | 35.6 | 50.3 | **67.0** | 54.1 | 37.9 |
| % of theoretical 73.2 | 33% | 49% | 69% | **92%** | 74% | 52% |

- **Too few:** each pass loads `x[i]` from L1 and stores it back. With 1–4 FMAs per load and store, moving `x[i]` (and the loop bookkeeping) is the limit, not the FMA units.
- **Too many:** the FMAs of one element depend on each other. A Zen 3 FMA takes 4 cycles and 2 can start per cycle, so about 8 independent FMAs must be in flight to keep both units busy. With 8 or more FMAs, LLVM vectorizes the loop with one 4-wide vector per iteration (checked in the assembly), so the independent work must come from overlapping several loop iterations. Long chains fill the CPU's out-of-order window with waiting FMAs, fewer iterations overlap, and the FMA units sit idle.
- **8** balances the two on this CPU. The best count can differ on other CPUs.

This is the same idea as the FLOP kernel of the **Empirical Roofline Toolkit** (ERT, Lo et al. 2014): ERT repeats `beta = beta * A[i] + alpha` a chosen number of times on each element of an array and stores the result back, sweeping the count and the array size. Our kernel is ERT's 16-FLOP setting (8 FMAs) on an array that fits in L1.

### Measuring the bandwidth, β

Two streaming kernels from McCalpin's **STREAM2** benchmark, written in Numba (`_stream_kernels` in `roofline.py`):

| Kernel | Code | Bytes per element | Roof |
|---|---|---|---|
| DAXPY | `a[i] = a[i] + s * b[i]` | 24: read `a`, read `b`, write `a` | DRAM, read + write (the default) |
| SUM | `t += a[i]` | 8: read `a` | DRAM, read-only |

- DAXPY reads `a` before writing it, so there is no hidden traffic. STREAM's better-known Triad writes a third array, and the CPU first reads each line of it before storing ("write-allocate"), which the byte count does not include.
- Each array is at least 128 MiB and at least 4 × the L3, so the data really comes from DRAM. The roof is the best of 10 runs.
- **1 core**: the serial kernel. **All cores**: the `prange` kernel at 1, 2, 4, 8, ... threads, and the roof is the **best** thread count, not the most threads. On the 5900X, read + write reached 34.1 GB/s with 1 thread but 24.9 with 24; read-only reached 30.8 with 1 thread and 47.9 with 12.
- **L3**: the same DAXPY with both arrays together filling half of one L3, repeated for at least 0.2 s.

### Which roof a computation is judged against

| Question | Rule |
|---|---|
| 1 core or all cores? | Numba functions compiled with `parallel=True` use the all-core roofs; everything else the 1-core roofs. |
| DRAM or L3? | L3 if the working set is at most 0.8 × the L3 the code can use: one L3 (one CCD on AMD) for serial code, the whole chip for parallel code. Windows only reports the chip's total, so pass `--l3-mb 32` on a Ryzen 5900X. |
| Read + write or read-only? | Read-only for computations that only read their big array. If the two DRAM numbers are within 15%, they count as one roof. |

### The verdict

- `AI < ridge`: **Memory-bound**; otherwise **Compute-bound**.
- Under 10% of the roof: "far below the roof (locality or overhead)", e.g. pure Python, or a loop order with strided accesses.
- Over 100%: "above the roof (some data reused from cache)". The byte model assumed every byte comes from DRAM, but part of the data was still in cache, e.g. a 134 MB matrix that is only 2 × the L3.

### Limits

- Bytes moved come from a model (e.g. "3 passes over A", or perfect cache reuse for the N³ kernels), not from hardware counters. Tools like `perf` or LIKWID would measure them.
- The roofs belong to one machine. On a new machine (or SLURM node) they are measured again automatically.
- Measured peaks are below theoretical (clock limits with all cores busy, shared cores, power). The chart uses the measured roof because that is what Numba code reaches on this machine.

## Using roofline.py

One computation in, one chart and one table out. The first run measures the machine (about a minute); later runs reuse `machine.json`.

### Command line

```bash
python demos/roofline.py --name my_kernel --flops 1.68e8 --bytes 4.03e8 --time 0.0135   # FLOPs, bytes, seconds
python demos/roofline.py --name my_kernel --gflops 12.4 --ai 0.42 --parallel            # or GFLOP/s and AI
python demos/roofline.py --points runs.csv      # many computations on one chart
python demos/roofline.py --example              # time two small built-in Numba kernels
python demos/roofline.py                        # only this machine's roofs
```

`--points` reads a CSV file with a header row. `name, flops, bytes, seconds` (or `name, gflops, ai`) are required; `parallel`, `working_set_mb`, `read_only` (yes/no) and `unit` (FLOP or OP) are optional:

```
name,flops,bytes,seconds,parallel,working_set_mb
gemver unfused,167776256,536870912,0.018022,no,134
gemver fused,167776256,402653184,0.013507,no,134
```

| Option | Meaning |
|---|---|
| `--name`, `--flops`, `--bytes`, `--time` | label, FLOPs, bytes moved, seconds |
| `--gflops`, `--ai` | instead of `--flops`/`--time` and `--flops`/`--bytes` |
| `--parallel` | it ran on all cores (all-core roofs) |
| `--working-set-mb MB` | data it works on; the L3 roof is used if that fits in L3 |
| `--read-only` | it only reads its big array (read-only DRAM roof) |
| `--unit OP` | integer or compare work, counted as operations |
| `--out DIR`, `--title` | output folder (default `demos/roofline_output/`), chart title |

Machine options (both scripts):

| Option | Meaning |
|---|---|
| `--remeasure` | measure the roofs again |
| `--peak-gflops 1CORE[,ALL]` | set the peak; one value sets both the 1-core and the all-core roof |
| `--bandwidth`, `--read-bandwidth`, `--l3-bandwidth` | set a bandwidth roof in GB/s, same format |
| `--l3-mb MB` | L3 size of one instance, e.g. 32 per CCD on a Ryzen 5900X |
| `--peak-source theoretical` | use the theoretical peak as the roof |

### From Python

```python
import sys
sys.path.insert(0, "demos")                    # the folder with roofline.py
import numpy as np
from numba import njit
import roofline as rl

@njit(fastmath=True)
def dot(x, y):
    s = 0.0
    for i in range(x.shape[0]):
        s += x[i] * y[i]
    return s

n = 20_000_000                                 # two 160 MB arrays: far larger than L3
x, y = np.random.rand(n), np.random.rand(n)
roofs = rl.build_roofs(rl.load_machine())      # measured once, then cached
p = rl.measure(dot, x, y, flops=2 * n, bytes_moved=16 * n, name="dot", read_only=True, roofs=roofs)
print(f"{p['P']:.2f} GFLOP/s, AI {p['AI']:.3f}, {100 * p['pct']:.0f}% of roof: {p['verdict']}")
rl.plot_roofline([p], roofs, "dot_roofline.png")
rl.write_csv([p], "dot_roofline.csv")
```

- `measure(fn, *args, flops=..., bytes_moved=...)` calls `fn(*args)` once (Numba compiles), then times 9 calls and uses the mean. For a kernel that changes its inputs in place, pass `setup=`: a function that returns fresh arguments before every call (not timed).
- If you already have the numbers: `rl.analyze_point(name, roofs, flops=..., bytes_moved=..., seconds=...)`, or with `gflops=` and `ai=`.
- `parallel=`, `working_set=` (bytes) and `read_only=` pick the roof as described above. The result is a dict with `P`, `AI`, `roof`, `ridge`, `pct`, `bound` and `verdict`.

### Output

In `demos/roofline_output/` (the folder ignores itself in git):

- `roofline.png`: the chart.
- `roofline.csv`: one row per computation, with columns `name, unit, flops, bytes, seconds, gflops, ai, parallel, cores, level, peak_gflops, bandwidth_gbs, ridge, roof_gflops, pct_of_roof, bound, verdict, cpu`.
- `machine.json`: the measured roofs, reused on the next run.

## Using run_roofline.py

### What it does

- Finds every `benchmarks/*_bench.py` and runs each file in its own Python process, so a broken or slow file is skipped and reported while everything else still renders.
- In each process, BLAS libraries are limited to 1 thread and no plot window can open. The script calls the file's `run_benchmark` with the arguments it expects and reads the returned `(name, gflops, seconds, correct)` rows; it never parses printed output.
- For every baseline:
  - FLOPs = the file's own GFLOP/s × time, so each dot matches the student's own table.
  - Bytes come from the kernel table in `run_roofline.py`, or from the file's `roofline_bytes` (below).
  - Serial or parallel comes from Numba's `parallel=True`.
  - Then `roofline.py`'s `analyze_point` gives the roof, % of roof and verdict.
- **Start** = the first compiled version (family 1); **best** = the fastest version that passed its correctness check (a serial version within 1% of a parallel one wins).
- `results.json` is saved and the charts are redrawn after every kernel, so a stopped job keeps what finished.

### Commands

```bash
python demos/run_roofline.py                                   # every benchmarks/*_bench.py
python demos/run_roofline.py benchmarks/matmul_numba_bench.py  # one kernel, with its detailed chart
python demos/run_roofline.py --n 256                           # same N for every kernel (faster)
python demos/run_roofline.py --n matmul=1024                   # N for one kernel only
python demos/run_roofline.py --plot-only --html                # redraw saved results, add the web page
python demos/run_roofline.py --plot-only --all                 # every baseline on the class chart
python demos/run_roofline.py --help                            # all options
```

| Option | Meaning |
|---|---|
| `FILE_OR_FOLDER ...` | bench files or folders (default: `benchmarks/`) |
| `--n N` or `--n KERNEL=N` | problem size for every kernel, or for one kernel (default: the kernel table's N) |
| `--plot-only` | run nothing; redraw from `results.json` |
| `--timeout SEC`, `--budget SEC` | stop one file after 600 s; start no new file after 2100 s in total |
| `--all`, `--include-python` | every baseline on the class chart; also the pure-Python baselines |
| `--per-kernel` | one detailed chart per kernel (on by default for a single file) |
| `--html`, `--show` | write the interactive page; open the class chart in a window |
| `--ai KERNEL=AI` | force the arithmetic intensity of one kernel |

The machine options are the same as for `roofline.py`.

### Output

In `demos/roofline_output/`:

- `roofline_class.png` / `.svg`: the class chart, with one start → best arrow per kernel and a verdict panel.
- `roofline_<kernel>_N<n>.png` / `.svg`: one chart per kernel, drawn by `roofline.py`.
- `roofline_summary.md`: the summary table, ready to paste into a README or slides.
- `roofline_points.csv`: every baseline of every kernel, with the `roofline.csv` columns plus kernel, student, N, role (start or best), correct and bytes source.
- `roofline.html` (with `--html`): an interactive page. Hover a dot for its numbers, click a kernel to hide it, and switch between start-and-best and all baselines.
- `results.json`, `machine.json`, `logs/` (one log per bench file).

### On the cluster

Add this line to `run_bench.sh`, after `pip install`:

```bash
python3 $(pwd)/demos/run_roofline.py
```

The whole class took about 13 minutes on a 12-core desktop. `--budget 2100` starts no new file after 35 minutes, which leaves room in the 45-minute SLURM limit. If Numba reports its `workqueue` threading layer (no OpenMP or TBB found), parallel baselines run slower; the chart says so, and installing OpenMP (`libgomp1` on Ubuntu) fixes it.

### Optional: `roofline_bytes(name, n)`

A bench file can make its dots more precise by defining:

```python
def roofline_bytes(name, n):
    """Bytes moved between memory and the core by baseline `name` at size n.
    Return None to use the built-in estimate."""
    return (3 if "fused" in name else 4) * 8.0 * n * n    # e.g. fused GEMVER reads A 3 times
```

Without it, the kernel table's estimate is used for every baseline.

### Adding a kernel

One entry in `KERNELS` at the top of `run_roofline.py`, for example:

```python
"gemver": dict(label="GEMVER", student="Bart", aliases=[], n=4096,
               flops=lambda n, g: 10 * n**2 + n,    # only a sanity check
               bytes=lambda n, g: 24 * n**2,        # sets the AI
               ws=lambda n, g: 8 * n**2),           # working set: L3 or DRAM roof
```

- `n` is the default size.
- `flops` only checks the file's own count: a warning if the two differ by more than 2×.
- `g` holds the module's globals (e.g. heat-3d's `steps`).

## References

1. S. Williams, A. Waterman, D. Patterson. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM* 52(4), 2009. https://doi.org/10.1145/1498765.1498785
2. Y. J. Lo, S. Williams, B. Van Straalen, T. J. Ligocki, M. J. Cordery, N. J. Wright, M. W. Hall, L. Oliker. "Roofline Model Toolkit: A Practical Tool for Architectural and Program Analysis." PMBS 2014, LNCS 8966, pp. 129–148, 2015. https://doi.org/10.1007/978-3-319-17248-4_7. Code: https://bitbucket.org/berkeleylab/cs-roofline-toolkit
3. J. D. McCalpin. "Memory Bandwidth and Machine Balance in Current High Performance Computers." *IEEE TCCA Newsletter*, December 1995, pp. 19–25. STREAM: https://www.cs.virginia.edu/stream/. STREAM2 (FILL, COPY, DAXPY, SUM): https://www.cs.virginia.edu/stream/stream2/
4. AnandTech. "AMD Zen 3 Ryzen Deep Dive Review", 2020, page 6: FMA latency 4 cycles, 2 per clock. https://www.anandtech.com/show/16214/amd-zen-3-ryzen-deep-dive-review-5950x-5900x-5800x-and-5700x-tested/6
