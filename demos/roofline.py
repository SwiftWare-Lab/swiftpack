"""
Class roofline for the ECE 752 SwiftPack benchmarks.

Runs every benchmarks/*_bench.py in its own subprocess, measures this machine's
roofs (peak GFLOP/s and memory bandwidth), and draws one class roofline that
marks each kernel memory-bound or compute-bound.

    python demos/roofline.py                                   # whole class
    python demos/roofline.py benchmarks/gemver_numba_bench.py  # one student, detailed chart
    python demos/roofline.py --n 4096                          # same N for every kernel
    python demos/roofline.py --n gemver=4096 --n 2mm=256       # per-kernel N
    python demos/roofline.py --plot-only --all --html          # redraw from saved results
    python demos/roofline.py --peak-gflops 72.8 --bandwidth 37.2 --plot-only

Output goes to demos/roofline_output/ (change with --out). Only the standard
library, numpy, numba and matplotlib are used.

# Optional: make your dots more precise
# - Add roofline_bytes(name, n) to your bench file.
# - Return how many bytes that baseline moves at size n (e.g. fused
#   versions that read a matrix fewer times move fewer bytes).
# - Return None to use the built-in estimate. Without the function the
#   built-in estimate is used for every baseline.
"""

import argparse
import ast
import datetime
import importlib.util
import inspect
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "demos" / "roofline_output"

# Library thread pins for every worker
# - Must be set before Python starts: setting them after `import numpy` has no effect.
# - They do not limit Numba's own threads (gemver sets the same and still runs 24).
BLAS_ENV = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


# ---------------------------------------------------------
# Kernel table
# - One entry per PolyBench kernel; adding a kernel is one entry.
# - All dimensions are N unless the entry says otherwise, 8 bytes per double.
# - Formulas are f(n, g): g holds module globals saved by the worker
#   (heat3d's `steps`, syrk's `K`), so --plot-only needs no module.
# - flops is only for a sanity warning; the plotted FLOPs are the student's own.
# - bytes assume perfect cache reuse (what a well-blocked version achieves).
# - ws = working set, used to pick the L3 or DRAM roof.
# - Default N: measured on a Ryzen 9 5900X (24 threads), whole class in 767 s:
#   gemver 4096: 117 s, matmul 512: 70 s, 2mm 512: 153 s, heat3d 128: 100 s,
#   correlation 512: 119 s, atax 4096: 138 s, syrk 512: 47 s, roofs: 11 s.
#   (atax 8192 removes its cache reuse, 86% of roof instead of 126%, but takes 532 s.)
# ---------------------------------------------------------
def _g(g, name, default):
    try:
        return float(g[name])
    except (KeyError, TypeError, ValueError):
        return float(default)


def _call_syrk(mod, n):
    # - Old syrk: run_benchmark(matrix_size=(N, K)).
    # - refactor_syrk: run_benchmark(matrix_size, alpha, beta, trans, uplo),
    #   all required; values are the defaults from its __main__.
    params = _required_params(mod.run_benchmark)
    if len(params) >= 5:
        return mod.run_benchmark((n, n // 2), 1.25, -0.4, "N", "U")
    return mod.run_benchmark((n, n // 2))


KERNELS = {
    "matmul": dict(label="MATMUL", student="Professor", aliases=["gemm"], n=512,
                   flops=lambda n, g: 2 * n**3,
                   bytes=lambda n, g: 32 * n**2,      # read A, B; read and write C
                   ws=lambda n, g: 24 * n**2),
    "gemver": dict(label="GEMVER", student="Bart", aliases=[], n=4096,
                   flops=lambda n, g: 10 * n**2 + n,
                   bytes=lambda n, g: 24 * n**2,      # 3 passes over A; the hook refines this
                   ws=lambda n, g: 8 * n**2),
    "2mm": dict(label="2MM", student="Yue", aliases=["_2mm", "twomm"], n=512,
                flops=lambda n, g: 4 * n**3 + 2 * n**2,
                bytes=lambda n, g: 56 * n**2,         # A, B, C; tmp written and re-read; D read and written
                ws=lambda n, g: 40 * n**2),
    "heat3d": dict(label="HEAT-3D", student="Nolan", aliases=["heat_3d", "heat-3d"], n=128,
                   flops=lambda n, g: 13 * _g(g, "steps", 20) * n**3,
                   # - each sweep reads one grid and writes the other
                   # - T_next is only written, so the CPU first reads each line for
                   #   ownership (write-allocate): 8 + 8 + 8 = 24 bytes per point
                   bytes=lambda n, g: 24 * _g(g, "steps", 20) * n**3,
                   ws=lambda n, g: 16 * n**3),
    # traffic="read": A is only read, so it is judged against the read-only DRAM roof
    "atax": dict(label="ATAX", student="Ethan", aliases=[], n=4096, traffic="read",
                 flops=lambda n, g: 4 * n**2,
                 bytes=lambda n, g: 8 * n**2,         # A read once (split versions read it twice)
                 ws=lambda n, g: 8 * n**2),
    "syrk": dict(label="SYRK", student="Junwei", aliases=[], n=512, call=_call_syrk,
                 flops=lambda n, g: n**2 * _g(g, "K", n // 2),
                 bytes=lambda n, g: 8 * n * _g(g, "K", n // 2) + 8 * n**2,   # A (N x K); half of C read and written
                 ws=lambda n, g: 8 * n * _g(g, "K", n // 2) + 8 * n**2),
    "syr2k": dict(label="SYR2K", student="Hammad", aliases=[], n=512,
                  flops=lambda n, g: 2 * n**3,
                  bytes=lambda n, g: 24 * n**2,       # A, B; lower half of C read and written
                  ws=lambda n, g: 24 * n**2),
    "correlation": dict(label="CORRELATION", student="Dima", aliases=["corr"], n=512,
                        flops=lambda n, g: n**3,
                        bytes=lambda n, g: 24 * n**2,  # data read and written; corr written
                        ws=lambda n, g: 16 * n**2),
    "floyd_warshall": dict(label="FLOYD-WARSHALL", student="Malko-Bani",
                           aliases=["floydwarshall", "fw"], n=512, unit="OP",
                           flops=lambda n, g: 2 * n**3,  # add + compare, counted as operations
                           bytes=lambda n, g: 16 * n**3,  # the path matrix streams once per k step
                           ws=lambda n, g: 8 * n**2),
}

LEVEL_NAMES = {"dram": "DRAM", "dram_read": "DRAM read-only", "l3": "L3"}

# Numba falls back to its simple 'workqueue' layer when neither OpenMP nor TBB is installed
# - Seen on a fresh Ubuntu (WSL): nested prange baselines (correlation, syrk) ran past 900 s.
WORKQUEUE_NOTE = ("Numba ran on its 'workqueue' threading layer (no OpenMP or TBB found), so parallel "
                  "baselines are slower than usual; on Linux, installing libgomp1 enables OpenMP.")

PAR_RE = re.compile(r"(^|_)par(allel)?(_|$)", re.IGNORECASE)
FAMILY_RE = re.compile(r"^v?(\d+)_")


def _norm(s):
    return s.lower().replace("-", "").replace("_", "")


def kernel_key(path):
    """File name minus _numba_bench.py / _bench.py, matched against keys and aliases."""
    stem = Path(path).name
    for suffix in ("_numba_bench.py", "_bench.py", ".py"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    want = _norm(stem)
    for key, entry in KERNELS.items():
        if want == _norm(key) or want in (_norm(a) for a in entry["aliases"]):
            return key
    return stem.lower()


def family(name):
    m = FAMILY_RE.match(name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------
# Worker: runs one bench file inside its own subprocess
# ---------------------------------------------------------
def _required_params(fn):
    sig = inspect.signature(fn)
    return [p for p in sig.parameters.values()
            if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]


def call_run_benchmark(mod, key, n):
    """Call run_benchmark with the argument shape this file expects."""
    entry = KERNELS.get(key, {})
    if entry.get("call"):
        return entry["call"](mod, n)
    run = mod.run_benchmark
    params = [p for p in inspect.signature(run).parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    required = [p for p in params if p.default is p.empty]
    if len(required) == 1:
        return run(n)
    if len(required) == 2:
        return run(n, n)                       # e.g. atax: run_benchmark(M_input, N_input)
    if not required and params:
        default = params[0].default
        if isinstance(default, tuple) and len(default) == 2:
            return run((n, n // 2))
        return run(n)
    raise TypeError(f"cannot call run_benchmark{inspect.signature(run)}; "
                    f"add a 'call' entry for '{key}' in the kernel table")


def _add_pairs(found, value):
    """Add (str, callable) pairs from a list to the name -> function map."""
    if not isinstance(value, (list, tuple)):
        return
    for item in value:
        if (isinstance(item, (list, tuple)) and len(item) >= 2
                and isinstance(item[0], str) and callable(item[1])):
            found.setdefault(item[0], item[1])


def _resolve(mod, node):
    if isinstance(node, ast.Name):
        return getattr(mod, node.id, None)
    if isinstance(node, ast.Attribute):
        base = _resolve(mod, node.value)
        return getattr(base, node.attr, None) if base is not None else None
    return None


def baseline_functions(mod, source):
    """
    Map each baseline name to its function, in this order:
    - a literal `functions = [("name", fn), ...]` inside run_benchmark
    - `functions = some_global_list`
    - any module-level list of (str, callable) pairs (refactor_syrk's FUNCTIONS)
    """
    found = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    run = None
    if tree is not None:
        run = next((node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "run_benchmark"), None)
    if run is not None:
        for node in ast.walk(run):
            if not (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "functions" for t in node.targets)):
                continue
            value = node.value
            if isinstance(value, (ast.List, ast.Tuple)):
                for elt in value.elts:
                    if (isinstance(elt, (ast.Tuple, ast.List)) and len(elt.elts) >= 2
                            and isinstance(elt.elts[0], ast.Constant)
                            and isinstance(elt.elts[0].value, str)):
                        fn = _resolve(mod, elt.elts[1])
                        if callable(fn):
                            found.setdefault(elt.elts[0].value, fn)
            elif isinstance(value, ast.Name):
                _add_pairs(found, getattr(mod, value.id, None))
    for value in list(vars(mod).values()):
        _add_pairs(found, value)
    return found


def classify(name, fn):
    """(is_numba, parallel, python) for one baseline."""
    is_numba = fn is not None and hasattr(fn, "targetoptions")
    if fn is not None:
        # - Numba: parallel only if compiled with parallel=True.
        # - NumPy / SciPy / pure Python run serial (the worker pins BLAS to 1 thread).
        parallel = bool(is_numba and fn.targetoptions.get("parallel"))
    else:
        parallel = bool(PAR_RE.search(name))
    python = (not is_numba) and "python" in name.lower()
    return is_numba, parallel, python


def _coerce_rows(raw, warnings):
    rows = []
    for item in raw or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            warnings.append(f"ignored a result row of unexpected shape: {item!r}"[:200])
            continue
        name = str(item[0])
        try:
            gflops, elapsed = float(item[1]), float(item[2])
        except (TypeError, ValueError):
            warnings.append(f"{name}: non-numeric gflops/elapsed, row dropped")
            continue
        if not (math.isfinite(gflops) and math.isfinite(elapsed)) or gflops <= 0 or elapsed <= 0:
            warnings.append(f"{name}: gflops={gflops:g}, elapsed={elapsed:g}, row dropped")
            continue
        if len(item) >= 4:
            correct = bool(item[3])
        else:
            correct = True
            warnings.append(f"{name}: no is_correct field, treated as correct")
        rows.append(dict(name=name, gflops=gflops, elapsed=elapsed, correct=correct))
    return rows


def worker_main(args):
    path = Path(args._worker).resolve()
    key = kernel_key(path)
    n = int(args.n[0])
    result = dict(kernel=key, file=str(path), n=n, started=_now(), warnings=[])
    t0 = time.perf_counter()
    try:
        sys.path[:0] = [str(path.parent), str(REPO / "src")]
        source = path.read_text(encoding="utf-8", errors="replace")
        mod_name = "bench_" + re.sub(r"\W", "_", key)
        spec = importlib.util.spec_from_file_location(mod_name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException as exc:
            raise RuntimeError(f"import failed: {type(exc).__name__}: {exc}") from exc
        if not callable(getattr(mod, "run_benchmark", None)):
            raise RuntimeError("no run_benchmark() in this file")

        raw = call_run_benchmark(mod, key, n)
        rows = _coerce_rows(raw, result["warnings"])

        fns = baseline_functions(mod, source)
        hook = getattr(mod, "roofline_bytes", None)
        hook_warned = False
        for row in rows:
            fn = fns.get(row["name"]) or getattr(mod, row["name"], None)
            fn = fn if callable(fn) else None
            row["numba"], row["parallel"], row["python"] = classify(row["name"], fn)
            row["mapped"] = fn is not None
            row["hook_bytes"] = None
            if callable(hook):
                try:
                    value = hook(row["name"], n)
                    if value is not None:
                        value = float(value)
                        if not math.isfinite(value) or value <= 0:
                            raise ValueError(f"returned {value}")
                        row["hook_bytes"] = value
                except Exception as exc:
                    if not hook_warned:
                        result["warnings"].append(f"roofline_bytes() failed ({exc}); using the table")
                        hook_warned = True
        unmapped = [r["name"] for r in rows if not r["mapped"] and not r["python"]]
        if unmapped:
            result["warnings"].append("serial/parallel guessed from the name for: " + ", ".join(unmapped))
        result["rows"] = rows
        result["has_hook"] = callable(hook)
        # Module globals the byte formulas may need (heat3d `steps`, syrk `K`)
        result["globals"] = {k: v for k, v in vars(mod).items()
                             if not k.startswith("_") and type(v) in (int, float)}
        try:
            import numba
            result["numba_threads"] = int(numba.get_num_threads())
            try:
                result["threading_layer"] = str(numba.threading_layer())
            except Exception:
                result["threading_layer"] = None   # raises until a parallel region has run
        except Exception:
            pass
    except BaseException as exc:
        msg = str(exc) if isinstance(exc, RuntimeError) else f"{type(exc).__name__}: {exc}"
        result["error"] = msg.strip().splitlines()[0][:300] if msg.strip() else type(exc).__name__
        traceback.print_exc()
    result["wall_s"] = time.perf_counter() - t0
    _write_json(Path(args.json), result)
    return 0


# ---------------------------------------------------------
# Small helpers
# ---------------------------------------------------------
def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _write_json(path, data):
    # - Temp file + os.replace: a crash never leaves a half-written file.
    # - os.rename fails on Windows when the target exists; os.replace does not.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _run_cmd(cmd, timeout=5):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             env=dict(os.environ, LC_ALL="C"))
        return out.stdout if out.returncode == 0 else ""
    except Exception:
        return ""


def _fmt(x):
    """Short number for labels: 485, 34.6, 1.93, 0.42."""
    if x is None or not math.isfinite(x):
        return "?"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    if x >= 1:
        return f"{x:.2f}"
    return f"{x:.2g}"


# ---------------------------------------------------------
# Machine profile
# - Detect the CPU only to label the chart and compute a rough theoretical peak.
# - Measure the roofs that are actually plotted.
# ---------------------------------------------------------
def _parse_size_mb(text):
    """'64 MiB (2 instances)' -> (64.0, 2); '32768K' -> (32.0, 1)."""
    m = re.search(r"([\d.]+)\s*([KMG])i?B?", text, re.IGNORECASE)
    if not m:
        return None, 1
    value = float(m.group(1)) * {"K": 1 / 1024, "M": 1, "G": 1024}[m.group(2).upper()]
    inst = re.search(r"(\d+)\s*instance", text)
    return value, int(inst.group(1)) if inst else 1


def detect_cpu():
    info = dict(os=platform.system(), host=socket.gethostname(), cpu=platform.processor() or "unknown CPU",
                sockets=1, cores=None, threads_per_core=1, mhz=None, flags=set(),
                l3_total_mb=None, l3_instances=1, l3_source=None)
    system = platform.system()
    if system == "Linux":
        text = _run_cmd(["lscpu"])
        fields = dict(line.split(":", 1) for line in text.splitlines() if ":" in line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        if fields:
            info["cpu"] = fields.get("Model name", info["cpu"])
            info["sockets"] = int(fields.get("Socket(s)", "1") or 1)
            cps = fields.get("Core(s) per socket")
            info["cores"] = int(cps) * info["sockets"] if cps and cps.isdigit() else None
            info["threads_per_core"] = int(fields.get("Thread(s) per core", "1") or 1)
            for k in ("CPU max MHz", "CPU MHz"):
                if fields.get(k):
                    info["mhz"] = float(fields[k])
                    break
            info["flags"] = set(fields.get("Flags", "").split())
            if fields.get("L3 cache"):
                size, inst = _parse_size_mb(fields["L3 cache"])
                info["l3_total_mb"], info["l3_instances"] = size, inst
                info["l3_source"] = "lscpu"
        if info["cpu"] == "unknown CPU" or not info["flags"]:
            try:
                text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
                m = re.search(r"model name\s*:\s*(.+)", text)
                info["cpu"] = m.group(1).strip() if m else info["cpu"]
                m = re.search(r"^(flags|Features)\s*:\s*(.+)$", text, re.MULTILINE)
                info["flags"] |= set(m.group(2).split()) if m else set()
            except OSError:
                pass
    elif system == "Windows":
        # wmic is deprecated; Get-CimInstance works on Windows 10 and 11
        text = _run_cmd(["powershell", "-NoProfile", "-Command",
                         "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,"
                         "NumberOfLogicalProcessors,MaxClockSpeed,L3CacheSize | ConvertTo-Json"], timeout=10)
        try:
            procs = json.loads(text)
            procs = procs if isinstance(procs, list) else [procs]
            p = procs[0]
            info["cpu"] = (p.get("Name") or info["cpu"]).strip()
            info["sockets"] = len(procs)
            info["cores"] = sum(int(q.get("NumberOfCores") or 0) for q in procs) or None
            logical = sum(int(q.get("NumberOfLogicalProcessors") or 0) for q in procs)
            if info["cores"] and logical:
                info["threads_per_core"] = max(1, logical // info["cores"])
            info["mhz"] = float(p.get("MaxClockSpeed") or 0) or None
            if p.get("L3CacheSize"):
                # Windows only reports the chip total (65536 KB on a 5900X, not 32 MB per CCD)
                info["l3_total_mb"] = float(p["L3CacheSize"]) * len(procs) / 1024
                info["l3_instances"] = 1
                info["l3_source"] = "Win32_Processor (chip total)"
        except (ValueError, AttributeError, IndexError, TypeError):
            pass
    elif system == "Darwin":
        def sysctl(name):
            return _run_cmd(["sysctl", "-n", name]).strip()
        info["cpu"] = sysctl("machdep.cpu.brand_string") or info["cpu"]
        cores = sysctl("hw.physicalcpu")
        info["cores"] = int(cores) if cores.isdigit() else None
        l3 = sysctl("hw.l3cachesize")
        if l3.isdigit() and int(l3):
            info["l3_total_mb"], info["l3_source"] = int(l3) / 2**20, "sysctl"
        hz = sysctl("hw.cpufrequency_max")
        info["mhz"] = int(hz) / 1e6 if hz.isdigit() else None
    try:
        import cpuinfo  # py-cpuinfo, only if already installed
        ci = cpuinfo.get_cpu_info()
        info["cpu"] = ci.get("brand_raw") or info["cpu"]
        info["flags"] |= set(ci.get("flags", []))
    except Exception:
        pass
    try:
        # LLVM (shipped with Numba) knows the host's SIMD features on every OS
        import llvmlite.binding as llvm
        feats = llvm.get_host_cpu_features()
        info["flags"] |= {f for f, on in feats.items() if on}
    except Exception:
        pass
    return info


def simd_lanes(flags):
    """float64 lanes per SIMD register."""
    if "avx512f" in flags:
        return 8, "AVX-512"
    if "avx2" in flags:
        return 4, "AVX2"
    if "asimd" in flags or "neon" in flags:
        return 2, "NEON"
    return 2, "SSE2"


def allocated_cpus():
    # SLURM may give this job fewer cores than the node has
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def machine_key(info, alloc):
    return f"{info['host']} | {info['cpu']} | {alloc} cpus"


def _peak_kernels():
    from numba import njit, prange
    import numpy as np

    # - 8 chained x = x*a + b per element per pass = 8 FMAs = 16 FLOPs.
    # - a and b MUST be arguments: as literals, fastmath folds the chain and the
    #   5900X reported an impossible 194.6 GFLOP/s (266% of theoretical).
    # - 256 doubles stay in L1; returning a checksum keeps the work alive.
    @njit(fastmath=True)
    def peak_serial(x, a, b, passes):
        n = x.shape[0]
        for _ in range(passes):
            for i in range(n):
                v = x[i]
                v = v * a + b
                v = v * a + b
                v = v * a + b
                v = v * a + b
                v = v * a + b
                v = v * a + b
                v = v * a + b
                v = v * a + b
                x[i] = v
        s = 0.0
        for i in range(n):
            s += x[i]
        return s

    @njit(parallel=True, fastmath=True)
    def peak_parallel(X, a, b, passes):
        # each thread works on its own row
        T = X.shape[0]
        sums = np.zeros(T)
        for t in prange(T):
            row = X[t]
            n = row.shape[0]
            for _ in range(passes):
                for i in range(n):
                    v = row[i]
                    v = v * a + b
                    v = v * a + b
                    v = v * a + b
                    v = v * a + b
                    v = v * a + b
                    v = v * a + b
                    v = v * a + b
                    v = v * a + b
                    row[i] = v
            s = 0.0
            for i in range(n):
                s += row[i]
            sums[t] = s
        return sums.sum()

    return peak_serial, peak_parallel


def _stream_kernels():
    from numba import njit, prange

    # Bart's measure_bandwidth() kernel (benchmarks/gemver_numba_bench.py)
    # - a = a + s*b: read a, read b, write a = 24 bytes per element.
    # - a is read before it is written, so there is no hidden write-allocate traffic.
    @njit(fastmath=True)
    def triad_serial(a, b, s):
        for i in range(a.shape[0]):
            a[i] = a[i] + s * b[i]

    @njit(parallel=True, fastmath=True)
    def triad_parallel(a, b, s):
        for i in prange(a.shape[0]):
            a[i] = a[i] + s * b[i]

    # Read-only stream: s += a[i] (8 bytes per element, nothing written)
    # - Kernels that only read their big array (atax) can beat the read + write roof.
    @njit(fastmath=True)
    def read_serial(a, b, s):
        t = 0.0
        for i in range(a.shape[0]):
            t += a[i]
        return t

    @njit(parallel=True, fastmath=True)
    def read_parallel(a, b, s):
        t = 0.0
        for i in prange(a.shape[0]):
            t += a[i]
        return t

    return triad_serial, triad_parallel, read_serial, read_parallel


def _best_gbps(fn, a, b, min_time=0.0, reps=10, bytes_per_elem=24.0):
    fn(a, b, 1e-9)                      # compile + touch pages
    nbytes = bytes_per_elem * a.shape[0]
    best, total, count = float("inf"), 0.0, 0
    while count < reps or total < min_time:
        t = time.perf_counter()
        fn(a, b, 1e-9)
        dt = time.perf_counter() - t
        best, total, count = min(best, dt), total + dt, count + 1
    return nbytes / best / 1e9


def _thread_counts(max_threads, physical):
    counts, t = [], 1
    while t < max_threads:
        counts.append(t)
        t *= 2
    counts += [physical, max_threads]
    return sorted({c for c in counts if 1 <= c <= max_threads})


def measure_machine(info, l3_instance_mb, log=print):
    """Measure peak GFLOP/s, DRAM and L3 bandwidth, 1 core and all cores (~30-60 s)."""
    import numpy as np
    import numba
    from numba import set_num_threads

    max_threads = numba.get_num_threads()
    physical = max(1, min(max_threads, (info.get("physical_alloc") or max_threads)))
    m = {}

    # Peak FLOP/s
    peak_serial, peak_parallel = _peak_kernels()
    a, b = 0.999999, 1e-7               # x converges to b / (1 - a) = 0.1, no overflow
    x = np.ones(256)
    peak_serial(x, a, b, 10)
    passes = 2000
    while True:
        t = time.perf_counter()
        peak_serial(x, a, b, passes)
        dt = time.perf_counter() - t
        if dt >= 0.2:
            break
        passes = int(passes * max(2.0, 0.25 / max(dt, 1e-6)))
    best = min(_timed(peak_serial, x, a, b, passes) for _ in range(3))
    m["peak_1"] = 16.0 * 256 * passes / best / 1e9
    X = np.ones((max_threads, 256 + 8))[:, :256].copy()
    peak_parallel(X, a, b, 10)
    best = min(_timed(peak_parallel, X, a, b, passes) for _ in range(3))
    m["peak_all"] = 16.0 * 256 * passes * max_threads / best / 1e9
    log(f"  peak: {m['peak_1']:.1f} GFLOP/s 1 core, {m['peak_all']:.1f} all cores")

    # DRAM bandwidth: arrays far larger than L3
    triad_serial, triad_parallel, read_serial, read_parallel = _stream_kernels()
    l3_total = (info.get("l3_total_mb") or 32.0) * 2**20
    n = int(max(128 * 2**20, 4 * l3_total) // 8)
    big_a, big_b = np.ones(n), np.ones(n)
    m["dram_1"] = _best_gbps(triad_serial, big_a, big_b)
    by_threads = {}
    for t in _thread_counts(max_threads, physical):
        set_num_threads(t)
        by_threads[t] = _best_gbps(triad_parallel, big_a, big_b)
    set_num_threads(max_threads)
    # - The all-core roof is the best thread count, not always the most threads:
    #   on the 5900X 24 threads got LESS than 1 (27.6 vs 37.2 GB/s).
    m["dram_by_threads"] = {str(k): v for k, v in by_threads.items()}
    m["dram_all"] = max(max(by_threads.values()), m["dram_1"])
    log(f"  DRAM: {m['dram_1']:.1f} GB/s 1 core, {m['dram_all']:.1f} best of "
        + ", ".join(f"{k} thr {v:.1f}" for k, v in by_threads.items()) + "  (read + write)")
    m["dram_read_1"] = _best_gbps(read_serial, big_a, big_b, bytes_per_elem=8.0)
    read_by = {}
    for t in _thread_counts(max_threads, physical):
        set_num_threads(t)
        read_by[t] = _best_gbps(read_parallel, big_a, big_b, bytes_per_elem=8.0)
    set_num_threads(max_threads)
    m["dram_read_by_threads"] = {str(k): v for k, v in read_by.items()}
    m["dram_read_all"] = max(max(read_by.values()), m["dram_read_1"])
    del big_a, big_b
    log(f"  DRAM: {m['dram_read_1']:.1f} GB/s 1 core, {m['dram_read_all']:.1f} best of "
        + ", ".join(f"{k} thr {v:.1f}" for k, v in read_by.items()) + "  (read only)")

    # L3 bandwidth: both arrays together fill about half of one L3 instance
    n = int(l3_instance_mb * 2**20 / 4 // 8)
    small_a, small_b = np.ones(n), np.ones(n)
    m["l3_1"] = _best_gbps(triad_serial, small_a, small_b, min_time=0.2)
    m["l3_all"] = max(_best_gbps(triad_parallel, small_a, small_b, min_time=0.2), m["l3_1"])
    log(f"  L3:   {m['l3_1']:.1f} GB/s 1 core, {m['l3_all']:.1f} all cores "
        f"(2 x {n * 8 / 2**20:.1f} MiB arrays)")
    m["numba_threads"] = max_threads
    return m


def _timed(fn, *args):
    t = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t


def load_machine(out, args, need_measure, log=print):
    """Return the machine profile, measuring it if needed and allowed."""
    path = out / "machine.json"
    saved = _read_json(path, None)
    if not need_measure:
        return saved

    info = detect_cpu()
    alloc = allocated_cpus()
    info["allocated_cpus"] = alloc
    info["physical_alloc"] = max(1, alloc // max(1, info["threads_per_core"]))
    key = machine_key(info, alloc)
    lanes, simd = simd_lanes(info["flags"])
    ghz = (info["mhz"] or 0) / 1000 or None
    # Theoretical peak per core = GHz x lanes x 2 FMA units x 2 FLOPs per FMA
    theo_1 = ghz * lanes * 2 * 2 if ghz else None
    profile = dict(key=key, host=info["host"], cpu=info["cpu"], os=info["os"],
                   sockets=info["sockets"], cores=info["cores"],
                   threads_per_core=info["threads_per_core"], allocated_cpus=alloc,
                   physical_alloc=info["physical_alloc"], ghz=ghz, simd=simd, lanes=lanes,
                   l3_total_mb=info["l3_total_mb"], l3_instances=info["l3_instances"],
                   l3_source=info["l3_source"],
                   theoretical={"1": theo_1, "all": theo_1 * info["physical_alloc"] if theo_1 else None})

    # L3 per instance (one CCD on AMD); --l3-mb wins, then a saved override
    l3_user = args.l3_mb or (saved or {}).get("l3_mb_user")
    if l3_user:
        profile["l3_mb_user"] = float(l3_user)
        profile["l3_instance_mb"] = float(l3_user)
    elif info["l3_total_mb"]:
        profile["l3_instance_mb"] = info["l3_total_mb"] / max(1, info["l3_instances"])
        if info["os"] == "Windows":
            log(f"  note: Windows only reports the chip's total L3 ({info['l3_total_mb']:.0f} MB). "
                "If it is split per CCD (AMD), pass --l3-mb with one instance's size, e.g. --l3-mb 32.")
    else:
        profile["l3_instance_mb"] = None

    # reuse the cached roofs only if they come from this machine and have every field
    if (saved and saved.get("key") == key and (saved.get("measured") or {}).get("dram_read_all")
            and not args.remeasure):
        profile["measured"] = saved["measured"]
        profile["measured_at"] = saved.get("measured_at")
        if profile != saved:
            _write_json(path, profile)
        log(f"Machine: {profile['cpu']} (cached roofs from {profile.get('measured_at')}; --remeasure to redo)")
        return profile

    log(f"Machine: {profile['cpu']}, {alloc} cpus, {simd}"
        + (f", {ghz:.2f} GHz" if ghz else "") + ". Measuring roofs (about a minute)...")
    measured = measure_machine(profile, profile["l3_instance_mb"] or 16.0, log)
    # Sanity check on the peak kernel
    # - Above 110% of theoretical the compiler removed work: use the theoretical peak.
    # - Below 40% something is off (turbo, power saving, a busy node).
    for c in ("1", "all"):
        theo = profile["theoretical"][c]
        if theo and measured[f"peak_{c}"] > 1.10 * theo:
            log(f"  warning: measured {c}-core peak {measured[f'peak_{c}']:.1f} is above 110% of "
                f"theoretical {theo:.1f}; the kernel was optimized away. Using theoretical.")
            measured[f"peak_{c}_rejected"] = measured[f"peak_{c}"]
            measured[f"peak_{c}"] = None
        elif theo and measured[f"peak_{c}"] < 0.40 * theo:
            log(f"  warning: measured {c}-core peak is only {100 * measured[f'peak_{c}'] / theo:.0f}% "
                "of theoretical.")
    profile["measured"] = measured
    profile["measured_at"] = _now()
    _write_json(path, profile)
    return profile


def _parse_pair(text):
    """'72.8' -> (72.8, 72.8); '72.8,610' -> (72.8, 610.0)."""
    parts = [float(p) for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError(text)
    return parts[0], parts[1] if len(parts) > 1 else parts[0]


def build_roofs(machine, args):
    """
    Four roof sets: {1 core, all cores} x {DRAM, L3}.
    - Measured values by default; --peak-source theoretical switches the peak.
    - A single override value sets both the 1-core and the all-core roof.
    """
    machine = machine or {}
    meas = machine.get("measured") or {}
    theo = machine.get("theoretical") or {}
    roofs = {}
    for c in ("1", "all"):
        peak, src = meas.get(f"peak_{c}"), "measured"
        if args.peak_source == "theoretical" or not peak:
            peak, src = theo.get(c), "theoretical"
        roofs[c] = dict(peak=peak, peak_src=src, theo=theo.get(c),
                        dram=meas.get(f"dram_{c}"), dram_src="measured",
                        dram_read=meas.get(f"dram_read_{c}"), dram_read_src="measured",
                        l3=meas.get(f"l3_{c}"), l3_src="measured")
    for flag, field in (("peak_gflops", "peak"), ("bandwidth", "dram"), ("read_bandwidth", "dram_read"),
                        ("l3_bandwidth", "l3")):
        value = getattr(args, flag, None)
        if value:
            one, allc = _parse_pair(value)
            roofs["1"][field], roofs["all"][field] = one, allc
            roofs["1"][field + "_src"] = roofs["all"][field + "_src"] = "user"
    for c in ("1", "all"):
        if not roofs[c]["peak"]:
            roofs[c]["peak"] = roofs["1" if c == "all" else "all"]["peak"]
        if not roofs[c]["dram"]:
            roofs[c]["dram"] = roofs["1" if c == "all" else "all"]["dram"]
    for c in ("1", "all"):
        # a read-only kernel can always do at least what the read + write stream does;
        # within 15% the two are treated as one roof (one line, one label)
        if roofs[c]["dram"] and (roofs[c]["dram_read"] or 0) < 1.15 * roofs[c]["dram"]:
            roofs[c]["dram_read"], roofs[c]["dram_read_src"] = roofs[c]["dram"], roofs[c]["dram_src"]
    roofs["l3_instance_mb"] = machine.get("l3_instance_mb")
    roofs["l3_total_mb"] = machine.get("l3_total_mb")
    return roofs


# ---------------------------------------------------------
# Metrics and verdict
# ---------------------------------------------------------
# Why the student's own FLOP count
# - F = GFLOP/s x time, so every dot keeps the GFLOP/s from the student's slides.
# - Under the memory roof the count cancels: P / (beta x AI) = B / (time x beta),
#   so "memory-bound" and "% of roof" depend only on bytes and time.
def analyze(run, roofs, ai_override=None):
    """Turn one kernel's raw rows into roofline points plus a summary row."""
    key, n = run["kernel"], run["n"]
    entry = KERNELS.get(key)
    g = run.get("globals") or {}
    warnings = list(run.get("warnings") or [])
    unit = (entry or {}).get("unit", "FLOP")
    pts = []
    for row in run.get("rows", []):
        F = row["gflops"] * row["elapsed"] * 1e9
        if ai_override:
            B, src = F / ai_override, "user"
        elif row.get("hook_bytes"):
            B, src = row["hook_bytes"], "hook"
        elif entry:
            B, src = float(entry["bytes"](n, g)), "table"
        else:
            B, src = None, None
        pts.append(dict(row, F=F, B=B, bytes_src=src, family=family(row["name"]),
                        P=row["gflops"]))

    numba_rows = [p for p in pts if not p["python"]]
    start = (next((p for p in numba_rows if p["family"] == 1), None)
             or next((p for p in numba_rows if p.get("numba")), None)
             or (numba_rows[0] if numba_rows else None))
    if start is None or start["B"] is None:
        return dict(kernel=key, n=n, run=run, points=pts, start=None, best=None, warnings=warnings,
                    status="AI unknown, add a table entry or roofline_bytes()" if start else "no compiled baselines")

    ws = float(entry["ws"](n, g)) if entry else None
    # L3 roof only if the working set fits comfortably in the L3 the baseline can use
    # - serial: one L3 instance (one CCD on AMD)
    # - parallel: all instances (threads on both CCDs share the work)
    l3_one = roofs.get("l3_instance_mb")
    l3_all = max(roofs.get("l3_total_mb") or 0, l3_one or 0) or None
    for p in pts:
        # Python rows (capped at N=128 in some files) take the start row's AI
        p["AI"] = start["F"] / start["B"] if p["python"] else p["F"] / p["B"]
        cores = "all" if p["parallel"] else "1"
        cap = l3_all if p["parallel"] else l3_one
        level = "l3" if (ws is not None and cap and ws <= 0.8 * cap * 2**20) else "dram"
        if level == "dram" and (entry or {}).get("traffic") == "read":
            level = "dram_read"           # mostly-read kernels: read-only DRAM roof
        lvl = level if roofs[cores].get(level) else "dram"
        pi, beta = roofs[cores]["peak"], roofs[cores][lvl]
        p["cores"], p["level"] = cores, lvl
        p["roof"] = min(pi, beta * p["AI"])
        p["ridge"] = pi / beta
        p["pct"] = p["P"] / p["roof"]
        p["bound"] = "Memory-bound" if p["AI"] < p["ridge"] else "Compute-bound"
        p["far"] = p["pct"] < 0.10 or p["python"]
        p["verdict"] = p["bound"] + (", far below the roof (locality or overhead)" if p["far"] else "")
        if p["pct"] > 1.0:
            # e.g. atax at N=4096: A (134 MB) is only 2x the L3, so repeated runs reuse part of it
            p["verdict"] += ", above the roof (some data reused from cache)"
        if entry and not p["python"]:
            ref = float(entry["flops"](n, g))
            ratio = p["F"] / ref if ref else 1.0
            if not 0.5 <= ratio <= 2.0:
                warnings.append(f"{p['name']}: FLOPs are {ratio:.2f}x the table's formula "
                                "(are all dimensions N?)")

    # Best: fastest correct row; a serial row within 1% wins, so noise does not pick the roof
    correct = [p for p in pts if p["correct"] and not p["python"]]
    best = None
    if correct:
        top = max(p["P"] for p in correct)
        near = [p for p in correct if p["P"] >= 0.99 * top]
        serial = [p for p in near if not p["parallel"]]
        best = max(serial or near, key=lambda p: p["P"])
    return dict(kernel=key, n=n, run=run, points=pts, start=start, best=best, unit=unit,
                warnings=warnings, ws=ws, status="ok")


def summary_row(res):
    key = res["kernel"]
    entry = KERNELS.get(key, {})
    row = dict(kernel=entry.get("label", key.upper()), key=key, student=entry.get("student", "?"),
               n=res["n"], status=res["status"])
    start, best = res.get("start"), res.get("best")
    if res["status"] != "ok":
        return row
    row.update(start_gflops=start["P"], unit=res["unit"])
    if best:
        cores = "all cores" if best["cores"] == "all" else "1 core"
        row.update(best=best["name"], best_gflops=best["P"], speedup=best["P"] / start["P"],
                   ai=best["AI"], roof_used=f"{LEVEL_NAMES[best['level']]}, {cores}",
                   roof=best["roof"], pct=best["pct"], verdict=best["verdict"],
                   bound=best["bound"], bytes_src=best["bytes_src"])
    else:
        row.update(best="(no correct baseline)", verdict="-", bytes_src=start["bytes_src"])
    return row


# ---------------------------------------------------------
# Results cache
# ---------------------------------------------------------
def load_results(out):
    data = _read_json(out / "results.json", None)
    if not isinstance(data, dict):
        data = {}
    data.setdefault("version", 1)
    data.setdefault("runs", [])
    data.setdefault("skipped", [])
    return data


def merge_run(data, run):
    """Replace the entry with the same (kernel, N, machine key); keep the rest."""
    k = (run["kernel"], run["n"], run.get("machine_key"))
    data["runs"] = [r for r in data["runs"] if (r["kernel"], r["n"], r.get("machine_key")) != k]
    data["runs"].append(run)
    data["skipped"] = [s for s in data["skipped"] if s["kernel"] != run["kernel"]]


def merge_skip(data, skip):
    data["skipped"] = [s for s in data["skipped"] if s["kernel"] != skip["kernel"]]
    data["skipped"].append(skip)


def select_runs(data, n_args, kernels=None):
    """Newest run per kernel, unless --n picks an N."""
    n_all, n_per = parse_n(n_args)
    chosen = {}
    for run in sorted(data["runs"], key=lambda r: r.get("started", "")):
        key = run["kernel"]
        if kernels and key not in kernels:
            continue
        want = n_per.get(key, n_all)
        if want is not None and run["n"] != want:
            continue
        chosen[key] = run
    order = list(KERNELS)
    return sorted(chosen.values(), key=lambda r: (order.index(r["kernel"]) if r["kernel"] in order
                                                  else len(order), r["kernel"]))


def parse_n(n_args):
    n_all, n_per = None, {}
    for item in n_args or []:
        if "=" in str(item):
            k, v = str(item).split("=", 1)
            n_per[kernel_key(k.strip() + "_bench.py")] = int(v)
        else:
            n_all = int(item)
    return n_all, n_per


# ---------------------------------------------------------
# Runner
# ---------------------------------------------------------
def find_bench_files(paths):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += sorted(p.glob("*_bench.py"))
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: {p} not found")
    seen, unique = set(), []
    for f in files:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def run_worker(path, key, n, timeout, out):
    """Run one bench file in a fresh Python; returns (run dict or None, error or None)."""
    env = dict(os.environ)
    for var in BLAS_ENV:
        env[var] = "1"
    env["MPLBACKEND"] = "Agg"        # no window, works with no display
    env["PYTHONUTF8"] = "1"          # Windows cp1252 crashes on characters like ≈ π β
    extra = [str(Path(path).parent), str(REPO / "src")]
    env["PYTHONPATH"] = os.pathsep.join(extra + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{key}_N{n}.log"
    fd, tmp = tempfile.mkstemp(dir=str(out), prefix=f".{key}_", suffix=".json")
    os.close(fd)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker", str(path),
           "--n", str(n), "--json", tmp]
    try:
        with open(log_path, "wb") as log:
            try:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                      cwd=str(REPO), timeout=timeout)
            except subprocess.TimeoutExpired:
                return None, f"timed out after {timeout:.0f} s (see logs/{log_path.name})"
        result = _read_json(Path(tmp), None)
        if not result:
            tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            last = tail[-1][:200] if tail else f"exit code {proc.returncode}"
            return None, f"worker crashed: {last} (see logs/{log_path.name})"
        if result.get("error"):
            return None, result["error"]
        return result, None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def choose_n(key, n_args):
    n_all, n_per = parse_n(n_args)
    if key in n_per:
        return n_per[key]
    if n_all is not None:
        return n_all
    return KERNELS.get(key, {}).get("n", 512)


# ---------------------------------------------------------
# Rendering and reporting (charts are defined further down)
# ---------------------------------------------------------
def build_report(data, machine, args, kernels=None):
    roofs = build_roofs(machine, args)
    ai_force = {}
    for item in args.ai or []:
        k, v = item.split("=", 1)
        ai_force[kernel_key(k.strip() + "_bench.py")] = float(v)
    results = [analyze(run, roofs, ai_force.get(run["kernel"])) for run in select_runs(data, args.n, kernels)]
    keys = {r["kernel"] for r in results}
    skipped = [s for s in data["skipped"] if s["kernel"] not in keys
               and (not kernels or s["kernel"] in kernels)]
    mkeys = {r["run"].get("machine_key") for r in results} - {None}
    notes = []
    if len(mkeys) > 1:
        notes.append("Results from different machines are mixed on this chart: " + "; ".join(sorted(mkeys)))
    return dict(roofs=roofs, results=results, skipped=skipped, machine=machine or {}, notes=notes)


def write_summary(report, out, args):
    rows = [summary_row(r) for r in report["results"]]
    lines = ["# Roofline summary", ""]
    m = report["machine"]
    roofs = report["roofs"]
    lines.append(f"Machine: {m.get('cpu', 'unknown')} · {m.get('allocated_cpus', '?')} cpus · "
                 f"{m.get('simd', '?')} · Numba threads {m.get('measured', {}).get('numba_threads', '?')}")
    for c, label in (("1", "1 core"), ("all", "all cores")):
        r = roofs[c]
        lines.append(f"- {label}: peak {_fmt(r['peak'])} GFLOP/s ({r['peak_src']}), "
                     f"DRAM {_fmt(r['dram'])} GB/s read + write ({r['dram_src']}), "
                     f"{_fmt(r['dram_read'])} GB/s read-only ({r['dram_read_src']})"
                     + (f", L3 {_fmt(r['l3'])} GB/s ({r['l3_src']})" if r.get("l3") else "")
                     + (f", ridge {_fmt(r['peak'] / r['dram'])} FLOP/byte" if r['peak'] and r['dram'] else ""))
    lines += ["", "| Kernel | Student | N | Start | Best baseline | Best | Speedup | AI | Roof used "
              "| Roof | % of roof | Verdict | Bytes |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"| {row['kernel']} | {row['student']} | {row['n']} | | {row['status']} "
                         "| | | | | | | | |")
            continue
        u = "GOP/s" if row.get("unit") == "OP" else "GFLOP/s"
        if row.get("best_gflops"):
            lines.append(
                f"| {row['kernel']} | {row['student']} | {row['n']} | {row['start_gflops']:.2f} {u} "
                f"| {row['best']} | {row['best_gflops']:.2f} {u} | {row['speedup']:.1f}x "
                f"| {row['ai']:.2f} | {row['roof_used']} | {row['roof']:.1f} | {100 * row['pct']:.0f}% "
                f"| {row['verdict']} | {row['bytes_src']} |")
        else:
            lines.append(f"| {row['kernel']} | {row['student']} | {row['n']} | {row['start_gflops']:.2f} {u} "
                         f"| {row['best']} | | | | | | | | {row['bytes_src']} |")
    for s in report["skipped"]:
        entry = KERNELS.get(s["kernel"], {})
        lines.append(f"| {entry.get('label', s['kernel'].upper())} | {entry.get('student', '?')} "
                     f"| {s.get('n', '')} | | skipped: {s['reason']} | | | | | | | | |")
    lines.append("")
    warn_lines = []
    for r in report["results"]:
        for w in r["warnings"]:
            warn_lines.append(f"- {KERNELS.get(r['kernel'], {}).get('label', r['kernel'])}: {w}")
    if warn_lines:
        lines += ["Warnings:", ""] + warn_lines + [""]
    lines += ["Notes:", ""] + [f"- {f}" for f in footnotes(report, args)] + [f"- {n}" for n in report["notes"]]
    text = "\n".join(lines) + "\n"
    tmp = out / "roofline_summary.md.tmp"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, out / "roofline_summary.md")
    return rows


def footnotes(report, args):
    notes = ["Bytes are an estimate: perfect cache reuse for the N^3 kernels, so naive versions "
             "sit well below the roof (that gap is loop order and blocking)."]
    if any(p.get("level") == "dram_read" for r in report["results"] for p in r["points"]):
        notes.append("Kernels that only read their big array (ATAX) use the read-only DRAM roof (dash-dot); "
                     "the others read and write, like Bart's a = a + s*b stream.")
    if any(KERNELS.get(r["kernel"], {}).get("unit") == "OP" for r in report["results"]):
        notes.append("Floyd-Warshall does integer/compare work; plotted as operations per second.")
    if not args.include_python:
        notes.append("Python baselines hidden (--include-python).")
    notes.append("BLAS limited to 1 thread, so library baselines can read lower than students' own tables.")
    if any(r["run"].get("threading_layer") == "workqueue" for r in report["results"]):
        notes.append(WORKQUEUE_NOTE)
    srcs = {report["roofs"][c][f] for c in ("1", "all") for f in ("peak_src", "dram_src")}
    notes.append("Peak and bandwidth " + ("set by the user (user)." if srcs == {"user"} else
                                          "measured on this machine." if "user" not in srcs else
                                          "measured on this machine; values marked (user) were given."))
    return notes


def print_summary(rows, report):
    print()
    head = f"{'Kernel':<16} {'Student':<11} {'N':>6} {'Start':>8} {'Best':>8} {'Speedup':>8} " \
           f"{'AI':>6} {'Roof':>8} {'%roof':>6}  {'Roof used':<17} Verdict / best baseline"
    print(head)
    print("-" * len(head))
    for row in rows:
        if row["status"] != "ok" or not row.get("best_gflops"):
            print(f"{row['kernel']:<16} {row['student']:<11} {row['n']:>6}  {row.get('best', row['status'])}")
            continue
        print(f"{row['kernel']:<16} {row['student']:<11} {row['n']:>6} {row['start_gflops']:>8.2f} "
              f"{row['best_gflops']:>8.2f} {row['speedup']:>7.1f}x {row['ai']:>6.2f} {row['roof']:>8.1f} "
              f"{100 * row['pct']:>5.0f}%  {row['roof_used']:<17} {row['bound']} · {row['best']}")
    for s in report["skipped"]:
        entry = KERNELS.get(s["kernel"], {})
        print(f"{entry.get('label', s['kernel'].upper()):<16} {entry.get('student', '?'):<11} "
              f"{s.get('n', ''):>6}  skipped: {s['reason']}")
    for r in report["results"]:
        for w in r["warnings"]:
            print(f"  warning ({r['kernel']}): {w}")
    for note in report["notes"]:
        print(f"  note: {note}")


def render_all(data, machine, args, out, kernels_detail=(), kernels=None):
    report = build_report(data, machine, args, kernels)
    rows = write_summary(report, out, args)
    plotted = [r for r in report["results"] if r["status"] == "ok"]
    if plotted and report["roofs"]["1"]["peak"] and report["roofs"]["1"]["dram"]:
        draw_class_chart(report, out, args)
        for r in plotted:
            if r["kernel"] in kernels_detail:
                draw_kernel_chart(r, report, out, args)
        if args.html:
            write_html(report, rows, out, args)
    elif plotted:
        print("No roofs available: run without --plot-only once, or pass --peak-gflops and --bandwidth.")
    return report, rows


# ---------------------------------------------------------
# Charts
# - matplotlib's Figure + Agg canvas directly (no pyplot): no window can pop up.
# - Axes sit at fixed positions so the slanted roof labels keep their angle.
# - Colors live in a theme dict, so the HTML page can draw a dark version too.
# ---------------------------------------------------------
# Kernel colors, fixed by position in KERNELS (a kernel keeps its color in every run)
# - Okabe-Ito (colorblind-safe), with black and #B8A400 replaced: black vanished
#   into the roof line, and #B8A400 next to #E69F00 failed the colorblind check.
# - All 36 pairs checked under simulated red/green colorblindness: worst
#   distance 7.6 (OKLab x100), and every kernel is also labelled by name.
KERNEL_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
                 "#56B4E9", "#8800FF", "#CC0099", "#882255"]

LIGHT = dict(
    surface="#ffffff", bg_mem="#eef4fb", bg_comp="#fdf6ec", ink="#111827", ink2="#4b5563",
    muted="#6b7280", grid="#e5e7eb", faint="#9ca3af", ridge="#6d28d9", l3="#6b7280",
    kernels=KERNEL_COLORS, other="#6b7280",
    pill={"Memory-bound": ("#dbeafe", "#1e3a8a"), "Compute-bound": ("#fdecc8", "#7c4a03")},
    bar={"Memory-bound": "#2a78d6", "Compute-bound": "#eda100"},
    # ordinal ramp for baseline families: early versions light, late versions dark
    ramp=["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
          "#1c5cab", "#184f95", "#104281", "#0d366b"],
)


def kernel_color(key, theme):
    order = list(KERNELS)
    if key in order and order.index(key) < len(theme["kernels"]):
        return theme["kernels"][order.index(key)]
    return theme["other"]          # never invent a 10th hue


def _ramp_color(theme, t):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("fam", theme["ramp"])
    return cmap(min(1.0, max(0.0, t)))


def _new_figure(w, h, theme):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(w, h), dpi=100, facecolor=theme["surface"])
    FigureCanvasAgg(fig)
    return fig


def _style_axes(ax, theme, xlim, ylim):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_facecolor(theme["surface"])
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10))
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["faint"])
    ax.tick_params(colors=theme["ink2"], labelsize=9, which="both")
    ax.tick_params(which="minor", length=2, color=theme["grid"])
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)", color=theme["ink2"], fontsize=10)
    ax.set_ylabel("Performance (GFLOP/s)", color=theme["ink2"], fontsize=10)


def _limits(points, roofs, extra_x=()):
    xs = [p["AI"] for p in points] + list(extra_x)
    ys = [p["P"] for p in points]
    peak_all = roofs["all"]["peak"] or roofs["1"]["peak"]
    theo_all = roofs["all"].get("theo") or 0
    xlo = min([0.05] + [x / 2 for x in xs])
    xhi = max([100.0] + [x * 2 for x in xs])
    ylo = min([peak_all / 100] + [y / 3 for y in ys])
    yhi = max(2 * max([peak_all] + ys), 1.6 * theo_all)
    # keep ~60 px above the top roof for its label and the corner text (axes ~480 px tall)
    for _ in range(3):
        decades = math.log10(yhi / ylo)
        yhi = max(yhi, max([peak_all] + ys) * 10 ** (60 * decades / 480))
    return (xlo, xhi), (ylo, yhi)


def _text_box(artist, renderer, pad=2):
    bb = artist.get_window_extent(renderer)
    return bb.expanded(1, 1).padded(pad)


class Obstacles:
    """Things a label must not cover: earlier labels, dots (boxes) and roof lines (sampled points)."""

    def __init__(self):
        import numpy as np
        self.boxes = []
        self.labels = []
        self.pts = np.empty((0, 2))

    def append(self, box):
        self.boxes.append(box)

    def add_label(self, box):
        self.labels.append(box)

    def add_line(self, ax, x, y):
        import numpy as np
        self.pts = np.vstack([self.pts, ax.transData.transform(np.column_stack([x, y]))])

    def add_segment(self, p0, p1, n=24):
        """A straight segment in display pixels (arrows, leader lines)."""
        import numpy as np
        t = np.linspace(0, 1, n)[:, None]
        self.pts = np.vstack([self.pts, np.asarray(p0) * (1 - t) + np.asarray(p1) * t])

    def segment_hits(self, p0, p1, n=16):
        """How many earlier labels a leader line from p0 to p1 would cross."""
        hits = 0
        for b in self.labels:
            for i in range(1, n):
                t = i / n
                x, y = p0[0] * (1 - t) + p1[0] * t, p0[1] * (1 - t) + p1[1] * t
                if b.x0 < x < b.x1 and b.y0 < y < b.y1:
                    hits += 1
                    break
        return hits

    def hits(self, box):
        # covering another label is worst, then a dot or roof label, then crossing a line
        n = 10 * sum(box.overlaps(b) for b in self.labels) + 3 * sum(box.overlaps(b) for b in self.boxes)
        if len(self.pts):
            p = self.pts
            n += int(((p[:, 0] > box.x0) & (p[:, 0] < box.x1) & (p[:, 1] > box.y0) & (p[:, 1] < box.y1)).any())
        return n


def _line_angle(ax, x0, x1, y0, y1):
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))


def _slant_label(ax, text, bw, xlim, ylim, ridge, frac, above, color, size=8.5, gid=None):
    """Text written along a bandwidth roof, rotated to its on-screen slope."""
    lo, hi = math.log10(ylim[0]), math.log10(ylim[1])
    y = 10 ** (lo + frac * (hi - lo))
    x = min(max(y / bw, xlim[0] * 1.6), ridge / 2.2)
    ang = _line_angle(ax, x, x * 2, bw * x, bw * x * 2)
    off = 4 if above else -4
    dx, dy = -math.sin(math.radians(ang)) * off, math.cos(math.radians(ang)) * off
    label = ax.annotate(text, (x, bw * x), xytext=(dx, dy), textcoords="offset points", rotation=ang,
                        rotation_mode="anchor", ha="center", va="bottom" if above else "top",
                        fontsize=size, color=color, gid=gid)
    # long labels: slide along the line until the whole label is inside the axes
    renderer = ax.figure.canvas.get_renderer()
    left = ax.get_window_extent(renderer).x0 + 2
    for _ in range(30):
        if label.get_window_extent(renderer).x0 >= left or x * 1.1 > ridge / 1.5:
            break
        x *= 1.1
        label.xy = (x, bw * x)
    return label


def _draw_roofs(ax, roofs, xlim, ylim, theme, levels_used, renderer, dots=()):
    """Draw the roofs, their labels and the ridge points. Returns the obstacles for labels."""
    from matplotlib.transforms import Bbox
    # peak labels must also clear dots and the corner texts already on the axes
    dot_boxes = [_dot_box(ax, xy, 7) for xy in dots] + [_text_box(t, renderer) for t in ax.texts]
    import numpy as np
    x_all = np.logspace(math.log10(xlim[0]), math.log10(xlim[1]), 400)
    artists = []
    avoid = Obstacles()
    sets = [("all", "dram"), ("1", "dram")] + [(c, "l3") for c in ("all", "1") if "l3" in levels_used]
    # read-only DRAM roofs only when a plotted baseline uses one and it differs from read + write
    sets += [(c, "dram_read") for c in ("all", "1") if "dram_read" in levels_used
             and roofs[c].get("dram_read") and roofs[c]["dram_read"] > 1.02 * (roofs[c]["dram"] or 0)]
    for cores, lvl in sets:
        r = roofs[cores]
        bw, pi = r.get(lvl), r["peak"]
        if not bw or not pi:
            continue
        x = x_all
        if lvl in ("l3", "dram_read") and r.get("dram"):
            # past the DRAM ridge this roof is the same flat line: stop there
            x = x_all[x_all <= pi / r["dram"]]
        y = np.minimum(pi, bw * x)
        avoid.add_line(ax, x, y)
        if lvl == "dram":
            style = dict(color=theme["ink"], linewidth=2.4 if cores == "all" else 1.4,
                         linestyle="-" if cores == "all" else (0, (5, 3)),
                         alpha=1.0 if cores == "all" else 0.55)
        elif lvl == "dram_read":
            style = dict(color=theme["ink"], linewidth=1.5 if cores == "all" else 1.1,
                         linestyle=(0, (7, 2.5, 1.5, 2.5)), alpha=0.75 if cores == "all" else 0.45)
        else:
            style = dict(color=theme["l3"], linewidth=1.3, linestyle=(0, (1, 2.2)),
                         alpha=0.9 if cores == "all" else 0.7)
        ax.plot(x, y, solid_capstyle="round", zorder=3, gid=f"roof-{cores}-{lvl}", **style)
        who = "all cores" if cores == "all" else "1 core"
        src = " (user)" if r.get(lvl + "_src") == "user" else ""
        name = {"dram": "DRAM", "dram_read": "DRAM read-only", "l3": "L3"}[lvl]
        one = roofs["1"].get("dram")
        if lvl == "dram" and one and r["dram"] and abs(roofs["all"]["dram"] - one) <= 0.15 * one:
            # the two DRAM slopes nearly coincide: one label under both lines
            if cores == "1":
                continue
            src1 = " (user)" if roofs["1"].get("dram_src") == "user" else ""
            low = min(bw, one)
            text = (f"DRAM {_fmt(bw)} GB/s{src}" if abs(bw - one) < 0.05 else
                    f"DRAM {_fmt(bw)} GB/s{src} (1 core {_fmt(one)}{src1})")
            artists.append(_slant_label(ax, text, low, xlim, ylim, pi / low, 0.24, above=False,
                                        color=theme["ink2"], gid=f"rooflabel-{cores}-{lvl}"))
            continue
        # all-core label above its line, 1-core below, so the two near-parallel slopes stay readable
        frac = {("all", "dram"): 0.30, ("1", "dram"): 0.22, ("all", "l3"): 0.52, ("1", "l3"): 0.40,
                # read-only labels go near the top of their slope, clear of the DRAM labels
                ("all", "dram_read"): 0.95, ("1", "dram_read"): 0.85}[(cores, lvl)]
        artists.append(_slant_label(ax, f"{name} {_fmt(bw)} GB/s · {who}{src}", bw, xlim, ylim, pi / bw,
                                    frac, above=(cores == "all"),
                                    color=theme["ink2"] if lvl != "l3" else theme["l3"],
                                    gid=f"rooflabel-{cores}-{lvl}"))
    # ridge points (DRAM roofs), drawn before the peak labels so those can avoid them
    for cores in ("all", "1"):
        r = roofs[cores]
        if not r["peak"] or not r["dram"] or (cores == "1" and roofs["all"]["peak"] == r["peak"]
                                              and roofs["all"]["dram"] == r["dram"]):
            continue
        ridge = r["peak"] / r["dram"]
        ax.plot([ridge], [r["peak"]], marker="D", markersize=6.5, markerfacecolor=theme["surface"],
                markeredgecolor=theme["ridge"], markeredgewidth=1.5, zorder=6, gid=f"ridge-{cores}")
        rl = ax.annotate(f"ridge ≈ {_fmt(ridge)} FLOP/byte", (ridge, r["peak"]), xytext=(7, -5),
                         textcoords="offset points", ha="left", va="top", fontsize=8.5,
                         color=theme["ridge"], gid=f"ridgelabel-{cores}")
        artists.append(rl)
        dot_boxes.append(_text_box(rl, renderer))
    # flat compute roofs: label once per core count
    for cores in ("all", "1"):
        r = roofs[cores]
        if not r["peak"] or (cores == "1" and roofs["all"]["peak"] == r["peak"]):
            continue
        who = "All cores" if cores == "all" else "1 core"
        how = r["peak_src"]
        theo = r.get("theo")
        show_theo = bool(theo and how != "theoretical" and abs(theo - r["peak"]) / r["peak"] > 0.03)
        # two short lines, above the faint theoretical line when it is close
        # - right end above the line, else right end below it, else next to the ridge;
        #   the first spot that covers no dot and no other roof line wins
        above = max(r["peak"], theo) if show_theo and theo < 1.3 * r["peak"] else r["peak"]
        ridge_x = r["peak"] / r["dram"] if r.get("dram") else xlim[0]
        spots = [(xlim[1], above, -6, "right", 1), (xlim[1], r["peak"], -6, "right", -1),
                 (ridge_x * 1.15, above, 6, "left", 1)]
        for i, (x_at, y_at, dx, ha, side) in enumerate(spots):
            va = "bottom" if side > 0 else "top"
            note = ax.annotate(how + (f" · theoretical ≈ {_fmt(theo)}" if show_theo else ""),
                               (x_at, y_at), xytext=(dx, 4 * side), textcoords="offset points",
                               ha=ha, va=va, fontsize=7.5, color=theme["muted"], gid=f"peaknote-{cores}")
            main = ax.annotate(f"{who} · peak ≈ {_fmt(r['peak'])} GFLOP/s",
                               (x_at, y_at), xytext=(dx, 15 * side), textcoords="offset points",
                               ha=ha, va=va, fontsize=8.5, color=theme["ink2"], gid=f"peaklabel-{cores}")
            box = Bbox.union([_text_box(note, renderer, 0), _text_box(main, renderer, 0)])
            if (not any(box.overlaps(b) for b in dot_boxes) and not avoid.hits(box)) or i == len(spots) - 1:
                break
            note.remove()
            main.remove()
        artists += [note, main]
        if show_theo and theo < ylim[1]:
            # faint line for the theoretical peak, from where the DRAM slope reaches it
            ax.plot([theo / r["dram"], xlim[1]], [theo, theo], color=theme["faint"], linewidth=0.9,
                    linestyle=(0, (4, 3)), alpha=0.7, zorder=2, gid=f"theo-{cores}")
    for a in artists:
        avoid.append(_text_box(a, renderer))
    return avoid


def _place_label(ax, renderer, xy, lines, avoid, theme, color_line, gid=None, size=8.5):
    """
    Greedy label placer: try offsets in rings around the dot, nearest first, and
    keep the first whose text box covers no earlier label, dot or roof line and
    stays inside the axes. If none is free, keep the one with the fewest hits.
    """
    from matplotlib.transforms import Bbox
    title, detail = lines
    axbox = ax.get_window_extent(renderer)
    offsets = []
    for r in (16, 32, 52, 76, 104):
        for ux, uy in ((1, 0.75), (1, -0.75), (-1, 0.75), (-1, -0.75), (0, 1.3), (0, -1.3), (1.4, 0), (-1.4, 0)):
            offsets.append((round(r * ux), round(r * uy)))
    line_h = size * 1.35

    def make(dx, dy):
        ha = "left" if dx > 0 else "right" if dx < 0 else "center"
        va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
        # title above the detail line; both hang away from the dot
        if dy < 0:
            y1, y2 = dy, dy - line_h
        elif dy > 0:
            y1, y2 = dy + line_h, dy
        else:
            y1, y2 = line_h / 2, -line_h / 2
        t1 = ax.annotate(title, xy, xytext=(dx, y1), textcoords="offset points", ha=ha, va=va,
                         fontsize=size, fontweight="bold", color=theme["ink"], zorder=8)
        t2 = ax.annotate(detail, xy, xytext=(dx, y2), textcoords="offset points", ha=ha, va=va,
                         fontsize=size - 0.5, color=theme["ink2"], zorder=8)
        return t1, t2, Bbox.union([_text_box(t1, renderer), _text_box(t2, renderer)])

    choice, fewest = None, None
    dot = ax.transData.transform(xy)
    px = ax.figure.dpi / 72.0
    for dx, dy in offsets:
        t1, t2, box = make(dx, dy)
        t1.remove()
        t2.remove()
        inside = box.x0 >= axbox.x0 and box.x1 <= axbox.x1 and box.y0 >= axbox.y0 and box.y1 <= axbox.y1
        corner = (dot[0] + dx * px, dot[1] + dy * px)
        hits = avoid.hits(box) + 10 * avoid.segment_hits(dot, corner) + (0 if inside else 100)
        if fewest is None or hits < fewest:
            choice, fewest = (dx, dy), hits
        if hits == 0:
            break
    dx, dy = choice
    t1, t2, box = make(dx, dy)
    # thin leader line from the dot to the label's corner
    lead = ax.annotate("", xy, xytext=(dx, dy), textcoords="offset points",
                       arrowprops=dict(arrowstyle="-", color=color_line, linewidth=0.7, shrinkA=1, shrinkB=5),
                       zorder=7)
    if gid:
        t1.set_gid(gid + "-t1")
        t2.set_gid(gid + "-t2")
        lead.arrow_patch.set_gid(gid + "-lead")
    avoid.add_label(box)
    avoid.add_segment(dot, (dot[0] + dx * px, dot[1] + dy * px))
    return box


def _dot_box(ax, xy, r=6):
    from matplotlib.transforms import Bbox
    px, py = ax.transData.transform(xy)
    return Bbox([[px - r, py - r], [px + r, py + r]])


def _plotted(res, args):
    return [p for p in res["points"] if args.include_python or not p["python"]]


def _subtitle(report, sizes):
    m = report["machine"]
    nt = (m.get("measured") or {}).get("numba_threads")
    parts = [m.get("cpu", "unknown CPU")]
    if m.get("physical_alloc"):
        parts.append(f"{m['physical_alloc']} cores / {m.get('allocated_cpus')} threads")
    layers = sorted({r["run"].get("threading_layer") for r in report["results"]} - {None})
    if nt:
        parts.append(f"Numba threads {nt}" + (f" ({', '.join(layers)})" if layers else ""))
    parts.append(datetime.date.today().isoformat())
    if sizes:
        parts.append("N: " + ", ".join(sizes))
    return " · ".join(parts)


def _footer(fig, notes, theme, y0=0.018):
    fig.text(0.04, y0, "\n".join(notes), fontsize=8, color=theme["muted"], ha="left", va="bottom",
             linespacing=1.45)


def _save(fig, out, stem, dpi=200):
    for ext in ("png", "svg"):
        tmp = out / f"{stem}.tmp.{ext}"
        fig.savefig(tmp, dpi=dpi, facecolor=fig.get_facecolor())
        os.replace(tmp, out / f"{stem}.{ext}")


def class_figure(report, args, theme=LIGHT, all_points=None, gid_prefix="c"):
    """The class chart: one machine's roofs, start -> best per kernel, verdict panel."""
    from matplotlib.patches import FancyArrowPatch
    show_all = args.all if all_points is None else all_points
    results = [r for r in report["results"] if r["status"] == "ok"]
    roofs = report["roofs"]
    # SVG ids for the HTML page: <prefix>-k<kernel>-pt<row> for start/best, -xpt<row> for the rest
    for ki, r in enumerate(results):
        for j, p in enumerate(r["points"]):
            p["gid"] = f"{gid_prefix}-k{ki}-{'pt' if p is r['start'] or p is r['best'] else 'xpt'}{j}"
    fig = _new_figure(13, 7.5, theme)
    renderer = fig.canvas.get_renderer()

    pts = []
    for r in results:
        pts += [r["start"]] + ([r["best"]] if r["best"] else [])
        if show_all:
            pts += _plotted(r, args)
        elif args.include_python:
            pts += [p for p in r["points"] if p["python"]]
    xlim, ylim = _limits(pts, roofs)
    # more than five footnotes: lift the plot so the footer clears the x-axis label
    lift = 0.024 * max(0, len(footnotes(report, args)) - 5)
    ax = fig.add_axes([0.065, 0.20 + lift, 0.615, 0.66 - lift])
    _style_axes(ax, theme, xlim, ylim)

    # memory-bound / compute-bound background, split at the all-core ridge
    ridge_all = roofs["all"]["peak"] / roofs["all"]["dram"]
    ax.axvspan(xlim[0], ridge_all, color=theme["bg_mem"], zorder=0, linewidth=0, gid="bg-mem")
    ax.axvspan(ridge_all, xlim[1], color=theme["bg_comp"], zorder=0, linewidth=0, gid="bg-comp")
    corner = dict(fontsize=7.5, color=theme["muted"], transform=ax.transAxes, va="top", zorder=1)
    ax.text(0.012, 0.985, "M E M O R Y - B O U N D", ha="left", **corner)
    ax.text(0.988, 0.985, "C O M P U T E - B O U N D", ha="right", **corner)

    levels = {p["level"] for p in pts}
    avoid = _draw_roofs(ax, roofs, xlim, ylim, theme, levels, renderer, [(p["AI"], p["P"]) for p in pts])
    avoid.append(_text_box(ax.texts[0], renderer))
    avoid.append(_text_box(ax.texts[1], renderer))

    # every baseline as a small dot (--all); failed ones as x
    for r in results:
        col = kernel_color(r["kernel"], theme)
        if show_all or args.include_python:
            for p in (_plotted(r, args) if show_all else [p for p in r["points"] if p["python"]]):
                if p is r["start"] or p is r["best"]:
                    continue
                ax.plot([p["AI"]], [p["P"]], marker="x" if not p["correct"] else "o",
                        markersize=4.5 if p["correct"] else 5, color=col, alpha=0.55,
                        markeredgewidth=1.1, linestyle="none", zorder=4, gid=p.get("gid"))
                avoid.append(_dot_box(ax, (p["AI"], p["P"]), 3))

    # start (hollow) -> best (filled) with an arrow, colored per kernel
    for r in results:
        col = kernel_color(r["kernel"], theme)
        s, b = r["start"], r["best"]
        if b is None:
            # every baseline failed its check: x markers, no best and no arrow
            ax.plot([s["AI"]], [s["P"]], marker="x", markersize=8, color=col, markeredgewidth=1.8,
                    linestyle="none", zorder=6, gid=s.get("gid"))
            avoid.append(_dot_box(ax, (s["AI"], s["P"]), 6))
            continue
        ax.plot([s["AI"]], [s["P"]], marker="o", markersize=7, markerfacecolor=theme["surface"],
                markeredgecolor=col, markeredgewidth=1.6, linestyle="none", zorder=6, gid=s.get("gid"))
        avoid.append(_dot_box(ax, (s["AI"], s["P"]), 6))
        if b is not s:
            arrow = FancyArrowPatch((s["AI"], s["P"]), (b["AI"], b["P"]), arrowstyle="-|>",
                                    connectionstyle="arc3,rad=0.15", mutation_scale=11, color=col,
                                    alpha=0.6, linewidth=1.4, shrinkA=5, shrinkB=6, zorder=5,
                                    gid=f"{gid_prefix}-k{results.index(r)}-arrow")
            ax.add_patch(arrow)
            avoid.add_segment(ax.transData.transform((s["AI"], s["P"])), ax.transData.transform((b["AI"], b["P"])))
        ax.plot([b["AI"]], [b["P"]], marker="o", markersize=9, markerfacecolor=col,
                markeredgecolor=theme["surface"], markeredgewidth=1.2, linestyle="none", zorder=7,
                gid=b.get("gid"))
        avoid.append(_dot_box(ax, (b["AI"], b["P"]), 7))

    # key for the marks (lower right is the emptiest corner of a roofline)
    from matplotlib.lines import Line2D
    key = [Line2D([], [], marker="o", linestyle="none", markersize=7, markerfacecolor=theme["surface"],
                  markeredgecolor=theme["ink2"], markeredgewidth=1.6, label="start (first compiled version)"),
           Line2D([], [], marker="o", linestyle="none", markersize=9, markerfacecolor=theme["ink2"],
                  markeredgecolor=theme["surface"], label="best correct version")]
    if show_all:
        key += [Line2D([], [], marker="o", linestyle="none", markersize=4.5, color=theme["ink2"], alpha=0.55,
                       label="other baselines"),
                Line2D([], [], marker="x", linestyle="none", markersize=5, color=theme["ink2"],
                       markeredgewidth=1.1, label="failed check")]
    leg = ax.legend(handles=key, loc="lower right", fontsize=8, frameon=True, framealpha=0.92,
                    edgecolor=theme["grid"], facecolor=theme["surface"], labelcolor=theme["ink2"])
    avoid.append(leg.get_window_extent(renderer))

    # labels by the best dots, placed greedily
    for r in results:
        b = r["best"] or r["start"]
        entry = KERNELS.get(r["kernel"], {})
        unit = "GOP/s" if r["unit"] == "OP" else "GFLOP/s"
        title = f"{entry.get('label', r['kernel'].upper())} · {entry.get('student', '?')}"
        if r["best"]:
            sp = b["P"] / r["start"]["P"]
            detail = f"{b['P']:.3g} {unit} · {100 * b['pct']:.0f}% of roof · {sp:.1f}×"
        else:
            detail = "no correct baseline"
        _place_label(ax, renderer, (b["AI"], b["P"]), (title, detail), avoid, theme, theme["faint"],
                     gid=f"{gid_prefix}-k{results.index(r)}-label")

    # title and subtitle (N per kernel on its own line when the subtitle gets long)
    sizes = [f"{KERNELS.get(r['kernel'], {}).get('label', r['kernel']).lower()} {r['n']}" for r in results]
    fig.text(0.04, 0.955, "ECE 752 SwiftPack · Class Roofline", fontsize=16, fontweight="bold",
             color=theme["ink"], ha="left", va="bottom")
    sub = _subtitle(report, sizes)
    if len(sub) > 130:
        sub = _subtitle(report, []) + "\nN: " + ", ".join(sizes)
    fig.text(0.04, 0.945, sub, fontsize=10.5, color=theme["muted"], ha="left", va="top", linespacing=1.4)

    # right panel: one row per kernel
    panel = fig.add_axes([0.715, 0.20 + lift, 0.265, 0.66 - lift])
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    panel.axis("off")
    panel.text(0, 1.0, "B E S T   B A S E L I N E   P E R   K E R N E L", fontsize=7.5, color=theme["muted"],
               va="top", ha="left")
    # kernels that ran but cannot be plotted (no byte model) are listed like skips
    not_plotted = [dict(kernel=r["kernel"], reason=r["status"]) for r in report["results"] if r["status"] != "ok"]
    rows = [(r, None) for r in results] + [(None, s) for s in not_plotted + report["skipped"]]
    step = min(0.095, 0.93 / max(1, len(rows)))
    y = 0.93
    for r, s in rows:
        key = r["kernel"] if r else s["kernel"]
        entry = KERNELS.get(key, {})
        col = kernel_color(key, theme)
        name = f"{entry.get('label', key.upper())} · {entry.get('student', '?')}"
        if r:
            panel.plot([0.02], [y], marker="o", markersize=9, color=col, markeredgecolor=theme["surface"],
                       markeredgewidth=1.2, transform=panel.transAxes, clip_on=False)
            panel.text(0.07, y, name, fontsize=9.5, color=theme["ink"], va="center", ha="left",
                       fontweight="bold")
            b = r["best"]
            if b:
                bg, fg = theme["pill"][b["bound"]]
                panel.text(0.855, y, b["bound"], fontsize=8, color=fg, va="center", ha="right",
                                  bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.75", facecolor=bg,
                                            edgecolor="none"))
                panel.text(1.0, y, f"{100 * b['pct']:.0f}%", fontsize=9.5, color=theme["ink"],
                           va="center", ha="right")
                sub = f"{b['name']} · N={r['n']} · {LEVEL_NAMES[b['level']]}, " \
                      f"{'all cores' if b['cores'] == 'all' else '1 core'}"
            else:
                sub = "no correct baseline"
            panel.text(0.07, y - step * 0.36, sub, fontsize=7.5, color=theme["muted"], va="center", ha="left")
        else:
            panel.plot([0.02], [y], marker="o", markersize=8, markerfacecolor=theme["surface"],
                       markeredgecolor=theme["faint"], markeredgewidth=1.2, transform=panel.transAxes,
                       clip_on=False)
            panel.text(0.07, y, name, fontsize=9.5, color=theme["muted"], va="center", ha="left")
            reason = s["reason"] if len(s["reason"]) <= 52 else s["reason"][:50] + "…"
            prefix = "not plotted: " if s in not_plotted else "skipped: "
            panel.text(0.07, y - step * 0.36, prefix + reason, fontsize=7.5, color=theme["muted"],
                       va="center", ha="left")
        y -= step
    panel.text(1.0, 1.0, "% OF ROOF", fontsize=7.5, color=theme["muted"], va="top", ha="right")

    _footer(fig, footnotes(report, args), theme)
    return fig


def draw_class_chart(report, out, args):
    fig = class_figure(report, args)
    _save(fig, out, "roofline_class")


def kernel_figure(res, report, args, theme=LIGHT):
    """Per-kernel chart: every baseline, colored by family, % of roof bars on the right."""
    pts = _plotted(res, args)
    roofs = report["roofs"]
    entry = KERNELS.get(res["kernel"], {})
    by_perf = sorted(pts, key=lambda p: p["P"], reverse=True)
    notes = footnotes(report, args)[:1] + [n for n in footnotes(report, args)[1:]
                                           if "Python" in n or "BLAS" in n or "workqueue" in n
                                           or ("read-only" in n and any(p["level"] == "dram_read" for p in pts))]
    notes += ["Warning: " + w[:150] for w in res["warnings"][:2]]
    h = max(7.5, 2.2 + 0.19 * len(by_perf))
    fig = _new_figure(13, h, theme)
    renderer = fig.canvas.get_renderer()
    top = 1 - 0.95 / h * 1.25
    bottom = (0.75 + 0.17 * len(notes)) / h     # room for the x-axis label and the footer
    xlim, ylim = _limits(pts, roofs)
    ax = fig.add_axes([0.065, bottom, 0.56, top - bottom])
    _style_axes(ax, theme, xlim, ylim)
    ridge_all = roofs["all"]["peak"] / roofs["all"]["dram"]
    ax.axvspan(xlim[0], ridge_all, color=theme["bg_mem"], zorder=0, linewidth=0)
    ax.axvspan(ridge_all, xlim[1], color=theme["bg_comp"], zorder=0, linewidth=0)
    corner = dict(fontsize=7.5, color=theme["muted"], transform=ax.transAxes, va="top", zorder=1)
    c1 = ax.text(0.012, 0.985, "M E M O R Y - B O U N D", ha="left", **corner)
    c2 = ax.text(0.988, 0.985, "C O M P U T E - B O U N D", ha="right", **corner)
    avoid = _draw_roofs(ax, roofs, xlim, ylim, theme, {p["level"] for p in pts}, renderer,
                        [(p["AI"], p["P"]) for p in pts])
    avoid.append(_text_box(c1, renderer))
    avoid.append(_text_box(c2, renderer))

    fams = sorted({p["family"] for p in pts if p["family"] is not None})
    def fam_color(p):
        if p["family"] is None or len(fams) < 2:
            return _ramp_color(theme, 0.6)
        return _ramp_color(theme, fams.index(p["family"]) / (len(fams) - 1))

    for p in sorted(pts, key=lambda p: p["P"]):
        col = fam_color(p)
        if not p["correct"]:
            ax.plot([p["AI"]], [p["P"]], marker="x", markersize=7, color=col, markeredgewidth=1.6,
                    linestyle="none", zorder=5)
        else:
            ax.plot([p["AI"]], [p["P"]], marker="^" if p["parallel"] else "o",
                    markersize=8 if p["parallel"] else 7.5, markerfacecolor=col,
                    markeredgecolor=theme["surface"], markeredgewidth=1.1, linestyle="none", zorder=6)
        avoid.append(_dot_box(ax, (p["AI"], p["P"]), 5))

    # legend: marker shapes and the family ramp
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=theme["ink2"],
                      markeredgecolor=theme["surface"], markersize=7, label="serial"),
               Line2D([], [], marker="^", linestyle="none", markerfacecolor=theme["ink2"],
                      markeredgecolor=theme["surface"], markersize=8, label="parallel"),
               Line2D([], [], marker="x", linestyle="none", color=theme["ink2"], markersize=7,
                      markeredgewidth=1.6, label="failed check")]
    if len(fams) >= 2:
        handles += [Line2D([], [], marker="s", linestyle="none", markersize=8, markeredgecolor="none",
                           markerfacecolor=_ramp_color(theme, 0.0), label=f"family {fams[0]} (first)"),
                    Line2D([], [], marker="s", linestyle="none", markersize=8, markeredgecolor="none",
                           markerfacecolor=_ramp_color(theme, 1.0), label=f"family {fams[-1]} (last)")]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True, framealpha=0.92,
                    edgecolor=theme["grid"], facecolor=theme["surface"], labelcolor=theme["ink2"])
    avoid.append(leg.get_window_extent(renderer))

    # labels: start, best and the top 3 others
    unit = "GOP/s" if res["unit"] == "OP" else "GFLOP/s"
    chosen = []
    for p in [res["best"], res["start"]] + [q for q in by_perf if q["correct"]]:
        if p is not None and p not in chosen and len(chosen) < 5:
            chosen.append(p)
    for p in chosen:
        tag = " (best)" if p is res["best"] else " (start)" if p is res["start"] else ""
        _place_label(ax, renderer, (p["AI"], p["P"]), (p["name"] + tag,
                     f"{p['P']:.3g} {unit} · {100 * p['pct']:.0f}% of roof"), avoid, theme, theme["faint"],
                     size=8)

    title = f"{entry.get('label', res['kernel'].upper())} · {entry.get('student', '?')} · Roofline, N={res['n']}"
    fig.text(0.04, 1 - 0.35 / h, title, fontsize=16, fontweight="bold", color=theme["ink"], ha="left", va="top")
    fig.text(0.04, 1 - 0.72 / h, _subtitle(report, []), fontsize=10.5, color=theme["muted"], ha="left", va="top")

    # right panel: % of roof per baseline, sorted by performance
    bx = fig.add_axes([0.80, bottom, 0.17, top - bottom])
    names = [p["name"] for p in by_perf]
    vals = [100 * p["pct"] for p in by_perf]
    cols = [theme["bar"][p["bound"]] for p in by_perf]
    ypos = list(range(len(by_perf)))[::-1]
    bars = bx.barh(ypos, vals, height=0.62, color=cols, edgecolor="none")
    for bar, p in zip(bars, by_perf):
        if p["far"]:
            bar.set_alpha(0.45)
        if not p["correct"]:
            bar.set_hatch("///")
            bar.set_alpha(0.35)
    vmax = max(vals + [100])
    bx.set_xlim(0, vmax * 1.28)
    bx.set_ylim(-0.7, len(by_perf) - 0.3)
    bx.set_yticks(ypos)
    bx.set_yticklabels([n + (" ✗" if not p["correct"] else "") for n, p in zip(names, by_perf)],
                       fontsize=7.5 if len(names) > 22 else 8.5, color=theme["ink2"])
    for lbl, p in zip(bx.get_yticklabels(), by_perf):
        if p is res["best"]:
            lbl.set_fontweight("bold")
            lbl.set_color(theme["ink"])
    for yy, v in zip(ypos, vals):
        bx.text(v + vmax * 0.02, yy, f"{v:.0f}%", va="center", ha="left",
                fontsize=7 if len(names) > 22 else 8, color=theme["ink2"])
    bx.axvline(100, color=theme["faint"], linewidth=0.8, linestyle=(0, (3, 2)))
    bx.tick_params(axis="y", length=0)
    bx.tick_params(axis="x", colors=theme["muted"], labelsize=8)
    for side in ("top", "right", "left"):
        bx.spines[side].set_visible(False)
    bx.spines["bottom"].set_color(theme["faint"])
    bx.set_facecolor(theme["surface"])
    bx.set_title("% of roof, sorted by performance", fontsize=8.5, color=theme["muted"], loc="left")
    from matplotlib.patches import Patch
    # bar legend under the panel, clear of the bars and the 100% line
    fig.legend(handles=[Patch(color=theme["bar"]["Memory-bound"], label="memory-bound"),
                        Patch(color=theme["bar"]["Compute-bound"], label="compute-bound"),
                        Patch(color=theme["bar"]["Memory-bound"], alpha=0.45, label="faded: < 10% of roof")],
               loc="lower right", bbox_to_anchor=(0.985, 0.1 / h), fontsize=8, frameon=False,
               labelcolor=theme["ink2"], ncol=1)

    _footer(fig, notes, theme, y0=0.12 / h)
    return fig


def draw_kernel_chart(res, report, out, args):
    fig = kernel_figure(res, report, args)
    _save(fig, out, f"roofline_{res['kernel']}_N{res['n']}")


# ---------------------------------------------------------
# Interactive page (--html)
# - One self-contained file: inline SVG of the class chart, no CDN, no internet.
# - The chart keeps its light, colorblind-checked colors in both modes; the page
#   around it follows the system's light or dark setting.
# ---------------------------------------------------------
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Class Roofline</title>
<style>
:root { color-scheme: light; --page: #f6f7f9; --card: #ffffff; --ink: #111827; --ink2: #4b5563;
  --muted: #6b7280; --line: #e5e7eb; --chip: #ffffff; --chip-off: #f3f4f6; --accent: #2a78d6;
  --tip: #111827; --tip-ink: #f9fafb; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { color-scheme: dark; --page: #0d0f12; --card: #ffffff; --ink: #f0efec;
    --ink2: #c3c2b7; --muted: #9a9890; --line: #2c2c2a; --chip: #1a1a19; --chip-off: #121212;
    --accent: #3987e5; --tip: #f0efec; --tip-ink: #111827; }
}
:root[data-theme="dark"] { color-scheme: dark; --page: #0d0f12; --card: #ffffff; --ink: #f0efec;
  --ink2: #c3c2b7; --muted: #9a9890; --line: #2c2c2a; --chip: #1a1a19; --chip-off: #121212;
  --accent: #3987e5; --tip: #f0efec; --tip-ink: #111827; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 16px/1.45 system-ui, -apple-system,
  "Segoe UI", sans-serif; }
main { max-width: 1500px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 28px; margin: 0 0 4px; }
.sub { color: var(--muted); margin: 0 0 18px; }
.bar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 14px; }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.seg button { border: 0; background: var(--chip); color: var(--ink2); padding: 7px 14px; font: inherit;
  font-size: 14px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--accent); color: #fff; }
.chip { display: inline-flex; align-items: center; gap: 7px; border: 1px solid var(--line); border-radius: 999px;
  background: var(--chip); color: var(--ink); padding: 5px 12px 5px 9px; font: inherit; font-size: 14px;
  cursor: pointer; }
.chip i { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
.chip[aria-pressed="false"] { background: var(--chip-off); color: var(--muted); text-decoration: line-through; }
.chip[aria-pressed="false"] i { opacity: 0.25; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 8px;
  overflow: hidden; }
.card svg { display: block; width: 100%; height: auto; }
/* both charts stay laid out (so their dots can be measured); the other one is just hidden */
#chart { display: grid; }
#chart .view { grid-area: 1 / 1; }
#chart:not(.all) .view.a, #chart.all .view.b { visibility: hidden; }
.hit { fill: transparent; pointer-events: none; outline: none; }
svg.near { cursor: pointer; }
.hit:focus { stroke: #2a78d6; stroke-width: 1.5; }
#tip { position: fixed; pointer-events: none; background: var(--tip); color: var(--tip-ink); border-radius: 8px;
  padding: 9px 12px; font-size: 13px; line-height: 1.5; box-shadow: 0 6px 24px rgba(0,0,0,.25);
  display: none; max-width: 340px; z-index: 10; }
#tip b { font-size: 14px; }
.tablewrap { overflow-x: auto; margin-top: 22px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pill { border-radius: 999px; padding: 2px 9px; font-size: 12.5px; }
.pill.mem { background: #dbeafe; color: #1e3a8a; }
.pill.comp { background: #fdecc8; color: #7c4a03; }
.sw { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }
.notes { color: var(--muted); font-size: 13px; margin-top: 16px; }
.notes li { margin: 2px 0; }
@media (max-width: 700px) { h1 { font-size: 22px; } .chip, .seg button { font-size: 13px; } }
</style>
</head>
<body>
<main>
<h1>__TITLE__</h1>
<p class="sub">__SUB__</p>
<div class="bar">
  <div class="seg" role="group" aria-label="Points shown">
    <button id="m-best" aria-pressed="true">Start and best</button>
    <button id="m-all" aria-pressed="false">All baselines</button>
  </div>
  __CHIPS__
</div>
<div class="card" id="chart"><div class="view b">__SVG_B__</div><div class="view a">__SVG_A__</div></div>
<div class="tablewrap"><table>
<thead><tr><th>Kernel</th><th>N</th><th>Start</th><th>Best baseline</th><th>Best</th><th>Speedup</th>
<th>AI</th><th>Roof used</th><th>Roof</th><th>% of roof</th><th>Verdict</th><th>Bytes</th></tr></thead>
<tbody>__ROWS__</tbody></table></div>
<ul class="notes">__NOTES__</ul>
</main>
<div id="tip" role="tooltip"></div>
<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const chart = document.getElementById("chart");
const tip = document.getElementById("tip");
const fmt = (v, d) => v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toPrecision(d || 3);

function showTip(id, x, y) {
  const p = DATA.points[id];
  if (!p) return;
  tip.innerHTML = `<b>${p.kernel} · ${p.student}</b><br>${p.name}${p.role ? " (" + p.role + ")" : ""}<br>` +
    `N = ${p.n} · ${p.parallel ? "parallel" : "serial"} · ${p.correct ? "check passed" : "check FAILED"}<br>` +
    `${fmt(p.P)} ${p.unit}/s at AI ${fmt(p.AI, 2)} FLOP/byte<br>` +
    `Roof ${fmt(p.roof)} ${p.unit}/s (${p.level}, ${p.cores}) · ${Math.round(100 * p.pct)}% of roof<br>${p.verdict}`;
  tip.style.display = "block";
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(window.innerWidth - w - 8, x + 14) + "px";
  tip.style.top = (y - h - 12 < 4 ? y + 16 : y - h - 12) + "px";
}
function hideTip() { tip.style.display = "none"; }

// Each dot gets a focusable circle (keyboard), and the mouse picks the nearest
// visible dot within 12 px, so dots stacked on top of each other stay reachable.
function addHits(svg, prefix) {
  const scale = svg.getScreenCTM().a || 1;
  const r = 12 / scale;
  const toRoot = svg.getCTM().inverse();
  const dots = [];
  for (const id of Object.keys(DATA.points)) {
    const el = document.getElementById(prefix + "-" + id);
    if (!el) continue;
    const bb = el.getBBox();
    const m = toRoot.multiply(el.getCTM());
    const pt = new DOMPoint(bb.x + bb.width / 2, bb.y + bb.height / 2).matrixTransform(m);
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", pt.x); c.setAttribute("cy", pt.y); c.setAttribute("r", r);
    c.setAttribute("class", "hit " + id.split("-")[0]);
    c.setAttribute("tabindex", "0");
    c.setAttribute("aria-label", DATA.points[id].kernel + " " + DATA.points[id].name);
    c.addEventListener("focus", () => { const b = c.getBoundingClientRect(); showTip(id, b.right, b.top); });
    c.addEventListener("blur", hideTip);
    svg.appendChild(c);
    dots.push({ id, x: pt.x, y: pt.y, c });
  }
  svg.addEventListener("mousemove", e => {
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const lim = 12 / (svg.getScreenCTM().a || 1);
    let best = null, bestD = lim * lim;
    for (const d of dots) {
      if (d.c.style.display === "none") continue;
      const dd = (d.x - p.x) ** 2 + (d.y - p.y) ** 2;
      if (dd <= bestD) { best = d; bestD = dd; }
    }
    svg.classList.toggle("near", !!best);
    if (best) showTip(best.id, e.clientX, e.clientY); else hideTip();
  });
  svg.addEventListener("mouseleave", () => { svg.classList.remove("near"); hideTip(); });
}

function setMode(all) {
  chart.classList.toggle("all", all);
  document.getElementById("m-all").setAttribute("aria-pressed", all);
  document.getElementById("m-best").setAttribute("aria-pressed", !all);
  try { localStorage.setItem("roofline-mode", all ? "all" : "best"); } catch (e) {}
}
document.getElementById("m-all").onclick = () => setMode(true);
document.getElementById("m-best").onclick = () => setMode(false);

for (const chip of document.querySelectorAll(".chip")) {
  chip.onclick = () => {
    const on = chip.getAttribute("aria-pressed") !== "true";
    chip.setAttribute("aria-pressed", on);
    const k = chip.dataset.k;
    for (const el of chart.querySelectorAll(`[id^="b-${k}-"], [id^="a-${k}-"], .hit.${k}`)) {
      el.style.display = on ? "" : "none";
    }
  };
}

addHits(chart.querySelector(".view.b svg"), "b");
addHits(chart.querySelector(".view.a svg"), "a");
let saved = null;
try { saved = localStorage.getItem("roofline-mode"); } catch (e) {}
setMode(saved === "all");
</script>
</body>
</html>
"""


def write_html(report, rows, out, args):
    import html
    import io

    def svg_text(all_points, prefix):
        fig = class_figure(report, args, all_points=all_points, gid_prefix=prefix)
        buf = io.StringIO()
        fig.savefig(buf, format="svg", facecolor=fig.get_facecolor(), metadata={"Date": None})
        text = buf.getvalue()
        text = text[text.index("<svg"):]
        # responsive: keep the viewBox, drop the fixed size
        return re.sub(r'<svg([^>]*?) width="[^"]+" height="[^"]+"', r'<svg\1', text, count=1)

    # two charts: start and best (b), every baseline (a); the toggle swaps them
    svg_b = svg_text(False, "b")
    svg_a = svg_text(True, "a")

    results = [r for r in report["results"] if r["status"] == "ok"]
    points, chips = {}, []
    for ki, r in enumerate(results):
        entry = KERNELS.get(r["kernel"], {})
        label, student = entry.get("label", r["kernel"].upper()), entry.get("student", "?")
        unit = "GOP" if r["unit"] == "OP" else "GFLOP"
        for p in r["points"]:
            if not p.get("gid") or (p["python"] and not args.include_python):
                continue
            role = "best" if p is r["best"] else "start" if p is r["start"] else ""
            points[p["gid"].split("-", 1)[1]] = dict(kernel=label, student=student, name=p["name"], role=role, n=r["n"],
                                    P=p["P"], AI=p["AI"], roof=p["roof"], pct=p["pct"], unit=unit,
                                    level=LEVEL_NAMES[p["level"]],
                                    cores="all cores" if p["cores"] == "all" else "1 core",
                                    verdict=p["verdict"], parallel=p["parallel"], correct=p["correct"])
        chips.append(f'<button class="chip" data-k="k{ki}" aria-pressed="true"><i style="background:'
                     f'{kernel_color(r["kernel"], LIGHT)}"></i>{html.escape(label)} · {html.escape(student)}</button>')

    trs = []
    for row in rows:
        k = f'<span class="sw" style="background:{kernel_color(row["key"], LIGHT)}"></span>' \
            f'{html.escape(row["kernel"])} · {html.escape(row["student"])}'
        if row["status"] != "ok" or not row.get("best_gflops"):
            trs.append(f'<tr><td>{k}</td><td class="num">{row["n"]}</td><td colspan="10">'
                       f'{html.escape(str(row.get("best", row["status"])))}</td></tr>')
            continue
        u = "GOP/s" if row.get("unit") == "OP" else "GFLOP/s"
        pill = "mem" if row["bound"] == "Memory-bound" else "comp"
        trs.append(
            f'<tr><td>{k}</td><td class="num">{row["n"]}</td><td class="num">{row["start_gflops"]:.2f}</td>'
            f'<td>{html.escape(row["best"])}</td><td class="num">{row["best_gflops"]:.2f} {u}</td>'
            f'<td class="num">{row["speedup"]:.1f}×</td><td class="num">{row["ai"]:.2f}</td>'
            f'<td>{row["roof_used"]}</td><td class="num">{row["roof"]:.1f}</td>'
            f'<td class="num">{100 * row["pct"]:.0f}%</td>'
            f'<td><span class="pill {pill}">{html.escape(row["verdict"])}</span></td><td>{row["bytes_src"]}</td></tr>')
    for s in report["skipped"]:
        entry = KERNELS.get(s["kernel"], {})
        trs.append(f'<tr><td>{html.escape(entry.get("label", s["kernel"].upper()))} · '
                   f'{html.escape(entry.get("student", "?"))}</td><td class="num">{s.get("n", "")}</td>'
                   f'<td colspan="10">skipped: {html.escape(s["reason"])}</td></tr>')

    sizes = [f"{KERNELS.get(r['kernel'], {}).get('label', r['kernel']).lower()} {r['n']}" for r in results]
    notes = footnotes(report, args) + report["notes"] + [
        "Hover or tab to a dot for its numbers; click a kernel to hide it."]
    page = (HTML_TEMPLATE
            .replace("__TITLE__", "ECE 752 SwiftPack · Class Roofline")
            .replace("__SUB__", html.escape(_subtitle(report, sizes)))
            .replace("__CHIPS__", "\n  ".join(chips))
            .replace("__ROWS__", "\n".join(trs))
            .replace("__NOTES__", "".join(f"<li>{html.escape(n)}</li>" for n in notes))
            .replace("__DATA__", json.dumps({"points": points}).replace("<", "\\u003c"))
            .replace("__SVG_B__", svg_b)
            .replace("__SVG_A__", svg_a))
    tmp = out / "roofline.html.tmp"
    tmp.write_text(page, encoding="utf-8")
    os.replace(tmp, out / "roofline.html")


# ---------------------------------------------------------
# Command line
# ---------------------------------------------------------
HELP_EXAMPLES = """\
examples:
  python demos/roofline.py                                   every benchmarks/*_bench.py
  python demos/roofline.py benchmarks/matmul_numba_bench.py  one kernel, with its detailed chart
  python demos/roofline.py --n 256                           same N for every kernel (faster)
  python demos/roofline.py --n matmul=1024                   N for one kernel only
  python demos/roofline.py --plot-only --html                redraw saved results, add the web page
  python demos/roofline.py --plot-only --all                 every baseline on the class chart

Charts, a summary table and results.json are written to demos/roofline_output/.
A bench file can define roofline_bytes(name, n) for exact bytes (see the top of this file).
"""


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="python demos/roofline.py",
        description="Run every benchmarks/*_bench.py, measure this machine's roofs, "
                    "and draw the class roofline.",
        epilog=HELP_EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)

    run = p.add_argument_group("what to run")
    run.add_argument("paths", nargs="*", metavar="FILE_OR_FOLDER",
                     help="bench files or folders (default: benchmarks/)")
    run.add_argument("--n", action="append", metavar="N | KERNEL=N",
                     help="problem size for every kernel, or for one kernel (repeatable)")
    run.add_argument("--plot-only", action="store_true", help="run nothing; redraw from saved results")
    run.add_argument("--out", default=str(DEFAULT_OUT), metavar="DIR",
                     help="output folder (default: demos/roofline_output/)")
    # - every kernel finishes in under 160 s on a 12-core desktop, so 600 s is ample headroom
    # - 2100 s leaves 10 of SLURM's 45 minutes for run_bench.sh's venv setup and matmul run
    run.add_argument("--timeout", type=float, default=600, metavar="SEC",
                     help="stop a bench file after this many seconds (default: 600)")
    run.add_argument("--budget", type=float, default=2100, metavar="SEC",
                     help="no new bench file starts after this many seconds (default: 2100)")

    chart = p.add_argument_group("chart options")
    chart.add_argument("--all", action="store_true", help="show every baseline, not just start and best")
    chart.add_argument("--per-kernel", action="store_true", default=None,
                       help="one detailed chart per kernel (on by default for a single file)")
    chart.add_argument("--include-python", action="store_true", help="also show the pure-Python baselines")
    chart.add_argument("--html", action="store_true", help="also write the interactive roofline.html")
    chart.add_argument("--show", action="store_true", help="open the class chart in a window at the end")

    machine = p.add_argument_group("machine numbers (measured automatically; override if needed)")
    machine.add_argument("--remeasure", action="store_true", help="measure the roofs again")
    machine.add_argument("--peak-gflops", metavar="GFLOPS[,ALL]",
                         help="peak GFLOP/s: 1 core, or 1 core,all cores")
    machine.add_argument("--bandwidth", metavar="GBS[,ALL]", help="DRAM GB/s, read + write")
    machine.add_argument("--read-bandwidth", metavar="GBS[,ALL]", help="DRAM GB/s, read only")
    machine.add_argument("--l3-bandwidth", metavar="GBS[,ALL]", help="L3 GB/s")
    machine.add_argument("--l3-mb", type=float, metavar="MB",
                         help="L3 size of one instance (e.g. 32 per CCD on a Ryzen 5900X)")
    machine.add_argument("--peak-source", choices=["measured", "theoretical"], default="measured",
                         help="which peak to use as the roof (default: measured)")
    machine.add_argument("--ai", action="append", metavar="KERNEL=AI",
                         help="force the arithmetic intensity of one kernel")

    p.add_argument("--_worker", help=argparse.SUPPRESS)
    p.add_argument("--json", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args._worker:
        return worker_main(args)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # the output folder ignores itself, so the repo's .gitignore needs no change
    if not (out / ".gitignore").exists():
        (out / ".gitignore").write_text("# written by demos/roofline.py\n*\n", encoding="utf-8")
    data = load_results(out)

    explicit = [Path(p) for p in args.paths]
    single_file = len(explicit) == 1 and explicit[0].is_file()
    per_kernel = single_file if args.per_kernel is None else args.per_kernel

    if args.plot_only:
        kernels = {kernel_key(p) for p in explicit if p.is_file()} or None
        machine = load_machine(out, args, need_measure=False)
        if not data["runs"]:
            print(f"No saved results in {out / 'results.json'}. Run without --plot-only first.")
            return 1
        detail = {r["kernel"] for r in select_runs(data, args.n, kernels)} if per_kernel else ()
        report, rows = render_all(data, machine, args, out, detail, kernels)
        print_summary(rows, report)
        _print_outputs(out, args)
        _maybe_show(args)
        return 0 if any(r["status"] == "ok" for r in report["results"]) else 1

    files = find_bench_files(explicit or [REPO / "benchmarks"])
    if not files:
        print("No *_bench.py files found.\n")
        parse_args(["--help"])
        return 2

    t_start = time.time()
    machine = load_machine(out, args, need_measure=True)
    kernels = {kernel_key(f) for f in files} if explicit else None
    detail = set()
    warned_workqueue = False
    print(f"\nRunning {len(files)} bench file(s); budget {args.budget:.0f} s, timeout {args.timeout:.0f} s each.")
    for i, path in enumerate(files, 1):
        key = kernel_key(path)
        n = choose_n(key, args.n)
        left = args.budget - (time.time() - t_start)
        label = f"[{i}/{len(files)}] {key} N={n}"
        if left < 10:
            print(f"{label}: skipped, budget exhausted")
            merge_skip(data, dict(kernel=key, n=n, file=str(path), reason="budget exhausted", when=_now()))
            continue
        print(f"{label}: running {path.name} ...", flush=True)
        t0 = time.time()
        run, err = run_worker(path, key, n, min(args.timeout, left), out)
        if err:
            print(f"{label}: skipped: {err}")
            merge_skip(data, dict(kernel=key, n=n, file=str(path), reason=err, when=_now()))
        else:
            run["machine_key"] = machine.get("key")
            run["machine"] = {k: v for k, v in machine.items() if k != "measured"}
            run["roofs_measured"] = machine.get("measured")
            merge_run(data, run)
            print(f"{label}: done in {time.time() - t0:.0f} s, {len(run['rows'])} baselines")
            if run.get("threading_layer") == "workqueue" and not warned_workqueue:
                print("    warning: " + WORKQUEUE_NOTE)
                warned_workqueue = True
            for w in run["warnings"]:
                print(f"    warning: {w}")
            if per_kernel:
                detail.add(key)
        # Save and render after every kernel, so a killed job keeps what finished
        _write_json(out / "results.json", data)
        report, rows = render_all(data, machine, args, out, detail if per_kernel else (), kernels)

    print_summary(rows, report)
    _print_outputs(out, args)
    _maybe_show(args)
    return 0 if any(r["status"] == "ok" for r in report["results"]) else 1


def _print_outputs(out, args):
    print(f"\nOutputs in {out}:")
    for f in sorted(out.glob("roofline*")):
        print(f"  {f.name}")


def _maybe_show(args):
    # charts are drawn without pyplot, so --show opens the saved class chart in a window
    png = Path(args.out).resolve() / "roofline_class.png"
    if args.show and png.exists():
        import matplotlib.pyplot as plt
        img = plt.imread(str(png))
        fig = plt.figure(figsize=(13, 7.5))
        fig.add_axes([0, 0, 1, 1]).imshow(img)
        fig.axes[0].axis("off")
        plt.show()


if __name__ == "__main__":
    sys.exit(main())
