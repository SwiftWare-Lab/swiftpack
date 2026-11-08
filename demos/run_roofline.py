"""
Class roofline for the ECE 752 SwiftPack benchmarks.

Runs every benchmarks/*_bench.py in its own subprocess, then calls roofline.py
(the roofline calculator in this folder) for every baseline: this machine's
roofs, memory-bound or compute-bound, % of roof. Draws one class roofline,
one chart per kernel, a summary table and an optional web page.

    python demos/run_roofline.py                                   # whole class
    python demos/run_roofline.py benchmarks/gemver_numba_bench.py  # one student, detailed chart
    python demos/run_roofline.py --n 4096                          # same N for every kernel
    python demos/run_roofline.py --n gemver=4096 --n 2mm=256       # per-kernel N
    python demos/run_roofline.py --plot-only --all --html          # redraw from saved results
    python demos/run_roofline.py --peak-gflops 72.8 --bandwidth 37.2 --plot-only

Output goes to demos/roofline_output/ (change with --out). Only the standard
library, numpy, numba and matplotlib are used. More in demos/README.md.

# Optional: make your dots more precise
# - Add roofline_bytes(name, n) to your bench file.
# - Return how many bytes that baseline moves at size n (e.g. fused
#   versions that read a matrix fewer times move fewer bytes).
# - Return None to use the built-in estimate. Without the function the
#   built-in estimate is used for every baseline.
"""

import argparse
import ast
import importlib.util
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

# the roofline calculator (machine roofs, math, drawing) lives next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
import roofline as rl  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = rl.DEFAULT_OUT

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
    result = dict(kernel=key, file=str(path), n=n, started=rl.now(), warnings=[])
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
    rl.write_json(Path(args.json), result)
    return 0


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

    # Every baseline goes through roofline.py's analyze_point (same roofs, same rules)
    # - ws = working set: picks the L3 roof when the data fits in L3.
    # - traffic="read" (atax): judged against the read-only DRAM roof.
    # - "above the roof" happens e.g. for atax at N=4096: A (134 MB) is only 2x the L3,
    #   so repeated runs reuse part of it.
    ws = float(entry["ws"](n, g)) if entry else None
    start_ai = start["F"] / start["B"]
    for p in pts:
        # Python rows (capped at N=128 in some files) take the start row's AI
        ai = start_ai if p["python"] else p["F"] / p["B"]
        p.update(rl.analyze_point(p["name"], roofs, flops=p["F"], bytes_moved=p["F"] / ai if p["python"] else p["B"],
                                  seconds=p["elapsed"], gflops=p["gflops"], ai=ai, parallel=p["parallel"],
                                  working_set=ws, read_only=(entry or {}).get("traffic") == "read", unit=unit))
        if p["python"] and not p["far"]:
            p["far"] = True               # pure Python is always far below its roof
            p["verdict"] = rl.verdict(p["bound"], p["pct"], far=True)
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
                   ai=best["AI"], roof_used=f"{rl.LEVEL_NAMES[best['level']]}, {cores}",
                   roof=best["roof"], pct=best["pct"], verdict=best["verdict"],
                   bound=best["bound"], bytes_src=best["bytes_src"])
    else:
        row.update(best="(no correct baseline)", verdict="-", bytes_src=start["bytes_src"])
    return row


# ---------------------------------------------------------
# Results cache
# ---------------------------------------------------------
def load_results(out):
    data = rl.read_json(out / "results.json", None)
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
        result = rl.read_json(Path(tmp), None)
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
    roofs = rl.build_roofs(machine, **rl.roof_overrides(args))
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
        lines.append(f"- {label}: peak {rl.fmt(r['peak'])} GFLOP/s ({r['peak_src']}), "
                     f"DRAM {rl.fmt(r['dram'])} GB/s read + write ({r['dram_src']}), "
                     f"{rl.fmt(r['dram_read'])} GB/s read-only ({r['dram_read_src']})"
                     + (f", L3 {rl.fmt(r['l3'])} GB/s ({r['l3_src']})" if r.get("l3") else "")
                     + (f", ridge {rl.fmt(r['peak'] / r['dram'])} FLOP/byte" if r['peak'] and r['dram'] else ""))
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
    notes.append(rl.roof_source_note(report["roofs"]))
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
        write_points_csv(report, out)
        for r in plotted:
            if r["kernel"] in kernels_detail:
                draw_kernel_chart(r, report, out, args)
        if args.html:
            write_html(report, rows, out, args)
    elif plotted:
        print("No roofs available: run without --plot-only once, or pass --peak-gflops and --bandwidth.")
    return report, rows


def write_points_csv(report, out):
    """roofline_points.csv: every baseline of every plotted kernel (roofline.py's write_csv)."""
    pts = []
    for r in report["results"]:
        if r["status"] != "ok":
            continue
        entry = KERNELS.get(r["kernel"], {})
        for p in r["points"]:
            role = "best" if p is r["best"] else "start" if p is r["start"] else ""
            pts.append(dict(p, kernel=entry.get("label", r["kernel"].upper()),
                            student=entry.get("student", "?"), n=r["n"], role=role))
    rl.write_csv(pts, out / "roofline_points.csv", cpu=report["machine"].get("cpu"),
                 extra=[("kernel", "kernel"), ("student", "student"), ("n", "n"), ("role", "role"),
                        ("correct", "correct"), ("bytes_src", "bytes_src")])


# ---------------------------------------------------------
# Charts
# - The roofs, labels and per-kernel charts are drawn by roofline.py.
# - This file adds the class chart (start -> best per kernel) and the web page.
# ---------------------------------------------------------
# Kernel colors, fixed by position in KERNELS (a kernel keeps its color in every run)
# - Okabe-Ito (colorblind-safe), with black and #B8A400 replaced: black vanished
#   into the roof line, and #B8A400 next to #E69F00 failed the colorblind check.
# - All 36 pairs checked under simulated red/green colorblindness: worst
#   distance 7.6 (OKLab x100), and every kernel is also labelled by name.
KERNEL_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
                 "#56B4E9", "#8800FF", "#CC0099", "#882255"]

THEME = dict(rl.LIGHT, kernels=KERNEL_COLORS, other="#6b7280")


def kernel_color(key, theme):
    order = list(KERNELS)
    if key in order and order.index(key) < len(theme["kernels"]):
        return theme["kernels"][order.index(key)]
    return theme["other"]          # never invent a 10th hue


def _plotted(res, args):
    return [p for p in res["points"] if args.include_python or not p["python"]]


def _subtitle(report, sizes):
    layers = sorted({r["run"].get("threading_layer") for r in report["results"]} - {None})
    return rl.machine_line(report["machine"], layers, ["N: " + ", ".join(sizes)] if sizes else [])


def class_figure(report, args, theme=THEME, all_points=None, gid_prefix="c"):
    """The class chart: one machine's roofs, start -> best per kernel, verdict panel."""
    from matplotlib.patches import FancyArrowPatch
    show_all = args.all if all_points is None else all_points
    results = [r for r in report["results"] if r["status"] == "ok"]
    roofs = report["roofs"]
    # SVG ids for the HTML page: <prefix>-k<kernel>-pt<row> for start/best, -xpt<row> for the rest
    for ki, r in enumerate(results):
        for j, p in enumerate(r["points"]):
            p["gid"] = f"{gid_prefix}-k{ki}-{'pt' if p is r['start'] or p is r['best'] else 'xpt'}{j}"
    fig = rl.new_figure(13, 7.5, theme)
    renderer = fig.canvas.get_renderer()

    pts = []
    for r in results:
        pts += [r["start"]] + ([r["best"]] if r["best"] else [])
        if show_all:
            pts += _plotted(r, args)
        elif args.include_python:
            pts += [p for p in r["points"] if p["python"]]
    xlim, ylim = rl.limits(pts, roofs)
    # more than five footnotes: lift the plot so the footer clears the x-axis label
    lift = 0.024 * max(0, len(footnotes(report, args)) - 5)
    ax = fig.add_axes([0.065, 0.20 + lift, 0.615, 0.66 - lift])
    rl.style_axes(ax, theme, xlim, ylim)

    # memory-bound / compute-bound background, split at the all-core ridge
    ridge_all = roofs["all"]["peak"] / roofs["all"]["dram"]
    ax.axvspan(xlim[0], ridge_all, color=theme["bg_mem"], zorder=0, linewidth=0, gid="bg-mem")
    ax.axvspan(ridge_all, xlim[1], color=theme["bg_comp"], zorder=0, linewidth=0, gid="bg-comp")
    corner = dict(fontsize=7.5, color=theme["muted"], transform=ax.transAxes, va="top", zorder=1)
    ax.text(0.012, 0.985, "M E M O R Y - B O U N D", ha="left", **corner)
    ax.text(0.988, 0.985, "C O M P U T E - B O U N D", ha="right", **corner)

    levels = {p["level"] for p in pts}
    avoid = rl.draw_roofs(ax, roofs, xlim, ylim, theme, levels, renderer, [(p["AI"], p["P"]) for p in pts])
    avoid.append(rl.text_box(ax.texts[0], renderer))
    avoid.append(rl.text_box(ax.texts[1], renderer))

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
                avoid.append(rl.dot_box(ax, (p["AI"], p["P"]), 3))

    # start (hollow) -> best (filled) with an arrow, colored per kernel
    for r in results:
        col = kernel_color(r["kernel"], theme)
        s, b = r["start"], r["best"]
        if b is None:
            # every baseline failed its check: x markers, no best and no arrow
            ax.plot([s["AI"]], [s["P"]], marker="x", markersize=8, color=col, markeredgewidth=1.8,
                    linestyle="none", zorder=6, gid=s.get("gid"))
            avoid.append(rl.dot_box(ax, (s["AI"], s["P"]), 6))
            continue
        ax.plot([s["AI"]], [s["P"]], marker="o", markersize=7, markerfacecolor=theme["surface"],
                markeredgecolor=col, markeredgewidth=1.6, linestyle="none", zorder=6, gid=s.get("gid"))
        avoid.append(rl.dot_box(ax, (s["AI"], s["P"]), 6))
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
        avoid.append(rl.dot_box(ax, (b["AI"], b["P"]), 7))

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
        rl.place_label(ax, renderer, (b["AI"], b["P"]), (title, detail), avoid, theme, theme["faint"],
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
                sub = f"{b['name']} · N={r['n']} · {rl.LEVEL_NAMES[b['level']]}, " \
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

    rl.footer(fig, footnotes(report, args), theme)
    return fig


def draw_class_chart(report, out, args):
    fig = class_figure(report, args)
    rl.save_figure(fig, out, "roofline_class")


def kernel_figure(res, report, args, theme=THEME):
    """Per-kernel chart: every baseline of one kernel, drawn by roofline.py's roofline_figure."""
    pts = _plotted(res, args)
    entry = KERNELS.get(res["kernel"], {})
    notes = footnotes(report, args)[:1] + [n for n in footnotes(report, args)[1:]
                                           if "Python" in n or "BLAS" in n or "workqueue" in n
                                           or ("read-only" in n and any(p["level"] == "dram_read" for p in pts))]
    notes += ["Warning: " + w[:150] for w in res["warnings"][:2]]
    title = f"{entry.get('label', res['kernel'].upper())} · {entry.get('student', '?')} · Roofline, N={res['n']}"
    return rl.roofline_figure(pts, report["roofs"], title, _subtitle(report, []), notes,
                              best=res["best"], start=res["start"], theme=theme)


def draw_kernel_chart(res, report, out, args):
    fig = kernel_figure(res, report, args)
    rl.save_figure(fig, out, f"roofline_{res['kernel']}_N{res['n']}")


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
                                    level=rl.LEVEL_NAMES[p["level"]],
                                    cores="all cores" if p["cores"] == "all" else "1 core",
                                    verdict=p["verdict"], parallel=p["parallel"], correct=p["correct"])
        chips.append(f'<button class="chip" data-k="k{ki}" aria-pressed="true"><i style="background:'
                     f'{kernel_color(r["kernel"], THEME)}"></i>{html.escape(label)} · {html.escape(student)}</button>')

    trs = []
    for row in rows:
        k = f'<span class="sw" style="background:{kernel_color(row["key"], THEME)}"></span>' \
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
  python demos/run_roofline.py                                   every benchmarks/*_bench.py
  python demos/run_roofline.py benchmarks/matmul_numba_bench.py  one kernel, with its detailed chart
  python demos/run_roofline.py --n 256                           same N for every kernel (faster)
  python demos/run_roofline.py --n matmul=1024                   N for one kernel only
  python demos/run_roofline.py --plot-only --html                redraw saved results, add the web page
  python demos/run_roofline.py --plot-only --all                 every baseline on the class chart

Charts, a summary table, roofline_points.csv and results.json are written to
demos/roofline_output/. A bench file can define roofline_bytes(name, n) for exact
bytes (see the top of this file). One computation on its own: demos/roofline.py.
"""


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="python demos/run_roofline.py",
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

    machine = rl.add_machine_args(p)
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

    # the output folder ignores itself, so the repo's .gitignore needs no change
    out = rl.ensure_out(Path(args.out).resolve())
    data = load_results(out)

    explicit = [Path(p) for p in args.paths]
    single_file = len(explicit) == 1 and explicit[0].is_file()
    per_kernel = single_file if args.per_kernel is None else args.per_kernel

    if args.plot_only:
        kernels = {kernel_key(p) for p in explicit if p.is_file()} or None
        machine = rl.load_machine(out, cached_only=True)
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
    machine = rl.load_machine(out, remeasure=args.remeasure, l3_mb=args.l3_mb)
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
            merge_skip(data, dict(kernel=key, n=n, file=str(path), reason="budget exhausted", when=rl.now()))
            continue
        print(f"{label}: running {path.name} ...", flush=True)
        t0 = time.time()
        run, err = run_worker(path, key, n, min(args.timeout, left), out)
        if err:
            print(f"{label}: skipped: {err}")
            merge_skip(data, dict(kernel=key, n=n, file=str(path), reason=err, when=rl.now()))
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
        rl.write_json(out / "results.json", data)
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
