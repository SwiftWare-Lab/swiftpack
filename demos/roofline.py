"""
Roofline calculator: where does a computation sit on this machine's roofline?

Give it a computation: its FLOPs, the bytes it moves and its run time, or a
Python function to time. It measures this machine's roofs once (peak GFLOP/s
and memory bandwidth, cached in machine.json), places the computation under
them, marks it memory-bound or compute-bound, and writes one chart (PNG) and
one table (CSV).

    python demos/roofline.py --name my_kernel --flops 1.68e8 --bytes 4.03e8 --time 0.0135
    python demos/roofline.py --example
    python demos/roofline.py --help

From Python: analyze_point(), measure(), plot_roofline() and write_csv().
How every number is calculated: README.md in this folder.
Only the standard library, numpy, numba and matplotlib are used.
"""

import argparse
import csv
import datetime
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
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "roofline_output"
LEVEL_NAMES = {"dram": "DRAM", "dram_read": "DRAM read-only", "l3": "L3"}


# ---------------------------------------------------------
# Small helpers
# ---------------------------------------------------------
def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def write_json(path, data):
    # - Temp file + os.replace: a crash never leaves a half-written file.
    # - os.rename fails on Windows when the target exists; os.replace does not.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def read_json(path, default):
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


def fmt(x):
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


def ensure_out(out):
    """Create the output folder. It ignores itself, so `git status` stays clean."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / ".gitignore").exists():
        (out / ".gitignore").write_text("# written by demos/roofline.py\n*\n", encoding="utf-8")
    return out


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

    # Peak FLOP/s micro-benchmark
    # - Same idea as the FLOP kernel of the Empirical Roofline Toolkit (ERT): a chain of
    #   FMAs on each element of a small array, stored back (ERT: beta = beta*A[i] + alpha).
    #   Lo et al., "Roofline Model Toolkit", PMBS 2014, doi:10.1007/978-3-319-17248-4_7.
    #   Code: https://bitbucket.org/berkeleylab/cs-roofline-toolkit
    # - 8 chained x = x*a + b per element per pass = 8 FMAs = 16 FLOPs (ERT_FLOP=16).
    # - 8 was the best count on a Ryzen 9 5900X (92% of theoretical; 4 -> 69%, 16 -> 74%):
    #   with fewer, loading and storing x[i] limits the loop; with more, the chains get
    #   too long to hide the FMA latency. Details in README.md (this folder).
    # - a and b MUST be arguments: as literals, fastmath folds the chain and the
    #   5900X reported an impossible 170-195 GFLOP/s (over 230% of theoretical).
    # - x settles at b / (1 - a) = 0.1: no overflow, no slow denormal numbers.
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

    # Memory bandwidth micro-benchmarks: the DAXPY and SUM kernels of McCalpin's STREAM2
    # - https://www.cs.virginia.edu/stream/stream2/ (STREAM: https://www.cs.virginia.edu/stream/)
    # - DAXPY a = a + s*b: read a, read b, write a = 24 bytes per element.
    # - a is read before it is written, so there is no hidden write-allocate traffic
    #   (STREAM's Triad writes a third array, which the CPU first reads for ownership).
    @njit(fastmath=True)
    def daxpy_serial(a, b, s):
        for i in range(a.shape[0]):
            a[i] = a[i] + s * b[i]

    @njit(parallel=True, fastmath=True)
    def daxpy_parallel(a, b, s):
        for i in prange(a.shape[0]):
            a[i] = a[i] + s * b[i]

    # - SUM t += a[i]: read only, 8 bytes per element, nothing written.
    # - Kernels that only read their big array can beat the read + write roof.
    @njit(fastmath=True)
    def sum_serial(a, b, s):
        t = 0.0
        for i in range(a.shape[0]):
            t += a[i]
        return t

    @njit(parallel=True, fastmath=True)
    def sum_parallel(a, b, s):
        t = 0.0
        for i in prange(a.shape[0]):
            t += a[i]
        return t

    return daxpy_serial, daxpy_parallel, sum_serial, sum_parallel


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
    daxpy_serial, daxpy_parallel, sum_serial, sum_parallel = _stream_kernels()
    l3_total = (info.get("l3_total_mb") or 32.0) * 2**20
    n = int(max(128 * 2**20, 4 * l3_total) // 8)
    big_a, big_b = np.ones(n), np.ones(n)
    m["dram_1"] = _best_gbps(daxpy_serial, big_a, big_b)
    by_threads = {}
    for t in _thread_counts(max_threads, physical):
        set_num_threads(t)
        by_threads[t] = _best_gbps(daxpy_parallel, big_a, big_b)
    set_num_threads(max_threads)
    # - The all-core roof is the best thread count, not always the most threads:
    #   on the 5900X 24 threads got LESS than 1 (27.6 vs 37.2 GB/s).
    m["dram_by_threads"] = {str(k): v for k, v in by_threads.items()}
    m["dram_all"] = max(max(by_threads.values()), m["dram_1"])
    log(f"  DRAM: {m['dram_1']:.1f} GB/s 1 core, {m['dram_all']:.1f} best of "
        + ", ".join(f"{k} thr {v:.1f}" for k, v in by_threads.items()) + "  (read + write)")
    m["dram_read_1"] = _best_gbps(sum_serial, big_a, big_b, bytes_per_elem=8.0)
    read_by = {}
    for t in _thread_counts(max_threads, physical):
        set_num_threads(t)
        read_by[t] = _best_gbps(sum_parallel, big_a, big_b, bytes_per_elem=8.0)
    set_num_threads(max_threads)
    m["dram_read_by_threads"] = {str(k): v for k, v in read_by.items()}
    m["dram_read_all"] = max(max(read_by.values()), m["dram_read_1"])
    del big_a, big_b
    log(f"  DRAM: {m['dram_read_1']:.1f} GB/s 1 core, {m['dram_read_all']:.1f} best of "
        + ", ".join(f"{k} thr {v:.1f}" for k, v in read_by.items()) + "  (read only)")

    # L3 bandwidth: both arrays together fill about half of one L3 instance
    n = int(l3_instance_mb * 2**20 / 4 // 8)
    small_a, small_b = np.ones(n), np.ones(n)
    m["l3_1"] = _best_gbps(daxpy_serial, small_a, small_b, min_time=0.2)
    m["l3_all"] = max(_best_gbps(daxpy_parallel, small_a, small_b, min_time=0.2), m["l3_1"])
    log(f"  L3:   {m['l3_1']:.1f} GB/s 1 core, {m['l3_all']:.1f} all cores "
        f"(2 x {n * 8 / 2**20:.1f} MiB arrays)")
    m["numba_threads"] = max_threads
    return m


def _timed(fn, *args):
    t = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t


def load_machine(out_dir=DEFAULT_OUT, remeasure=False, l3_mb=None, cached_only=False, log=print):
    """
    This machine's profile: CPU details plus the measured roofs.
    - Cached in out_dir/machine.json; measured again for a new machine or remeasure=True.
    - l3_mb: L3 size of one instance (Windows only reports the chip total).
    - cached_only=True: just read the saved profile (None if there is none).
    """
    path = Path(out_dir) / "machine.json"
    saved = read_json(path, None)
    if cached_only:
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
    l3_user = l3_mb or (saved or {}).get("l3_mb_user")
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
            and not remeasure):
        profile["measured"] = saved["measured"]
        profile["measured_at"] = saved.get("measured_at")
        if profile != saved:
            write_json(path, profile)
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
    profile["measured_at"] = now()
    write_json(path, profile)
    return profile


def _parse_pair(value):
    """'72.8' or 72.8 -> (72.8, 72.8); '72.8,610' or (72.8, 610) -> (72.8, 610.0)."""
    if isinstance(value, (tuple, list)):
        value = ",".join(str(v) for v in value)
    parts = [float(p) for p in str(value).split(",") if p.strip()]
    if not parts:
        raise ValueError(value)
    return parts[0], parts[1] if len(parts) > 1 else parts[0]


def build_roofs(machine, peak_gflops=None, bandwidth=None, read_bandwidth=None, l3_bandwidth=None,
                peak_source="measured"):
    """
    Roof sets for 1 core and all cores: peak, DRAM (read + write), DRAM read-only, L3.
    - Measured values by default; peak_source="theoretical" switches the peak.
    - Overrides (GFLOP/s or GB/s): one value sets both the 1-core and the all-core
      roof; "1core,allcores" or a pair sets them separately.
    """
    machine = machine or {}
    meas = machine.get("measured") or {}
    theo = machine.get("theoretical") or {}
    overrides = dict(peak_gflops=peak_gflops, bandwidth=bandwidth, read_bandwidth=read_bandwidth,
                     l3_bandwidth=l3_bandwidth)
    roofs = {}
    for c in ("1", "all"):
        peak, src = meas.get(f"peak_{c}"), "measured"
        if peak_source == "theoretical" or not peak:
            peak, src = theo.get(c), "theoretical"
        roofs[c] = dict(peak=peak, peak_src=src, theo=theo.get(c),
                        dram=meas.get(f"dram_{c}"), dram_src="measured",
                        dram_read=meas.get(f"dram_read_{c}"), dram_read_src="measured",
                        l3=meas.get(f"l3_{c}"), l3_src="measured")
    for flag, field in (("peak_gflops", "peak"), ("bandwidth", "dram"), ("read_bandwidth", "dram_read"),
                        ("l3_bandwidth", "l3")):
        value = overrides[flag]
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


def roof_overrides(args):
    """The flags from add_machine_args() as keywords for build_roofs()."""
    return dict(peak_gflops=args.peak_gflops, bandwidth=args.bandwidth,
                read_bandwidth=args.read_bandwidth, l3_bandwidth=args.l3_bandwidth,
                peak_source=args.peak_source)


def add_machine_args(parser):
    """Machine flags shared by every command line; returns the group so a caller can add more."""
    machine = parser.add_argument_group("machine numbers (measured automatically; override if needed)")
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
    return machine


def machine_line(machine, layers=(), extra=()):
    """One line for a chart subtitle: CPU · cores / threads · Numba threads · date."""
    m = machine or {}
    nt = (m.get("measured") or {}).get("numba_threads")
    parts = [m.get("cpu", "unknown CPU")]
    if m.get("physical_alloc"):
        parts.append(f"{m['physical_alloc']} cores / {m.get('allocated_cpus')} threads")
    if nt:
        parts.append(f"Numba threads {nt}" + (f" ({', '.join(layers)})" if layers else ""))
    parts.append(datetime.date.today().isoformat())
    return " · ".join(parts + list(extra))


def roof_source_note(roofs):
    """Footnote: where the roof numbers come from."""
    srcs = {roofs[c][f] for c in ("1", "all") for f in ("peak_src", "dram_src")}
    return "Peak and bandwidth " + ("set by the user (user)." if srcs == {"user"} else
                                    "measured on this machine." if "user" not in srcs else
                                    "measured on this machine; values marked (user) were given.")


# ---------------------------------------------------------
# Roofline math for one computation
# - P = performance (GFLOP/s), AI = arithmetic intensity (FLOP per byte moved).
# - Roof(AI) = min(peak, bandwidth x AI); ridge = peak / bandwidth.
# - Left of the ridge the bandwidth roof is lower: memory-bound. Right: compute-bound.
# - Under the memory roof the FLOP count cancels: P / (bandwidth x AI) =
#   bytes / (time x bandwidth), so "% of roof" there depends only on bytes and time.
# ---------------------------------------------------------
def verdict(bound, pct, far):
    """The bound, plus a note when the point is far below or above its roof."""
    text = bound + (", far below the roof (locality or overhead)" if far else "")
    if pct > 1.0:
        # e.g. an array only ~2x the L3: repeated runs still find part of it in cache
        text += ", above the roof (some data reused from cache)"
    return text


def analyze_point(name, roofs, flops=None, bytes_moved=None, seconds=None, gflops=None, ai=None,
                  parallel=False, working_set=None, read_only=False, unit="FLOP"):
    """
    Roofline numbers for one computation (one baseline).
    - Performance: gflops, or flops / seconds.
    - Arithmetic intensity: ai, or flops / bytes_moved.
    - parallel=True uses the all-core roofs; working_set (bytes) uses the L3 roof
      when the data fits in L3; read_only=True uses the read-only DRAM roof.
    Returns a dict: P, AI, roof, ridge, pct (fraction of the roof), bound, verdict, ...
    """
    if gflops is None:
        if not (flops and seconds):
            raise ValueError(f"{name}: give gflops, or flops and seconds")
        gflops = flops / seconds / 1e9
    if ai is None:
        if not (flops and bytes_moved):
            raise ValueError(f"{name}: give ai, or flops and bytes moved")
        ai = flops / bytes_moved
    P, AI = float(gflops), float(ai)
    if not (P > 0 and AI > 0):
        raise ValueError(f"{name}: performance and arithmetic intensity must be positive")
    cores = "all" if parallel else "1"
    # L3 roof only if the working set fits comfortably in the L3 the computation can use
    # - serial: one L3 instance (one CCD on AMD)
    # - parallel: all instances (threads on every CCD share the work)
    l3_one = roofs.get("l3_instance_mb")
    l3_all = max(roofs.get("l3_total_mb") or 0, l3_one or 0) or None
    cap = l3_all if parallel else l3_one
    level = "l3" if (working_set is not None and cap and working_set <= 0.8 * cap * 2**20) else "dram"
    if level == "dram" and read_only:
        level = "dram_read"           # only reads its big array: read-only DRAM roof
    lvl = level if roofs[cores].get(level) else "dram"
    pi, beta = roofs[cores]["peak"], roofs[cores][lvl]
    roof = min(pi, beta * AI)
    pct = P / roof
    bound = "Memory-bound" if AI < pi / beta else "Compute-bound"
    far = pct < 0.10
    return dict(name=name, unit=unit, F=flops, B=bytes_moved, seconds=seconds, P=P, AI=AI,
                parallel=bool(parallel), cores=cores, level=lvl, peak=pi, bw=beta, roof=roof,
                ridge=pi / beta, pct=pct, bound=bound, far=far, verdict=verdict(bound, pct, far))


def measure(fn, *args, flops, bytes_moved, name=None, setup=None, reps=9, parallel=None,
            working_set=None, read_only=False, unit="FLOP", roofs=None):
    """
    Time fn(*args), then place it on the roofline (see analyze_point).
    - One warm-up call first: Numba compiles on the first call.
    - setup(): optional; returns fresh arguments before every call, for kernels that
      change their inputs in place. It is not timed.
    - Time = mean of `reps` calls (the warm-up call is not counted).
    - parallel=None: read it from the Numba function (compiled with parallel=True?).
    - roofs=None: this machine's roofs (measured once, then cached).
    """
    if roofs is None:
        roofs = build_roofs(load_machine())
    if parallel is None:
        parallel = bool(getattr(fn, "targetoptions", {}).get("parallel"))
    fn(*(setup() if setup else args))
    total = 0.0
    for _ in range(reps):
        call_args = setup() if setup else args
        t = time.perf_counter()
        fn(*call_args)
        total += time.perf_counter() - t
    return analyze_point(name or getattr(fn, "__name__", "computation"), roofs, flops=flops,
                         bytes_moved=bytes_moved, seconds=total / reps, parallel=parallel,
                         working_set=working_set, read_only=read_only, unit=unit)


# ---------------------------------------------------------
# Charts
# - matplotlib's Figure + Agg canvas directly (no pyplot): no window can pop up.
# - Axes sit at fixed positions so the slanted roof labels keep their angle.
# - Colors live in a theme dict, so a caller can add colors (e.g. one per series).
# ---------------------------------------------------------
LIGHT = dict(
    surface="#ffffff", bg_mem="#eef4fb", bg_comp="#fdf6ec", ink="#111827", ink2="#4b5563",
    muted="#6b7280", grid="#e5e7eb", faint="#9ca3af", ridge="#6d28d9", l3="#6b7280",
    pill={"Memory-bound": ("#dbeafe", "#1e3a8a"), "Compute-bound": ("#fdecc8", "#7c4a03")},
    bar={"Memory-bound": "#2a78d6", "Compute-bound": "#eda100"},
    # ordinal ramp for baseline families: early versions light, late versions dark
    ramp=["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
          "#1c5cab", "#184f95", "#104281", "#0d366b"],
)


def _ramp_color(theme, t):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("fam", theme["ramp"])
    return cmap(min(1.0, max(0.0, t)))


def new_figure(w, h, theme):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(w, h), dpi=100, facecolor=theme["surface"])
    FigureCanvasAgg(fig)
    return fig


def style_axes(ax, theme, xlim, ylim):
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


def limits(points, roofs, extra_x=()):
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


def text_box(artist, renderer, pad=2):
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


def _slant_hits(label, dots, renderer, r=6):
    """Does a dot (display pixels) sit on this rotated label? Checks the rotated text box."""
    w, h, _ = renderer.get_text_width_height_descent(label.get_text(), label.get_fontproperties(),
                                                     ismath=False)
    bb = label.get_window_extent(renderer)
    cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
    ang = math.radians(label.get_rotation())
    ux, uy = math.cos(ang), math.sin(ang)
    for px, py in dots:
        along, across = (px - cx) * ux + (py - cy) * uy, -(px - cx) * uy + (py - cy) * ux
        if abs(along) <= w / 2 + r and abs(across) <= h / 2 + r:
            return True
    return False


def _slant_label(ax, text, bw, xlim, ylim, ridge, frac, above, color, size=8.5, gid=None, dots=()):
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
    # a dot on the label: slide along the line to the nearest clear spot (else stay)
    if dots and _slant_hits(label, dots, renderer):
        for k in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, 8):
            xk = x * 1.15 ** k
            if xk > ridge / 1.5:
                continue
            label.xy = (xk, bw * xk)
            if label.get_window_extent(renderer).x0 >= left and not _slant_hits(label, dots, renderer):
                break
        else:
            label.xy = (x, bw * x)
    return label


def draw_roofs(ax, roofs, xlim, ylim, theme, levels_used, renderer, dots=()):
    """Draw the roofs, their labels and the ridge points. Returns the obstacles for labels."""
    from matplotlib.transforms import Bbox
    # peak labels must also clear dots and the corner texts already on the axes
    dot_boxes = [dot_box(ax, xy, 7) for xy in dots] + [text_box(t, renderer) for t in ax.texts]
    dot_px = [tuple(ax.transData.transform(xy)) for xy in dots]    # roof labels slide off dots
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
            text = (f"DRAM {fmt(bw)} GB/s{src}" if abs(bw - one) < 0.05 else
                    f"DRAM {fmt(bw)} GB/s{src} (1 core {fmt(one)}{src1})")
            artists.append(_slant_label(ax, text, low, xlim, ylim, pi / low, 0.24, above=False,
                                        color=theme["ink2"], gid=f"rooflabel-{cores}-{lvl}", dots=dot_px))
            continue
        # all-core label above its line, 1-core below, so the two near-parallel slopes stay readable
        frac = {("all", "dram"): 0.30, ("1", "dram"): 0.22, ("all", "l3"): 0.52, ("1", "l3"): 0.40,
                # read-only labels go near the top of their slope, clear of the DRAM labels
                ("all", "dram_read"): 0.95, ("1", "dram_read"): 0.85}[(cores, lvl)]
        artists.append(_slant_label(ax, f"{name} {fmt(bw)} GB/s · {who}{src}", bw, xlim, ylim, pi / bw,
                                    frac, above=(cores == "all"),
                                    color=theme["ink2"] if lvl != "l3" else theme["l3"],
                                    gid=f"rooflabel-{cores}-{lvl}", dots=dot_px))
    # ridge points (DRAM roofs), drawn before the peak labels so those can avoid them
    for cores in ("all", "1"):
        r = roofs[cores]
        if not r["peak"] or not r["dram"] or (cores == "1" and roofs["all"]["peak"] == r["peak"]
                                              and roofs["all"]["dram"] == r["dram"]):
            continue
        ridge = r["peak"] / r["dram"]
        ax.plot([ridge], [r["peak"]], marker="D", markersize=6.5, markerfacecolor=theme["surface"],
                markeredgecolor=theme["ridge"], markeredgewidth=1.5, zorder=6, gid=f"ridge-{cores}")
        rl = ax.annotate(f"ridge ≈ {fmt(ridge)} FLOP/byte", (ridge, r["peak"]), xytext=(7, -5),
                         textcoords="offset points", ha="left", va="top", fontsize=8.5,
                         color=theme["ridge"], gid=f"ridgelabel-{cores}")
        artists.append(rl)
        dot_boxes.append(text_box(rl, renderer))
    # flat compute roofs: label once per core count
    for cores in ("all", "1"):
        r = roofs[cores]
        if not r["peak"] or (cores == "1" and roofs["all"]["peak"] == r["peak"]):
            continue
        # one peak for both (e.g. a single --peak-gflops value): no "All cores", and no
        # theoretical line unless both core counts share the same theoretical peak
        same = roofs["1"]["peak"] == roofs["all"]["peak"]
        who = "Peak" if same else "All cores" if cores == "all" else "1 core"
        how = r["peak_src"]
        theo = r.get("theo") if not same or roofs["1"].get("theo") == roofs["all"].get("theo") else None
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
            note = ax.annotate(how + (f" · theoretical ≈ {fmt(theo)}" if show_theo else ""),
                               (x_at, y_at), xytext=(dx, 4 * side), textcoords="offset points",
                               ha=ha, va=va, fontsize=7.5, color=theme["muted"], gid=f"peaknote-{cores}")
            main = ax.annotate(f"Peak ≈ {fmt(r['peak'])} GFLOP/s" if same else
                               f"{who} · peak ≈ {fmt(r['peak'])} GFLOP/s",
                               (x_at, y_at), xytext=(dx, 15 * side), textcoords="offset points",
                               ha=ha, va=va, fontsize=8.5, color=theme["ink2"], gid=f"peaklabel-{cores}")
            box = Bbox.union([text_box(note, renderer, 0), text_box(main, renderer, 0)])
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
        avoid.append(text_box(a, renderer))
    return avoid


def place_label(ax, renderer, xy, lines, avoid, theme, color_line, gid=None, size=8.5):
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
        return t1, t2, Bbox.union([text_box(t1, renderer), text_box(t2, renderer)])

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


def dot_box(ax, xy, r=6):
    from matplotlib.transforms import Bbox
    px, py = ax.transData.transform(xy)
    return Bbox([[px - r, py - r], [px + r, py + r]])


def footer(fig, notes, theme, y0=0.018):
    fig.text(0.04, y0, "\n".join(notes), fontsize=8, color=theme["muted"], ha="left", va="bottom",
             linespacing=1.45)


def save_figure(fig, out, stem, dpi=200):
    for ext in ("png", "svg"):
        tmp = out / f"{stem}.tmp.{ext}"
        fig.savefig(tmp, dpi=dpi, facecolor=fig.get_facecolor())
        os.replace(tmp, out / f"{stem}.{ext}")


def roofline_figure(points, roofs, title, subtitle="", notes=(), best=None, start=None, theme=LIGHT):
    """
    One roofline chart: this machine's roofs, one dot per point, % of roof bars on the right.
    - points: dicts from analyze_point(). Optional keys: "family" (versions of one idea
      share a number; dots go from light to dark) and "correct" (False draws an x).
    - best / start: labelled first, tagged "(best)" and "(start)"; then the fastest
      correct points, 5 labels at most.
    - One point (or none): no bar panel, since the dot's label already gives % of roof.
    """
    pts = list(points)
    by_perf = sorted(pts, key=lambda p: p["P"], reverse=True)
    notes = list(notes)
    show_bars = len(pts) >= 2
    h = max(7.5, 2.2 + 0.19 * len(by_perf))
    fig = new_figure(13 if show_bars else 10, h, theme)
    renderer = fig.canvas.get_renderer()
    top = 1 - 0.95 / h * 1.25
    bottom = (0.75 + 0.17 * len(notes)) / h     # room for the x-axis label and the footer
    xlim, ylim = limits(pts, roofs)
    ax = fig.add_axes([0.065, bottom, 0.56, top - bottom] if show_bars else [0.085, bottom, 0.87, top - bottom])
    style_axes(ax, theme, xlim, ylim)
    ridge_all = roofs["all"]["peak"] / roofs["all"]["dram"]
    ax.axvspan(xlim[0], ridge_all, color=theme["bg_mem"], zorder=0, linewidth=0)
    ax.axvspan(ridge_all, xlim[1], color=theme["bg_comp"], zorder=0, linewidth=0)
    corner = dict(fontsize=7.5, color=theme["muted"], transform=ax.transAxes, va="top", zorder=1)
    c1 = ax.text(0.012, 0.985, "M E M O R Y - B O U N D", ha="left", **corner)
    c2 = ax.text(0.988, 0.985, "C O M P U T E - B O U N D", ha="right", **corner)
    # no points: draw every roof, so the chart shows the whole machine
    levels = {p["level"] for p in pts} if pts else {"l3", "dram_read"}
    avoid = draw_roofs(ax, roofs, xlim, ylim, theme, levels, renderer,
                       [(p["AI"], p["P"]) for p in pts])
    avoid.append(text_box(c1, renderer))
    avoid.append(text_box(c2, renderer))

    fams = sorted({p.get("family") for p in pts if p.get("family") is not None})
    def fam_color(p):
        if p.get("family") is None or len(fams) < 2:
            return _ramp_color(theme, 0.6)
        return _ramp_color(theme, fams.index(p["family"]) / (len(fams) - 1))

    for p in sorted(pts, key=lambda p: p["P"]):
        col = fam_color(p)
        if not p.get("correct", True):
            ax.plot([p["AI"]], [p["P"]], marker="x", markersize=7, color=col, markeredgewidth=1.6,
                    linestyle="none", zorder=5)
        else:
            ax.plot([p["AI"]], [p["P"]], marker="^" if p["parallel"] else "o",
                    markersize=8 if p["parallel"] else 7.5, markerfacecolor=col,
                    markeredgecolor=theme["surface"], markeredgewidth=1.1, linestyle="none", zorder=6)
        avoid.append(dot_box(ax, (p["AI"], p["P"]), 5))

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
    if not show_bars:
        # one dot (or none): only the marker that is on the chart
        kinds = {"failed check" if not p.get("correct", True) else "parallel" if p["parallel"] else "serial"
                 for p in pts}
        handles = [hd for hd in handles if hd.get_label() in kinds]
    if handles:
        leg = ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True, framealpha=0.92,
                        edgecolor=theme["grid"], facecolor=theme["surface"], labelcolor=theme["ink2"])
        avoid.append(leg.get_window_extent(renderer))

    # labels: best, start and the fastest others
    unit = "GOP/s" if pts and pts[0].get("unit") == "OP" else "GFLOP/s"
    chosen = []
    for p in [best, start] + [q for q in by_perf if q.get("correct", True)]:
        if p is not None and p not in chosen and len(chosen) < 5:
            chosen.append(p)
    for p in chosen:
        tag = " (best)" if p is best else " (start)" if p is start else ""
        place_label(ax, renderer, (p["AI"], p["P"]), (p["name"] + tag,
                    f"{p['P']:.3g} {unit} · {100 * p['pct']:.0f}% of roof"), avoid, theme, theme["faint"],
                    size=8)

    fig.text(0.04, 1 - 0.35 / h, title, fontsize=16, fontweight="bold", color=theme["ink"], ha="left", va="top")
    fig.text(0.04, 1 - 0.72 / h, subtitle, fontsize=10.5, color=theme["muted"], ha="left", va="top")

    if show_bars:
        # right panel: % of roof per point, sorted by performance
        bx = fig.add_axes([0.80, bottom, 0.17, top - bottom])
        names = [p["name"] for p in by_perf]
        vals = [100 * p["pct"] for p in by_perf]
        cols = [theme["bar"][p["bound"]] for p in by_perf]
        slots = max(len(by_perf), 6)        # few points: keep normal bar thickness, at the top
        ypos = [slots - 1 - i for i in range(len(by_perf))]
        bars = bx.barh(ypos, vals, height=0.62, color=cols, edgecolor="none")
        for bar, p in zip(bars, by_perf):
            if p["far"]:
                bar.set_alpha(0.45)
            if not p.get("correct", True):
                bar.set_hatch("///")
                bar.set_alpha(0.35)
        vmax = max(vals + [100])
        bx.set_xlim(0, vmax * 1.28)
        bx.set_ylim(-0.7, slots - 0.3)
        bx.set_yticks(ypos)
        bx.set_yticklabels([n + (" ✗" if not p.get("correct", True) else "") for n, p in zip(names, by_perf)],
                           fontsize=7.5 if len(names) > 22 else 8.5, color=theme["ink2"])
        for lbl, p in zip(bx.get_yticklabels(), by_perf):
            if p is best:
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

    footer(fig, notes, theme, y0=0.12 / h)
    return fig


def default_notes(points, roofs):
    """Footnotes for a chart of computations given by the user."""
    notes = ["Bytes moved are the user's estimate (no hardware counters), so AI is a model."]
    if any(p["level"] == "dram_read" and roofs[p["cores"]]["dram_read"] > 1.02 * roofs[p["cores"]]["dram"]
           for p in points):
        notes.append("Read-only computations use the read-only DRAM roof (dash-dot).")
    notes.append(roof_source_note(roofs))
    return notes


def plot_roofline(points, roofs, png_path, title=None, subtitle=None, notes=None, machine=None,
                  dpi=200, theme=LIGHT):
    """Draw roofline_figure() and save it as one PNG. Returns the PNG's path."""
    pts = list(points)
    if title is None:
        title = "Roofline · " + pts[0]["name"] if len(pts) == 1 else "Roofline"
    if subtitle is None:
        subtitle = machine_line(machine)
    if notes is None:
        notes = default_notes(pts, roofs)
    fig = roofline_figure(pts, roofs, title, subtitle, notes, theme=theme)
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = png_path.with_name(png_path.stem + ".tmp.png")
    fig.savefig(tmp, dpi=dpi, facecolor=fig.get_facecolor())
    os.replace(tmp, png_path)
    return png_path


# ---------------------------------------------------------
# Table (CSV): one row per computation
# ---------------------------------------------------------
CSV_COLUMNS = [("name", "name"), ("unit", "unit"), ("flops", "F"), ("bytes", "B"),
               ("seconds", "seconds"), ("gflops", "P"), ("ai", "AI"), ("parallel", "parallel"),
               ("cores", "cores"), ("level", "level"), ("peak_gflops", "peak"),
               ("bandwidth_gbs", "bw"), ("ridge", "ridge"), ("roof_gflops", "roof"),
               ("pct_of_roof", "pct"), ("bound", "bound"), ("verdict", "verdict")]


def _cell(key, value):
    if value is None:
        return ""
    if key == "pct":
        return f"{100 * value:.1f}"                  # percent of the roof
    if key == "cores":
        return "all cores" if value == "all" else "1 core"
    if key == "level":
        return LEVEL_NAMES.get(value, value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_csv(points, path, cpu=None, extra=()):
    """
    One row per point, plus a header row.
    - extra: (column, key) pairs written first, e.g. [("kernel", "kernel")].
    - pct_of_roof is in percent; other numbers keep 6 significant digits.
    """
    cols = list(extra) + CSV_COLUMNS
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([c for c, _ in cols] + ["cpu"])
        for p in points:
            writer.writerow([_cell(k, p.get(k)) for _, k in cols] + [cpu or ""])
    os.replace(tmp, path)
    return path


def _yes(value):
    return str(value or "").strip().lower() in ("1", "true", "yes", "y")


def read_points(path, roofs, log=print):
    """
    Computations from a CSV file with a header row, one per line:
    - name, flops, bytes, seconds (or name, gflops, ai)
    - optional: parallel, working_set_mb, read_only (yes/no), unit (FLOP or OP)
    """
    pts = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for line, raw in enumerate(csv.DictReader(fh), 2):
            row = {(k or "").strip().lower(): (v.strip() if isinstance(v, str) else "")
                   for k, v in raw.items()}

            def num(key):
                return float(row[key]) if row.get(key) else None
            try:
                ws = num("working_set_mb")
                pts.append(analyze_point(row.get("name") or f"line {line}", roofs, flops=num("flops"),
                                         bytes_moved=num("bytes"), seconds=num("seconds"),
                                         gflops=num("gflops"), ai=num("ai"),
                                         parallel=_yes(row.get("parallel")),
                                         working_set=ws * 2**20 if ws else None,
                                         read_only=_yes(row.get("read_only")),
                                         unit=row.get("unit") or "FLOP"))
            except ValueError as exc:
                log(f"warning: {path}, line {line} skipped: {exc}")
    return pts


# ---------------------------------------------------------
# Example: two small Numba kernels, timed with measure()
# ---------------------------------------------------------
def example_points(roofs, log=print):
    """A GEMV (memory-bound) and a simple matmul (compute-bound) on this machine."""
    import numpy as np
    from numba import njit

    @njit(fastmath=True)
    def gemv(A, x, y):
        # y = A x: each element of A is used once, so A streams from memory
        n, m = A.shape
        for i in range(n):
            s = 0.0
            for j in range(m):
                s += A[i, j] * x[j]
            y[i] = s

    @njit(fastmath=True)
    def matmul_ikj(A, B, C):
        # C += A B in i-k-j order: every element of A, B and C is used n times
        n = A.shape[0]
        for i in range(n):
            for k in range(n):
                a = A[i, k]
                for j in range(n):
                    C[i, j] += a * B[k, j]

    # GEMV: A as large as the bandwidth test (at least 4x the L3), so it comes from DRAM
    l3 = (roofs.get("l3_total_mb") or 32.0) * 2**20
    n = int(math.sqrt(max(128 * 2**20, 4 * l3) / 8))
    A, x, y = np.random.rand(n, n), np.random.rand(n), np.zeros(n)
    log(f"  gemv, n={n} (A = {8 * n * n / 2**20:.0f} MiB) ...")
    # - 2 FLOPs per element of A; bytes: A once (8n^2), plus x and y (O(n))
    # - A is only read: read-only DRAM roof
    gemv_pt = measure(gemv, A, x, y, flops=2.0 * n * n, bytes_moved=8.0 * n * n + 16.0 * n,
                      name=f"gemv y = A x, n={n}", working_set=8.0 * n * n, read_only=True, roofs=roofs)
    del A
    # matmul: three 2 MiB matrices fit in L3
    # - 2 FLOPs per (i, k, j); bytes with perfect reuse: read A, B; read and write C (32 n^2)
    # - setup() gives a zeroed C before every call (C += A B changes it), untimed
    m = 512
    A, B = np.random.rand(m, m), np.random.rand(m, m)
    log(f"  matmul (i-k-j order), n={m} ...")
    mm_pt = measure(matmul_ikj, flops=2.0 * m**3, bytes_moved=32.0 * m * m, name=f"matmul ikj, n={m}",
                    setup=lambda: (A, B, np.zeros((m, m))), working_set=24.0 * m * m, roofs=roofs)
    return [gemv_pt, mm_pt]


# ---------------------------------------------------------
# Command line
# ---------------------------------------------------------
HELP_EXAMPLES = """\
examples:
  python demos/roofline.py --name my_kernel --flops 1.68e8 --bytes 4.03e8 --time 0.0135
  python demos/roofline.py --name my_kernel --gflops 12.4 --ai 0.42 --parallel
  python demos/roofline.py --points runs.csv      many computations on one chart
  python demos/roofline.py --example              time two small built-in Numba kernels
  python demos/roofline.py                        only this machine's roofs

Writes roofline.png and roofline.csv to roofline_output/ next to this script
(change with --out). How every number is calculated: README.md in this folder.
"""


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="python demos/roofline.py",
        description="Place a computation on this machine's roofline: measure the roofs, "
                    "mark it memory-bound or compute-bound, and write one chart (PNG) "
                    "and one table (CSV).",
        epilog=HELP_EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)

    comp = p.add_argument_group("the computation")
    comp.add_argument("--name", default="computation", help="its label on the chart")
    comp.add_argument("--flops", type=float, help="floating-point operations it does")
    comp.add_argument("--bytes", type=float, help="bytes it moves between memory and the core")
    comp.add_argument("--time", type=float, metavar="SEC", help="its run time in seconds")
    comp.add_argument("--gflops", type=float, help="its GFLOP/s (instead of --flops and --time)")
    comp.add_argument("--ai", type=float, metavar="FLOP/BYTE",
                      help="its arithmetic intensity (instead of --flops and --bytes)")
    comp.add_argument("--parallel", action="store_true", help="it ran on all cores (all-core roof)")
    comp.add_argument("--working-set-mb", type=float, metavar="MB",
                      help="data it works on; if that fits in L3, the L3 roof is used")
    comp.add_argument("--read-only", action="store_true",
                      help="it only reads its big array (read-only DRAM roof)")
    comp.add_argument("--unit", choices=["FLOP", "OP"], default="FLOP",
                      help="OP for integer/compare work (default: FLOP)")
    comp.add_argument("--points", metavar="CSV",
                      help="many computations: name,flops,bytes,seconds (or name,gflops,ai), "
                           "optional parallel,working_set_mb,read_only,unit")
    comp.add_argument("--example", action="store_true", help="time two small built-in Numba kernels")

    out = p.add_argument_group("output")
    out.add_argument("--out", default=str(DEFAULT_OUT), metavar="DIR",
                     help="output folder (default: roofline_output/ next to this script)")
    out.add_argument("--title", help="chart title")

    add_machine_args(p)
    return p.parse_args(argv)


def print_roofs(roofs, machine):
    m = machine or {}
    print(f"\nRoofs ({m.get('cpu', 'this machine')}):")
    print(f"  {'':<26}{'1 core':>9}{'all cores':>12}")
    for label, f in (("peak GFLOP/s", "peak"), ("DRAM GB/s, read + write", "dram"),
                     ("DRAM GB/s, read only", "dram_read"), ("L3 GB/s", "l3")):
        one, allc = roofs["1"].get(f), roofs["all"].get(f)
        if one or allc:
            print(f"  {label:<26}{fmt(one):>9}{fmt(allc):>12}   ({roofs['1'].get(f + '_src')})")
    ridge = [r["peak"] / r["dram"] if r.get("peak") and r.get("dram") else None
             for r in (roofs["1"], roofs["all"])]
    print(f"  {'ridge FLOP/byte (DRAM)':<26}{fmt(ridge[0]):>9}{fmt(ridge[1]):>12}")


def print_points(points):
    if not points:
        return
    head = (f"{'Computation':<30} {'GFLOP/s':>8} {'AI':>7} {'Roof':>8} {'% roof':>7}  "
            f"{'Roof used':<24} Verdict")
    print("\n" + head + "\n" + "-" * len(head))
    for p in points:
        used = f"{LEVEL_NAMES[p['level']]}, {'all cores' if p['cores'] == 'all' else '1 core'}"
        print(f"{p['name'][:30]:<30} {p['P']:>8.3g} {p['AI']:>7.3g} {p['roof']:>8.3g} "
              f"{100 * p['pct']:>6.0f}%  {used:<24} {p['verdict']}")


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    out = ensure_out(Path(args.out).resolve())
    machine = load_machine(out, remeasure=args.remeasure, l3_mb=args.l3_mb)
    roofs = build_roofs(machine, **roof_overrides(args))
    if not (roofs["1"]["peak"] and roofs["1"]["dram"]):
        print("No roofs: the machine could not be measured. Pass --peak-gflops and --bandwidth.")
        return 1
    print_roofs(roofs, machine)

    try:
        if args.example:
            print("\nTiming the example kernels:")
            points, title = example_points(roofs), "Roofline · two example kernels"
        elif args.points:
            points, title = read_points(args.points, roofs), f"Roofline · {Path(args.points).name}"
        elif any(v is not None for v in (args.flops, args.bytes, args.time, args.gflops, args.ai)):
            ws = args.working_set_mb * 2**20 if args.working_set_mb else None
            points = [analyze_point(args.name, roofs, flops=args.flops, bytes_moved=args.bytes,
                                    seconds=args.time, gflops=args.gflops, ai=args.ai,
                                    parallel=args.parallel, working_set=ws,
                                    read_only=args.read_only, unit=args.unit)]
            title = f"Roofline · {args.name}"
        else:
            points, title = [], "Roofline · this machine"
    except (ValueError, OSError) as exc:
        print(f"error: {exc}")
        return 2

    png = plot_roofline(points, roofs, out / "roofline.png", title=args.title or title,
                        subtitle=machine_line(machine))
    print_points(points)
    print(f"\nOutputs in {out}:\n  {png.name}")
    if points:
        print(f"  {write_csv(points, out / 'roofline.csv', cpu=machine.get('cpu')).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
