# =============================================================================
#    Copyright (C) 2026  Nate MacFadden for the Liam McAllister Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Extremal rays of torically inherited Mori cones, this package
#               prior art, over a family of Calabi-Yau hypersurfaces at
#               increasing h11. Every method solves the SAME problem (given
#               generators of a pointed cone, return the minimal generating
#               subset), and every method's answer is checked against ours,
#               so a fast wrong answer cannot masquerade as a win.
#
#               Methods: this package; CYTools' per-ray LP (the incumbent);
#               lrs (reverse search, redund mode); cddlib (double description,
#               redcheck); Normaliz (the standard tool for cones in toric
#               geometry). Missing tools are skipped, not failed.
#
#               Run:  python benchmarks/benchmark_mori_cone.py --check
#                     python benchmarks/benchmark_mori_cone.py --timeout 60
#                     python benchmarks/benchmark_mori_cone.py --plot-only
# -----------------------------------------------------------------------------
from __future__ import annotations

# stdlib imports
import argparse
import json
import os
import pathlib
import signal
import shutil
import subprocess
import sys
import tempfile
import time

# external imports
import numpy as np

# CYTools runs from a pinned virtualenv in a spawned interpreter, never
# in-process. Two reasons: the comparison is against a RELEASED version, which
# whatever is importable here need not be, and ortools ships its own libhighs
# that clashes with highspy at dlopen time ("Symbol not found:
# setLocalOptionValue"), a hazard that disappears once it is out-of-process.
# Its parallelism is left at the CYTools default (all cores via joblib above
# 32 rays), because that is what a user gets

# local imports (import highspy transitively; see the note above)
import _cytools_env
from _bench import timed_median
import extremalrays
from extremalrays import exhaustive

HERE = pathlib.Path(__file__).parent
CONES = HERE / "data" / "mori_cones_by_h11.npz"
DOCS = HERE.parent / "docs"


# -----------------------------------------------------------------------------
# input formats for the external tools
# -----------------------------------------------------------------------------

def _ine(R: np.ndarray) -> str:
    """cdd/lrs V-representation; a leading 0 marks a ray rather than a point."""
    n, d = R.shape
    body = "\n".join("0 " + " ".join(str(int(x)) for x in row) for row in R)
    return (f"cone\nV-representation\nbegin\n{n} {d + 1} integer\n"
            f"{body}\nend\n")


def _nmz(R: np.ndarray) -> str:
    """Normaliz cone input."""
    n, d = R.shape
    body = "\n".join(" ".join(str(int(x)) for x in row) for row in R)
    return f"amb_space {d}\ncone {n}\n{body}\n"


def _count_vrep(text: str) -> "int | None":
    """Number of rows in the V-representation a tool printed."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "begin" and i + 1 < len(lines):
            head = lines[i + 1].split()
            if head and head[0].isdigit():
                return int(head[0])
    return None


# -----------------------------------------------------------------------------
# the methods
# -----------------------------------------------------------------------------

def run_ours(R):
    return len(exhaustive(R))


def run_cytools(R, timeout=None):
    """
    CYTools' per-ray LP, from the pinned env, in a spawned interpreter.

    Spawned rather than forked: forking a process that then starts joblib
    workers deadlocks, and a separate process is the only way to put a
    wall-clock limit on a call that has none of its own.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "rays.npy"
        np.save(path, R)
        out = _spawn([_cytools_env.interpreter(quiet=True),
                      str(HERE / "_cytools_driver.py"), str(path)],
                     timeout=timeout)
    return int(out.split()[-1])


def _spawn(argv, timeout=None):
    """Run argv in its own process group; kill the GROUP on timeout."""
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        raise
    if proc.returncode != 0:
        raise RuntimeError(err.strip().splitlines()[-1] if err.strip() else
                           f"exit {proc.returncode}")
    return out.strip()


def _external(cmd, payload, suffix, parse, timeout, outfile=None,
              stdin=False):
    """Run an external tool on `payload`, returning the parsed ray count."""
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / f"cone{suffix}"
        path.write_text(payload)
        argv = list(cmd) if stdin else [*cmd, str(path)]
        # Run in its own process group and kill the GROUP on timeout.
        # subprocess.run(timeout=) kills only the direct child, and Normaliz
        # spawns OpenMP children that survive it: three such orphans were
        # found still running 80+ minutes after their parent was killed,
        # driving the load average to 118 on a 10-core machine and inflating
        # every measurement taken alongside them
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if stdin else None, text=True, cwd=tmp,
            start_new_session=True)
        try:
            out, _err = proc.communicate(payload if stdin else None,
                                         timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.communicate()
            raise
        res = subprocess.CompletedProcess(argv, proc.returncode, out, _err)
        produced = pathlib.Path(tmp) / outfile if outfile else None
        text = (produced.read_text() if produced and produced.exists()
                else res.stdout)
        return parse(text)


def run_lrs(R, timeout):
    return _external(["lrs"], _ine(R).replace("end\n", "end\nredund 0 0\n"),
                     ".ine", _count_vrep, timeout)


def run_cdd(R, timeout):
    # cddexec takes its input on stdin; a file argument is ignored
    return _external(["cddexec", "--redcheck"], _ine(R), ".ine",
                     _count_vrep, timeout, stdin=True)


def run_normaliz(R, timeout):
    def parse(text):
        for line in text.splitlines():
            if "extreme rays" in line:
                return int(line.split()[0])
        return None
    return _external(["normaliz", "--ExtremeRays", "-f"], _nmz(R), ".in",
                     parse, timeout, outfile="cone.out")


def measure_startup(exe: str, trials: int = 15) -> float:
    """
    Seconds this binary costs to fork, exec, dynamically link and exit.

    Handing an external tool a nonexistent file makes it do everything
    except the work. Subtracting this is being deliberately generous to the
    competition: process creation is an artifact of driving a CLI from a
    benchmark, not a property of the algorithm. It matters: measured 10.8
    ms for lrs, 9.1 for cddexec and 27.4 for normaliz (OpenMP linking),
    against a 5.4 ms floor for /usr/bin/true, which is enough to BE the
    entire measurement on small cones.
    """
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        try:
            subprocess.run([exe, "/nonexistent-input-for-timing"],
                           capture_output=True, timeout=30)
        except Exception:                              # noqa: BLE001
            return 0.0
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


# a corrected time is only meaningful if the raw time stands clearly above
# the process-creation cost that was subtracted from it
RESOLUTION = 1.5

# A method with no time limit of its own (CYTools) is retired once a single
# cone costs it this long. Set generously: a lower bar retires a method that
# is merely slow rather than failing, which reads on a plot as a cliff it did
# not actually hit
RETIRE_SECONDS = 600.0


def measure_import(argv, trials: int = 5) -> float:
    """Median wall time of a command that only starts up and exits."""
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        try:
            subprocess.run(argv, capture_output=True, timeout=120, check=True)
        except Exception:                              # noqa: BLE001
            return 0.0
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def available():
    """The methods this machine can actually run."""
    methods = {"extremalrays": (run_ours, False)}
    try:
        _cytools_env.interpreter()
        methods["CYTools (per-ray LP)"] = (run_cytools, True)
    except Exception as exc:                           # noqa: BLE001
        print(f"skipping CYTools: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:110]}")
    for name, fn, exe in (("lrs", run_lrs, "lrs"),
                          ("cddlib", run_cdd, "cddexec"),
                          ("Normaliz", run_normaliz, "normaliz")):
        if shutil.which(exe):
            methods[name] = (fn, True)
        else:
            print(f"skipping {name}: {exe} not on PATH")
    return methods


# -----------------------------------------------------------------------------
# driver
# -----------------------------------------------------------------------------

def _write(path, truth, startup, results, versions=None):
    """Persist results so far; called after every h11."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"extremal": truth,
         "startup_subtracted": startup,
         # which implementations produced these numbers. Without it a result
         # is unfalsifiable: CYTools' extremal_rays is version-sensitive, and
         # v1.4.13 is expected to change it outright
         "versions": versions or {},
         "times": {k: {str(h): v for h, v in d.items()}
                   for k, d in results.items()}}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="seconds per method per cone (default 60)")
    ap.add_argument("--max-h11", type=int, default=10**9)
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure from the existing results")
    ap.add_argument("--check", action="store_true",
                    help="verify every method agrees on the smaller cones")
    ap.add_argument("--json", default=str(HERE.parent / "perf-work"
                                          / "prior_art.json"))
    ap.add_argument("--cones", dest="cones_file", default=str(CONES),
                    help="npz of cones; keys rays_h11_<h>_<i> or cap_h11_<h>")
    args = ap.parse_args()

    if args.plot_only:
        subprocess.run(
            [sys.executable, str(HERE / "_plot.py"), "--json", args.json,
             "--out", str(HERE.parent / "docs" / "benchmark_prior_art.png")],
            check=True)
        return None, None

    data = np.load(args.cones_file)
    by_h11 = {}
    for k in data.files:
        if k.startswith("rays_h11_") or k.startswith("cap_h11_"):
            h = int(k.split("_")[2])
            by_h11.setdefault(h, []).append(k)
    h11s = sorted(h for h in by_h11 if h <= args.max_h11)
    for h in h11s:
        by_h11[h].sort()

    methods = available()
    exes = {"lrs": "lrs", "cddlib": "cddexec", "Normaliz": "normaliz"}
    startup = {name: measure_startup(exe) for name, exe in exes.items()
               if name in methods}
    versions = {"extremalrays": extremalrays.__version__}
    if "CYTools (per-ray LP)" in methods:
        py = _cytools_env.interpreter(quiet=True)
        versions["CYTools"] = _cytools_env.installed_version(py)
        # interpreter start-up plus the cytools import, which is seconds, not
        # milliseconds; charging it to the algorithm would be a real distortion
        startup["CYTools (per-ray LP)"] = measure_import(
            [py, str(HERE / "_cytools_driver.py"), "--import-only"])
    print(f"methods: {', '.join(methods)}")
    if startup:
        print("process startup subtracted from external tools: "
              + ", ".join(f"{k} {1000 * v:.1f} ms" for k, v in startup.items()))
    print(f"cones: {sum(len(v) for v in by_h11.values())} over "
          f"h11 = {h11s[0]}..{h11s[-1]}, up to "
          f"{max(len(v) for v in by_h11.values())} polytopes each\n")

    results = {name: {} for name in methods}
    dead = set()
    truth = {}
    for h11 in h11s:
        cones = [data[k] for k in by_h11[h11]]
        exts = [run_ours(R) for R in cones]
        truth[h11] = int(np.median(exts))
        sizes = f"n={int(np.median([c.shape[0] for c in cones])):5d}"
        row = [f"h11={h11:4d} {sizes} d={cones[0].shape[1]:4d} "
               f"ext~{truth[h11]:5d} [{len(cones)}]"]
        for name, (fn, takes_timeout) in methods.items():
            if name in dead:
                row.append(f"{name}: -")
                continue
            per_cone = []
            for R, n_ext in zip(cones, exts):
                call = ((lambda f=fn, r=R: f(r, args.timeout)) if takes_timeout
                        else (lambda f=fn, r=R: f(r)))
                try:
                    if call() != n_ext:
                        dead.add(name)
                        break
                    med, _lo, _hi = timed_median(
                        call, warmup=1, repeats=5,
                        max_total=min(30.0, args.timeout))
                except subprocess.TimeoutExpired:
                    # retire on the first timeout and skip every larger h11:
                    # hunting for a completable instance costs a full cap per
                    # attempt and tells us nothing new
                    dead.add(name)
                    break
                except Exception:                      # noqa: BLE001
                    dead.add(name)
                    break
                off = startup.get(name, 0.0)
                if off and med < RESOLUTION * off:
                    continue          # not measurable above process creation
                per_cone.append(max(med - off, 0.0))
            if name in dead:
                row.append(f"{name}: retired")
                continue
            if not per_cone:
                row.append(f"{name}: <resolution")
                continue
            # the spread ACROSS polytopes at this h11 is the honest error bar:
            # cone size and structure vary a lot at fixed h11
            med = float(np.median(per_cone))
            results[name][h11] = (med, min(per_cone), max(per_cone))
            row.append(f"{name}: {med:.3f}s")
            if med > args.timeout or (not takes_timeout
                                      and med > RETIRE_SECONDS):
                dead.add(name)
        print("  ".join(row), flush=True)
        # write after every h11 so a long run can be plotted while it runs
        _write(args.json, truth, startup, results, versions)

    _write(args.json, truth, startup, results, versions)
    print(f"\nwrote {args.json}")
    return results, truth


if __name__ == "__main__":
    main()
