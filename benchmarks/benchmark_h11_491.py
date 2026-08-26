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
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Benchmark harness for extremal-rays. Repeats every measurement,
#               discards a warmup, reports median and spread rather than one
#               number, records the machine and dependency versions that
#               produced it, and can emit JSON so results are comparable across
#               runs and machines. The classical per-ray LP baseline the README
#               compares against is runnable here too (--baseline), so the
#               comparison is reproducible instead of asserted.
#
#               Run:  python benchmarks/benchmark_h11_491.py                # default
#                     python benchmarks/benchmark_h11_491.py --repeat 5 --json out.json
#                     python benchmarks/benchmark_h11_491.py --cones caps --verify
#                     python benchmarks/benchmark_h11_491.py --baseline --time-limit 5
# -----------------------------------------------------------------------------
from __future__ import annotations

# stdlib imports
import argparse
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time

# external imports
import numpy as np

# local imports
import extremal_rays
from extremal_rays import core, exhaustive, verify

DATA = pathlib.Path(__file__).parent / "data" / "mori_rays_h11_491.npz"
CAPS = pathlib.Path(__file__).parent.parent / "tests" / "data" / "mori_cap_crosscheck.npz"

# expected extremal counts, asserted so a silent regression fails the run
EXPECTED = {"mori_491": 884, "cap_h11_15": 18, "cap_h11_20": 28, "cap_h11_25": 26}


def machine() -> dict:
    """Everything needed to compare this run against another one."""
    def version(mod):
        # highspy exposes no __version__; fall back to installed metadata
        try:
            return __import__(mod).__version__
        except Exception:
            pass
        try:
            import importlib.metadata as md
            return md.version({"flint": "python-flint"}.get(mod, mod))
        except Exception:
            return "absent"

    ram = None
    try:  # POSIX only; absent elsewhere rather than wrong
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**30
    except (ValueError, AttributeError, OSError):
        pass
    # Power state matters more than it looks: on Apple silicon, macOS Low
    # Power Mode throttles the CPU enough to change wall times by ~40% (the
    # same exact-arithmetic call measured 0.187 s on AC and 0.253-0.358 s on
    # battery with LPM on). A benchmark that does not record this invites
    # exactly the phantom-regression hunt it is meant to prevent.
    power = {}
    if platform.system() == "Darwin":
        try:
            batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                                  text=True, timeout=5).stdout
            power["source"] = ("battery" if "Battery Power" in batt
                               else "ac" if "AC Power" in batt else "unknown")
            custom = subprocess.run(["pmset", "-g", "custom"],
                                    capture_output=True, text=True,
                                    timeout=5).stdout
            for line in custom.splitlines():
                if "lowpowermode" in line:
                    power["low_power_mode"] = line.split()[-1] == "1"
                    break
        except Exception:
            power = {"source": "unknown"}

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pathlib.Path(__file__).parent, capture_output=True, text=True,
            timeout=5).stdout.strip() or None
    except Exception:
        commit = None
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gib": ram,
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "highspy": version("highspy"),
        "python_flint": version("flint"),
        "extremal_rays": extremal_rays.__version__,
        "commit": commit,
        "power": power or {"source": "unknown"},
    }


def load_cones(which: str) -> "list[tuple[str, np.ndarray]]":
    cones = []
    if which in ("all", "mori491"):
        cones.append(("mori_491", np.load(DATA)["rays"].astype(np.int64)))
    if which in ("all", "caps"):
        data = np.load(CAPS)
        for h11 in (15, 20, 25):
            cones.append((f"cap_h11_{h11}", data[f"rays_h11_{h11}"]))
    if not cones:
        raise SystemExit(f"unknown --cones value {which!r}")
    return cones


def time_calls(fn, repeat: int, warmup: int) -> "tuple[list[float], object]":
    """Run fn repeat+warmup times; return the post-warmup wall times."""
    times, out = [], None
    for k in range(repeat + warmup):
        t = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t
        if k >= warmup:
            times.append(dt)
    return times, out


def summarize(times: "list[float]") -> dict:
    return {
        "median": statistics.median(times),
        "mean": statistics.fmean(times),
        "sd": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "n": len(times),
    }


def baseline(R: np.ndarray, extremal: np.ndarray, time_limit: float,
             sample: int) -> dict:
    """
    The classical per-ray LP the README compares against: is ray i a
    non-negative combination of all the others? Vendored from CYTools via
    tests/conftest.py, run under a time limit so a non-terminating case is
    recorded as unresolved rather than hanging the harness.

    The cost is strongly BIMODAL -- a redundant ray is proved redundant by
    any feasible point, while an extremal ray needs an infeasibility proof
    for a large degenerate system -- so the two classes are sampled and
    reported separately. Sampling uniformly instead would mostly draw
    redundant rays and make the baseline look uniformly fast, which is
    exactly the misreading this mode exists to prevent.
    """
    from scipy.optimize import linprog
    rng = np.random.default_rng(0)
    ext = set(int(i) for i in extremal)
    classes = {
        "extremal": np.array(sorted(ext), dtype=int),
        "redundant": np.array([i for i in range(len(R)) if i not in ext],
                              dtype=int),
    }
    out = {"time_limit": time_limit}
    for label, pool in classes.items():
        if not len(pool):
            continue
        idx = np.sort(rng.choice(pool, min(sample, len(pool)), replace=False))
        rows, unresolved = [], 0
        for i in idx:
            others = np.delete(R, i, axis=0).astype(float)
            t = time.perf_counter()
            res = linprog(c=np.zeros(len(others)), A_eq=others.T,
                          b_eq=R[i].astype(float), bounds=[(0, None)],
                          method="highs", options={"time_limit": time_limit})
            dt = time.perf_counter() - t
            # status 2 = infeasible = extremal; 1 = hit the time limit
            rows.append({"ray": int(i), "seconds": dt, "status": int(res.status)})
            unresolved += res.status == 1
        times = [r["seconds"] for r in rows]
        out[label] = {
            "sampled": len(idx), "unresolved": unresolved,
            "median_seconds": statistics.median(times),
            "max_seconds": max(times), "rays": rows,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cones", default="mori491",
                    help="all | mori491 | caps (default: mori491)")
    ap.add_argument("--repeat", type=int, default=3,
                    help="timed runs per cone after the warmup (default: 3)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="discarded runs before timing (default: 1)")
    ap.add_argument("--workers", type=int, default=0,
                    help="n_workers passed to exhaustive (default: 0)")
    ap.add_argument("--verify", action="store_true",
                    help="run the full certificate audit and time it too")
    ap.add_argument("--baseline", action="store_true",
                    help="also time the classical per-ray LP on a sample")
    ap.add_argument("--baseline-sample", type=int, default=20,
                    help="rays sampled for --baseline (default: 20)")
    ap.add_argument("--time-limit", type=float, default=5.0,
                    help="seconds per baseline LP (default: 5)")
    ap.add_argument("--json", metavar="PATH",
                    help="write the full record here")
    ap.add_argument("--seed-shots", default="auto",
                    help="'auto', or an integer (0 disables seeding)")
    args = ap.parse_args()
    seed_shots = (args.seed_shots if args.seed_shots == "auto"
                  else int(args.seed_shots))

    env = machine()
    print("machine: {platform}, {cpu_count} cores, {ram_gib} GiB".format(**env))
    pw = env["power"]
    if pw.get("low_power_mode") or pw.get("source") == "battery":
        print(f"  WARNING: power source {pw.get('source')}, low-power mode "
              f"{pw.get('low_power_mode')} -- wall times are throttled and are "
              "NOT comparable to numbers measured on AC power")
    print("versions: python {python}, numpy {numpy}, scipy {scipy}, "
          "highspy {highspy}, extremal-rays {extremal_rays} "
          "({commit})".format(**env))
    print(f"protocol: {args.warmup} warmup + {args.repeat} timed runs per cone\n")

    record = {"machine": env, "args": vars(args), "cones": []}
    hdr = f"{'cone':14s} {'rays':>7s} {'dim':>5s} {'ext':>5s} " \
          f"{'median':>9s} {'sd':>8s} {'LPs':>6s}"
    print(hdr + ("  " + f"{'verify':>9s}" if args.verify else ""))
    print("-" * (len(hdr) + (11 if args.verify else 0)))

    failures = []
    for name, R in load_cones(args.cones):
        times, idx = time_calls(
            lambda: exhaustive(R, seed_shots=seed_shots, n_workers=args.workers),
            args.repeat, args.warmup)
        prof = dict(core.LAST_PROFILE)
        entry = {
            "cone": name, "rays": int(R.shape[0]), "dim": int(R.shape[1]),
            "n_extremal": int(len(idx)), "wall": summarize(times),
            "n_lp": int(prof["n_lp_main"] + prof["n_lp_cleanup"]),
            "profile": {k: v for k, v in prof.items()
                        if isinstance(v, (int, float))},
        }
        line = (f"{name:14s} {R.shape[0]:7d} {R.shape[1]:5d} {len(idx):5d} "
                f"{entry['wall']['median']:8.3f}s {entry['wall']['sd']:7.3f}s "
                f"{entry['n_lp']:6d}")
        if args.verify:
            vtimes, (ok, report) = time_calls(lambda: verify(R, idx), 1, 0)
            entry["verify"] = {
                "ok": bool(ok), "seconds": vtimes[0],
                "worst_membership_residual": report["worst_membership_residual"],
                "worst_separation_margin": report["worst_separation_margin"],
                "failures": report["failures"][:5],
            }
            line += f"  {vtimes[0]:8.3f}s" + ("" if ok else "  AUDIT FAILED")
            if not ok:
                failures.append(f"{name}: verify reported {len(report['failures'])} failures")
        print(line, flush=True)

        if name in EXPECTED and len(idx) != EXPECTED[name]:
            failures.append(
                f"{name}: expected {EXPECTED[name]} extremal rays, got {len(idx)}")

        if args.baseline:
            b = baseline(R, idx, args.time_limit, args.baseline_sample)
            entry["baseline"] = b
            for label in ("extremal", "redundant"):
                if label not in b:
                    continue
                c = b[label]
                print(f"{'':14s} baseline per-ray LP, {label:9s}: "
                      f"{c['sampled']:3d} sampled, median "
                      f"{c['median_seconds']:7.3f}s, {c['unresolved']:2d} "
                      f"unresolved at {b['time_limit']:g}s", flush=True)
        record["cones"].append(entry)

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(record, indent=2))
        print(f"\nwrote {args.json}")

    if failures:
        print("\nFAILURES:", *failures, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
