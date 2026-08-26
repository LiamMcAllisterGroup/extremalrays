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
# Description:  Where parallelism actually pays. The two phases have very
#               different structure: exhaustive() grows its confirmed set as
#               it sweeps, so workers necessarily test against stale
#               snapshots, while verify() asks an independent question per
#               candidate once the answer is fixed. This measures both rather
#               than assuming.
#
#               Run:  python benchmarks/bench_parallel.py [--workers 1 2 4 8]
# -----------------------------------------------------------------------------
from __future__ import annotations

# stdlib imports
import argparse
import os
import pathlib
import statistics
import time

# external imports
import numpy as np

# local imports
from extremal_rays import core, exhaustive, verify

DATA = pathlib.Path(__file__).parent / "data" / "mori_rays_h11_491.npz"


def timed(fn, repeats=3):
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    rays = np.load(DATA)["rays"].astype(np.int64)
    print(f"cone: {rays.shape}, {os.cpu_count()} cores\n")

    base, idx = timed(lambda: exhaustive(rays), args.repeats)
    print(f"exhaustive, serial: {base:7.2f}s  ({len(idx)} rays)")
    for w in args.workers:
        if w == 0:
            continue
        t, got = timed(lambda w=w: exhaustive(rays, n_workers=w), args.repeats)
        assert got.tolist() == idx.tolist(), "parallel changed the answer"
        cpu = core.LAST_PROFILE.get("total", float("nan"))
        print(f"exhaustive, {w:2d} workers: {t:7.2f}s  "
              f"speedup {base / t:5.2f}x  (profile total {cpu:.1f}s)")

    print()
    vbase, (ok, rep) = timed(lambda: verify(rays, idx), 1)
    print(f"verify, serial: {vbase:7.2f}s  ok={ok}, "
          f"worst residual {rep['worst_membership_residual']:.2e}")
    for w in args.workers:
        if w == 0:
            continue
        t, (ok_w, rep_w) = timed(lambda w=w: verify(rays, idx, n_workers=w), 1)
        agree = (ok_w == ok and rep_w["failures"] == rep["failures"])
        print(f"verify, {w:2d} workers: {t:7.2f}s  speedup {vbase / t:5.2f}x  "
              f"identical={agree}")


if __name__ == "__main__":
    main()
