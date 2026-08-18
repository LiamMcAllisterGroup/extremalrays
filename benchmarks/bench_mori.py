"""Benchmark on the toric Mori cone (in basis) of the h11=491 Calabi-Yau.

3509 generating rays in 491 dimensions; 884 are extremal. The classical
per-ray redundancy LP never terminates on this input (a single infeasibility
proof exceeds 15 minutes in HiGHS); this implementation finishes in seconds.

Run:  python benchmarks/bench_mori.py [--verify]
"""

import argparse
import pathlib
import time

import numpy as np

from extremal_rays import extremal_rays, verify_extremal_rays

DATA = pathlib.Path(__file__).parent / "data" / "mori_rays_h11_491.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="run the full certificate audit afterwards")
    ap.add_argument("--seed-shots", default="auto",
                    help="'auto', or an integer (0 disables seeding)")
    args = ap.parse_args()
    seed_shots = args.seed_shots if args.seed_shots == "auto" else int(args.seed_shots)

    rays = np.load(DATA)["rays"].astype(np.int64)
    print(f"rays: {rays.shape}")

    t0 = time.time()
    idx = extremal_rays(rays, seed_shots=seed_shots, verbose=True)
    t1 = time.time()
    print(f"\n{len(idx)} extremal rays in {t1 - t0:.1f}s")
    assert len(idx) == 884, f"expected 884 extremal rays, got {len(idx)}"

    if args.verify:
        t0 = time.time()
        ok, report = verify_extremal_rays(rays, idx, verbose=True)
        print(f"verified: {ok} in {time.time() - t0:.1f}s")
        assert ok, report["failures"]


if __name__ == "__main__":
    main()
