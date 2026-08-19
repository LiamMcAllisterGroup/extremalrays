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
# Description:  Benchmark on the toric Mori cone (in a basis) of the h11=491
#               Calabi-Yau: 3509 generating rays in 491 dimensions, of which
#               884 are extremal. The classical per-ray redundancy LP never
#               terminates on this input (a single infeasibility proof
#               exceeds 15 minutes in HiGHS); this implementation finishes
#               in seconds.
#
#               Run:  python benchmarks/bench_mori.py [--verify]
# -----------------------------------------------------------------------------

# stdlib imports
import argparse
import pathlib
import time

# external imports
import numpy as np

# local imports
from extremal_rays import exhaustive, verify

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
    idx = exhaustive(rays, seed_shots=seed_shots, verbose=True)
    t1 = time.time()
    print(f"\n{len(idx)} extremal rays in {t1 - t0:.1f}s")
    assert len(idx) == 884, f"expected 884 extremal rays, got {len(idx)}"

    if args.verify:
        t0 = time.time()
        ok, report = verify(rays, idx, verbose=True)
        print(f"verified: {ok} in {time.time() - t0:.1f}s")
        assert ok, report["failures"]


if __name__ == "__main__":
    main()
