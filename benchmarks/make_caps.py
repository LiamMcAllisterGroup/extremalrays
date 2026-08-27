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
# Description:  Build the Mcap for a range of h11 and save the rays.
#               Being an intersection, the Mcap is a SMALLER cone than any of
#               the ones it intersects, but a far bigger problem: 20,899
#               generators at h11=50 against 333 for one Mori cone, and
#               10,026,843 at h11=491. That is what makes it the interesting
#               benchmark family.
#
#               Circuit enumeration lives outside this package, in the mcap
#               tooling; point --mcap-path at that checkout. The
#               generated rays are cached, so the benchmarks and figures can
#               be reproduced without it.
#
#               Run:  python benchmarks/make_caps.py --h11 10 20 30 40 50
# -----------------------------------------------------------------------------
from __future__ import annotations

# stdlib imports
import argparse
import os
import pathlib
import sys
import time

# external imports
import numpy as np

HERE = pathlib.Path(__file__).parent
OUT = HERE / "data" / "mori_caps_by_h11.npz"
DEFAULT_MCAP = pathlib.Path.home() / "mcap_extremal_run"
SENTINEL = 2 ** 31          # padding label in the encoded circuit rows


def cap_rays(vertices):
    """Dense ray matrix of the Mcap of the CY from this polytope."""
    from cytools import Polytope                    # before highspy: see below
    from mori_cap_rework import (_cap_data, _encode_circuits,
                                 _origin_circuits, _two_face_circuits,
                                 _unique_rows)

    p = Polytope([list(map(int, v)) for v in vertices]).dual()
    if not p.is_favorable(lattice="N"):
        raise ValueError("non-favorable")
    cy = p.triangulate().cy()
    pts_ext, facets, _triangles, lookup, n_lab = _cap_data(cy)
    faces = cy.triangulation().simplices(on_faces_dim=2, split_by_face=True,
                                         as_np_array=False)
    two = sorted(_two_face_circuits(pts_ext, faces), key=sorted)
    rows = _unique_rows(np.vstack([_encode_circuits(two),
                                   _origin_circuits(pts_ext, faces, facets)]))
    half = rows.shape[1] // 2
    dense = np.zeros((rows.shape[0], n_lab), dtype=np.int64)
    for k in range(rows.shape[0]):
        for j in range(half):
            label = rows[k, j]
            if label != SENTINEL:
                dense[k, lookup[label]] = rows[k, j + half]
    return dense, int(cy.h11())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h11", type=int, nargs="+", required=True)
    ap.add_argument("--mcap-path", default=str(DEFAULT_MCAP),
                    help="checkout providing the circuit enumeration")
    ap.add_argument("--tries", type=int, default=15,
                    help="polytopes to try per h11 before giving up")
    args = ap.parse_args()
    sys.path.insert(0, args.mcap_path)

    import pandas as pd
    import pyarrow.parquet as pq
    from make_cones import polytope_table                     # noqa: F401

    frames = [pq.read_table(HERE / "data"
                            / f"polytopes-4d-{k:02d}-vertices.parquet"
                            ).to_pandas() for k in (5, 6, 7)]
    df = pd.concat(frames, ignore_index=True)
    saved = dict(np.load(OUT, allow_pickle=True)) if os.path.exists(OUT) else {}

    for h11 in args.h11:
        key = f"cap_h11_{h11}"
        if key in saved:
            print(f"h11={h11:4d}: cached {saved[key].shape}", flush=True)
            continue
        rows = df[df.h11 == h11].sort_values("point_count")
        found, t0 = None, time.perf_counter()
        for _, cand in rows.head(args.tries).iterrows():
            try:
                found, _real = cap_rays(cand.vertices)
                break
            except Exception:                                  # noqa: BLE001
                continue
        if found is None:
            print(f"h11={h11:4d}: no usable polytope", flush=True)
            continue
        saved[key] = found
        print(f"h11={h11:4d}: cap {found.shape}  "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
        np.savez_compressed(OUT, **saved)
    print(f"{OUT}: {sum(1 for k in saved if k.startswith('cap_'))} caps")


if __name__ == "__main__":
    main()
