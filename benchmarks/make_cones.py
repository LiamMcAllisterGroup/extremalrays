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
# Description:  Build the benchmark family: for a range of h11, take a
#               4d reflexive polytope, triangulate it, and record the toric
#               Mori cone of the resulting CY hypersurface in a basis. Both
#               the polytope vertices and the cone rays are saved, so the
#               comparison benchmark runs without CYTools or a network.
#
#               Polytopes come from the Kreuzer-Skarke classification
#               (arXiv:hep-th/0002240), via the calabi-yau-data/polytopes-4d
#               dataset on Hugging Face (CC BY-SA 4.0). The 5-vertex file is
#               used: it is small and its h11 already spans 1..491.
#
#               Run:  python benchmarks/make_cones.py --h11 2 3 5 10 20
# -----------------------------------------------------------------------------
from __future__ import annotations

# stdlib imports
import argparse
import pathlib
import time
import urllib.request

# external imports
import numpy as np

HF_BASE = ("https://huggingface.co/datasets/calabi-yau-data/polytopes-4d/"
           "resolve/main/polytopes-4d-{:02d}-vertices.parquet")
HERE = pathlib.Path(__file__).parent
OUT = HERE / "data" / "mori_cones_by_h11.npz"

# vertex-count files to search, cheapest first. 5 vertices already spans
# h11 = 1..491, but has no favorable polytope in much of 36..124, which the
# 6- and 7-vertex files fill in
VERTEX_FILES = (5, 6, 7)


def polytope_table(vertex_counts=VERTEX_FILES):
    """Reflexive 4d polytopes from the named files, downloading once each."""
    import pandas as pd
    import pyarrow.parquet as pq
    frames = []
    for k in vertex_counts:
        cache = HERE / "data" / f"polytopes-4d-{k:02d}-vertices.parquet"
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {HF_BASE.format(k)}")
            urllib.request.urlretrieve(HF_BASE.format(k), cache)
        frames.append(pq.read_table(cache).to_pandas())
    return pd.concat(frames, ignore_index=True)


def mori_rays(vertices):
    """
    Toric Mori cone rays (in a basis) of the CY from this polytope.

    The CY lives in a triangulation of the DUAL polytope, and only favorable
    ones are used; non-favorable cases need CYTools' experimental features
    and are a different object than the h11 column describes.
    """
    from cytools import Polytope
    p = Polytope([list(map(int, v)) for v in vertices]).dual()
    if not p.is_favorable(lattice="N"):
        raise ValueError("non-favorable")
    cy = p.triangulate().cy()
    return np.asarray(cy.toric_mori_cone(in_basis=True).rays()), int(cy.h11())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h11", type=int, nargs="+", required=True,
                    help="h11 values to build")
    ap.add_argument("--per-h11", type=int, default=3,
                    help="polytopes to sample at each h11 (default 3). More "
                         "than one matters: cone size varies a lot between "
                         "polytopes at the same h11, so a single sample makes "
                         "a benchmark curve jump around.")
    ap.add_argument("--tries", type=int, default=40,
                    help="candidates to test per h11 before giving up")
    args = ap.parse_args()

    df = polytope_table()
    saved = dict(np.load(OUT, allow_pickle=True)) if OUT.exists() else {}
    for h11 in args.h11:
        have = [k for k in saved if k.startswith(f"rays_h11_{h11}_")]
        if len(have) >= args.per_h11:
            print(f"h11={h11:4d}  cached ({len(have)} samples)")
            continue
        rows = df[df.h11 == h11].sort_values("point_count")
        if not len(rows):
            print(f"h11={h11:4d}  no polytope with this h11")
            continue
        found, seen_shapes, t = len(have), set(), time.perf_counter()
        for _, cand in rows.head(args.tries).iterrows():
            if found >= args.per_h11:
                break
            try:
                R, got = mori_rays(cand.vertices)
            except Exception:                          # noqa: BLE001, S112
                continue
            if R.shape in seen_shapes:   # near-duplicate cone: keep variety
                continue
            seen_shapes.add(R.shape)
            saved[f"rays_h11_{h11}_{found}"] = R.astype(np.int64)
            saved[f"vertices_h11_{h11}_{found}"] = np.array(
                [list(map(int, v)) for v in cand.vertices], dtype=np.int64)
            found += 1
        if found == len(have):
            print(f"h11={h11:4d}  no usable polytope in {args.tries} tries")
            continue
        shapes = [saved[f"rays_h11_{h11}_{i}"].shape for i in range(found)]
        print(f"h11={h11:4d}  {found} samples, rays {shapes}  "
              f"({time.perf_counter() - t:.1f}s)", flush=True)
        np.savez_compressed(OUT, **saved)
    n = sum(1 for k in saved if k.startswith("rays_"))
    print(f"\n{OUT}: {n} cones")


if __name__ == "__main__":
    main()
