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
# Description:  Shared test helpers: random pointed cones, brute-force
#               references (the vendored CYTools per-ray LP and an index-level
#               variant for float input), and small exact constructions with
#               known extremal sets.
# -----------------------------------------------------------------------------

# external imports
import numpy as np
from scipy.optimize import linprog


def cytools_reference(R):
    """Reference implementation: per-ray LP extremality, extracted from
    CYTools (github.com/LiamMcAllisterGroup/cytools, src/cytools/cone.py,
    Cone.extremal_rays / is_extremal with method="lp", GPL-3.0-or-later
    like this package): ray i is extremal iff (R \\ r_i) lmbda = r_i,
    lmbda >= 0 is infeasible; known-redundant rays are pruned from the
    comparison set as in the original. Serialized, and one deviation: an
    LP failure raises instead of counting as extremal (the original
    treats any non-success as extremality). Returns the extremal rays as
    a set of tuples. Only viable for small inputs."""
    rays = np.array(sorted({tuple(r) for r in np.asarray(R, dtype=int)}))
    flags = np.ones(len(rays), dtype=bool)
    for i in range(len(rays)):
        others = np.delete(rays, i, axis=0)[np.delete(flags, i)]
        res = linprog(
            c=np.zeros(len(others)),
            A_eq=others.T.astype(float),
            b_eq=rays[i].astype(float),
            bounds=[(0, None)],
            method="highs",
        )
        assert res.status in (0, 2), f"reference LP failed: {res.message}"
        if res.status == 0:  # feasible: a combination of the others
            flags[i] = False
    return {tuple(r) for r in rays[flags]}


def brute_force_indices(R):
    """Index-level per-ray LP reference for float or integer input whose
    rows are pairwise non-parallel and nonzero: the sorted indices of rays
    that are not non-negative combinations of the others."""
    R = np.asarray(R, dtype=float)
    keep = []
    for i in range(len(R)):
        others = np.delete(R, i, axis=0)
        res = linprog(c=np.zeros(len(others)), A_eq=others.T, b_eq=R[i],
                      bounds=[(0, None)], method="highs")
        assert res.status in (0, 2), f"reference LP failed: {res.message}"
        if res.status == 2:
            keep.append(i)
    return keep


def random_pointed_rays(seed, n=40, d=4, lo=-5, hi=6):
    """Unique primitive integer rays with first coordinate >= 1 (so the
    cone is pointed: w = e_0 is a positive functional), sorted."""
    rng = np.random.default_rng(seed)
    R = np.column_stack([rng.integers(1, 6, n),
                         rng.integers(lo, hi, (n, d - 1))])
    g = np.gcd.reduce(np.abs(R), axis=1)
    return np.unique(R // g[:, None], axis=0)


def random_float_rays(seed, n=40, d=4):
    """Generic float rays with first coordinate in [1, 2] (pointed)."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(1, 2, n),
                            rng.standard_normal((n, d - 1))])


def random_unimodular(seed, d):
    """A random element of GL(d, Z) built from elementary operations."""
    rng = np.random.default_rng(seed)
    M = np.eye(d, dtype=np.int64)
    for _ in range(4 * d):
        i, j = rng.choice(d, 2, replace=False)
        M[i] += rng.integers(-2, 3) * M[j]
    perm = rng.permutation(d)
    return M[perm]


def grid_face(k=5):
    """A k x k integer grid on the face x0 = 1 (heavy ray-shooting ties);
    returns (rays, sorted indices of the 4 corners)."""
    R = [(1, i, j) for i in range(k) for j in range(k)]
    corners = sorted(
        m for m, (_, i, j) in enumerate(R)
        if i in (0, k - 1) and j in (0, k - 1)
    )
    return np.array(R), corners
