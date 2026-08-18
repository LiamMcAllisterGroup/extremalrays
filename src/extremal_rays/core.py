"""Clarkson-style computation of the extremal rays of a pointed polyhedral cone.

Given rays R (rows), find the unique minimal subset generating the same cone.

The classical approach tests each ray for redundancy against all n-1 others,
so proving a ray extremal requires an infeasibility certificate for a large,
typically degenerate LP -- catastrophically slow in high dimension (LP solvers
can grind for hours on a single such proof). This module instead tests each
candidate only against the set E of *confirmed* extremal rays via a small
separation LP that is always feasible and bounded:

    maximize c.p  subject to  c.e <= 0 for e in E,  -1 <= c <= 1

Optimal value 0 means p is in cone(E) (Farkas), hence redundant. A positive
value yields a separating functional c, which is used to "ray shoot": the
(lexicographically tie-broken) maximizer of c.s over the remaining candidates
is guaranteed extremal and joins E; the candidate is then retested. Every
failed test permanently grows E, so the total LP count is at most n + s
(s = number of extremal rays), each LP having at most s rows. Extremality is
never established through an infeasibility proof -- only constructively, by
being the maximizer of an explicit functional.

Reference: K. L. Clarkson, "More output-sensitive geometric algorithms" (1994).
"""

import warnings
from fractions import Fraction

import numpy as np
import highspy
from scipy.optimize import linprog

_INF = highspy.kHighsInf


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def _as_integer(R):
    """Return an integer copy of R, or None if R is genuinely non-integral."""
    if np.issubdtype(R.dtype, np.integer):
        return R.astype(np.int64)
    Rr = np.round(R)
    if np.allclose(R, Rr, rtol=0, atol=1e-9):
        return Rr.astype(np.int64)
    return None


def _unique_primitive(R):
    """Reduce rays to unique primitive representatives.

    Returns (U, rep) where U are the unique (primitive, for integer input)
    rays and rep[k] is the index in the original array of the first ray with
    that direction. Zero rays are dropped.
    """
    Ri = _as_integer(R)
    if Ri is not None:
        g = np.gcd.reduce(np.abs(Ri), axis=1)
        nonzero = g > 0
        prim = Ri[nonzero] // g[nonzero, None]
        orig = np.flatnonzero(nonzero)
    else:
        norms = np.linalg.norm(R, axis=1)
        nonzero = norms > 1e-12
        prim = R[nonzero] / norms[nonzero, None]
        orig = np.flatnonzero(nonzero)

    seen = {}
    rep = []
    keep = []
    for k, row in enumerate(prim):
        key = tuple(row)
        if key not in seen:
            seen[key] = True
            keep.append(k)
            rep.append(orig[k])
    return prim[keep].astype(float), np.array(rep, dtype=int)


def positive_functional(R):
    """Find w with R @ w >= 1 for every row of R.

    Such a w exists iff cone(R) is pointed (contained in an open halfspace).
    Raises ValueError otherwise. The returned w defines the affine slice
    w.x = 1 on which rays are normalized to points.
    """
    n, d = R.shape
    res = linprog(
        c=R.sum(axis=0).astype(float),
        A_ub=-R.astype(float),
        b_ub=-np.ones(n),
        bounds=[(None, None)] * d,
        method="highs",
    )
    if not res.success:
        raise ValueError(
            "Cone is not pointed (no functional is strictly positive on all "
            "rays). Decompose into lineality space + pointed quotient first."
        )
    w = res.x
    slack = R @ w
    if slack.min() <= 0.5:
        raise RuntimeError(
            f"positive-functional certificate failed (min slack {slack.min():.3e})"
        )
    return w


# ---------------------------------------------------------------------------
# separation oracle: one persistent HiGHS model, warm-started across calls
# ---------------------------------------------------------------------------

class _SeparationOracle:
    """Persistent LP  max c.p  s.t.  c.e <= 0 (e in E),  -1 <= c <= 1.

    Rows are added as E grows; only the objective changes between candidate
    tests, so HiGHS warm-starts from the previous basis. Rows can be relaxed
    (bounds widened to free) and restored, which is how the cleanup pass
    tests a confirmed ray against the others without rebuilding the model.
    """

    def __init__(self, d):
        self.d = d
        self._col_idx = np.arange(d, dtype=np.int32)
        self.h = highspy.Highs()
        self.h.silent()
        self.h.addVars(d, np.full(d, -1.0), np.full(d, 1.0))
        self.row_of = {}

    def add_row(self, e, key):
        nz = np.flatnonzero(e).astype(np.int32)
        self.h.addRow(-_INF, 0.0, len(nz), nz, e[nz].astype(float))
        self.row_of[key] = len(self.row_of)

    def relax(self, key):
        self.h.changeRowBounds(self.row_of[key], -_INF, _INF)

    def restore(self, key):
        self.h.changeRowBounds(self.row_of[key], -_INF, 0.0)

    def separate(self, p):
        """Return (val, c) with val = max c.p. val ~ 0 iff p in cone(E).

        Raises on any non-optimal solver status: a solver failure must never
        be silently interpreted as a verdict.
        """
        self.h.changeColsCost(self.d, self._col_idx, (-p).astype(float))
        self.h.run()
        status = self.h.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"separation LP not solved to optimality: {status}")
        val = -self.h.getInfo().objective_function_value
        c = np.array(self.h.getSolution().col_value, dtype=float)
        return val, c


# ---------------------------------------------------------------------------
# ray shooting
# ---------------------------------------------------------------------------

def _shoot(P, c, cand, rel_tol=1e-9):
    """Index (from cand) of the lexicographically tie-broken maximizer of
    P[cand] @ c. In exact arithmetic the result is a vertex of conv(P)."""
    vals = P[cand] @ c
    atol = rel_tol * max(1.0, float(np.abs(vals).max()))
    T = cand[vals >= vals.max() - atol]
    k = 0
    d = P.shape[1]
    while len(T) > 1 and k < d:
        col = P[T, k]
        atol_k = rel_tol * max(1.0, float(np.abs(col).max()))
        T = T[col >= col.max() - atol_k]
        k += 1
    return int(T[0])


# ---------------------------------------------------------------------------
# exact rational fallback (integer rays only)
# ---------------------------------------------------------------------------

def _exact_membership(r, A_rows, lam_float, support_tol=1e-9):
    """One-sided exact certifier: try to confirm r = sum lam_i a_i, lam >= 0,
    over the rationals, using the support of a float LP solution. Returns
    True only on a rigorous success; False means inconclusive."""
    supp = np.flatnonzero(lam_float > support_tol)
    if len(supp) == 0:
        return bool(np.all(r == 0))
    d = len(r)
    ncol = len(supp)
    M = [
        [Fraction(int(A_rows[j][k])) for j in supp] + [Fraction(int(r[k]))]
        for k in range(d)
    ]
    piv_cols, ri = [], 0
    for cj in range(ncol):
        pr = next((q for q in range(ri, d) if M[q][cj] != 0), None)
        if pr is None:
            continue
        M[ri], M[pr] = M[pr], M[ri]
        pv = M[ri][cj]
        M[ri] = [v / pv for v in M[ri]]
        for q in range(d):
            if q != ri and M[q][cj] != 0:
                f = M[q][cj]
                M[q] = [a - f * b for a, b in zip(M[q], M[ri])]
        piv_cols.append(cj)
        ri += 1
        if ri == d:
            break
    for q in range(ri, d):
        if M[q][ncol] != 0:
            return False
    x = [Fraction(0)] * ncol
    for k, cj in enumerate(piv_cols):
        x[cj] = M[k][ncol]
    return all(v >= 0 for v in x)


# ---------------------------------------------------------------------------
# main algorithm
# ---------------------------------------------------------------------------

def extremal_rays(
    R,
    tol=1e-7,
    seed_shots="auto",
    cleanup=True,
    verbose=False,
    rng_seed=0,
):
    """Indices of a minimal generating subset of the rays R of a pointed cone.

    Arguments:
    - R: (n, d) array whose rows generate the cone. Integer input enables
      exact primitive-vector deduplication and the exact rational fallback in
      the cleanup pass.
    - tol: relative threshold on the separation LP value for deciding
      membership vs. separation.
    - seed_shots: number of random functionals shot before the main loop to
      pre-populate E cheaply (each shot is a matvec argmax, no LP). "auto"
      picks min(2d, n). 0 disables.
    - cleanup: retest each confirmed ray against the others. Floating-point
      tie-breaking in ray shooting can rarely admit a redundant ray into E;
      cleanup restores minimality and should stay on unless a slightly
      non-minimal generating set is acceptable.
    - verbose: print progress.
    - rng_seed: seed for the seeding functionals (results are deterministic
      for a fixed value).

    Returns a sorted integer array of indices into R (first occurrence for
    duplicated directions). The corresponding rows are the extremal rays.
    """
    R_in = np.asarray(R)
    if R_in.ndim != 2 or R_in.shape[0] == 0:
        raise ValueError("R must be a non-empty 2d array of rays")
    U, rep = _unique_primitive(R_in)
    n, d = U.shape
    if n == 1:
        return rep.copy()

    R_int = _as_integer(R_in)

    w = positive_functional(U)
    P = U / (U @ w)[:, None]

    # status: 0 unknown, 1 confirmed extremal, -1 confirmed redundant
    status = np.zeros(n, dtype=np.int8)
    E = []
    oracle = _SeparationOracle(d)

    def confirm(j):
        status[j] = 1
        E.append(j)
        oracle.add_row(P[j], j)

    # --- seeding: argmaxes of random functionals are vertices; no LPs needed
    if seed_shots == "auto":
        seed_shots = min(2 * d, n)
    if seed_shots:
        rng = np.random.default_rng(rng_seed)
        C = rng.standard_normal((d, seed_shots))
        all_idx = np.arange(n)
        hits = set()
        for k in range(seed_shots):
            hits.add(_shoot(P, C[:, k], all_idx))
        for j in sorted(hits):
            confirm(j)
        if verbose:
            print(f"seeding: {len(E)} extremal rays from {seed_shots} shots")

    # --- main loop
    n_lp = 0
    for i in range(n):
        if status[i] != 0:
            continue
        for _ in range(n + 1):
            val, c = oracle.separate(P[i])
            n_lp += 1
            scale = max(1.0, float(np.abs(P[i]).max()))
            if val <= tol * scale:
                status[i] = -1
                break
            j = _shoot(P, c, np.flatnonzero(status == 0))
            confirm(j)
            if j == i:
                break
        else:
            raise RuntimeError(f"failed to resolve candidate {i}")
        if verbose and (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n} candidates, |E| = {len(E)}, LPs = {n_lp}")

    # --- cleanup: restore minimality lost to floating-point tie-breaking
    if cleanup:
        for e in sorted(E):
            others = [x for x in E if x != e]
            if not others:
                continue
            oracle.relax(e)
            val, _c = oracle.separate(P[e])
            n_lp += 1
            scale = max(1.0, float(np.abs(P[e]).max()))
            if val > tol * scale:
                oracle.restore(e)
                continue
            # not separable: demand a positive certificate of redundancy
            res = linprog(
                c=np.zeros(len(others)),
                A_eq=P[others].T,
                b_eq=P[e],
                bounds=[(0, None)],
                method="highs",
            )
            resid = (
                float(np.abs(P[others].T @ res.x - P[e]).max())
                if res.success else np.inf
            )
            certified = res.success and resid < 1e-6
            if not certified and R_int is not None and res.success:
                certified = _exact_membership(
                    R_int[rep[e]], R_int[rep[others]], res.x
                )
            if certified:
                E.remove(e)
                status[e] = -1
                if verbose:
                    print(f"  cleanup: removed redundant ray {e}")
            else:
                oracle.restore(e)
                warnings.warn(
                    f"ray {rep[e]} is numerically borderline (separation "
                    f"margin {val:.2e}, no membership certificate); keeping "
                    "it -- the result generates the cone but may not be "
                    "minimal. Consider verify_extremal_rays()."
                )

    if verbose:
        print(f"done: {len(E)} extremal rays, {n_lp} LPs")
    return np.sort(rep[sorted(E)])
