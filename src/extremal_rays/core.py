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
# Description:  This module computes the extremal rays of pointed polyhedral
#               cones with Clarkson's output-sensitive algorithm. Candidates
#               are tested against the confirmed-extremal set via small,
#               always-feasible separation LPs; extremality is established
#               constructively by ray shooting, never by an infeasibility
#               proof.
# -----------------------------------------------------------------------------
"""
Extremal rays of pointed polyhedral cones, via Clarkson's output-sensitive
algorithm.

The public entry point is :func:`exhaustive`, which returns the indices of
the unique minimal generating subset of the input rays. Candidates are
tested only against the set of already-confirmed extremal rays, using small
separation LPs that are always feasible and bounded; extremality is
established constructively by ray shooting and never through an
infeasibility proof. :func:`positive_functional` exposes the pointedness
certificate the algorithm builds on.

The two workhorses are :class:`_SeparationOracle` (is this ray outside the
cone of the confirmed set?) and :class:`_MembershipOracle` (is this ray a
non-negative combination of those rays?). Both are persistent HiGHS models
and both carry docstrings explaining the numerical failure modes that shaped
them; start there when changing anything about tolerances.

A wall-time and LP-count breakdown of the most recent :func:`exhaustive`
call is left in :data:`LAST_PROFILE`; ``closest_member`` and
``closest_separated`` in particular say how much work the tolerance did.
"""
from __future__ import annotations

# stdlib imports
import hashlib
import os
import time
import warnings
from fractions import Fraction

# external imports
import numpy as np
from numpy.typing import ArrayLike
import highspy
from scipy import sparse
from scipy.optimize import linprog

_INF = highspy.kHighsInf

# wall-time breakdown of the most recent exhaustive() call, in seconds
LAST_PROFILE: dict = {}

_CKPT_HINT_DELAY = 30.0      # seconds of sweeping before estimating
_CKPT_HINT_REMAINING = 600.0  # warn if this much work remains uncheckpointed

_CG_THRESHOLD = 200_000  # rows above which the pointedness LP uses
                         # constraint generation (direct HiGHS fails there)
_CG_BATCH = 8            # subset and growth size, in multiples of dim
_CG_ROUNDS = 60          # rounds before giving up (suggest w=)

_CLEANUP_LP_TIME_LIMIT = 60.0  # seconds per membership-certificate LP

# HiGHS options for the separation oracle. Only the objective changes between
# candidate tests, which leaves the incumbent basis PRIMAL feasible and dual
# infeasible, so primal simplex (strategy 4) resumes from it directly, while
# the default (1, dual plain) must re-establish dual feasibility every solve.
# Measured on the h11=491 Mori cone: 12.6 s vs 22.2 s for 3123 LPs, with a
# byte-identical index set, and identical answers across 49 cones (fixtures,
# random, tie-heavy, float, sparse, degenerate). `presolve="off"` is
# deliberately NOT set: it adds only ~2% on top of this (inside noise) and is
# slower than the default on its own, which does not justify giving up a
# numerical safety net on degenerate input
_HIGHS_OPTIONS = {"simplex_strategy": 4}

# residual below which a membership reconstruction counts as certified.
# Applied to the unit-normalized question (see _MembershipOracle), so it is
# scale-free; a ray is only ever REMOVED on such a certificate, making a
# strict value the sound direction (at worst the result is non-minimal)
_MEMBERSHIP_RESID_TOL = 1e-9

# The ambiguous band. A separation value of exactly zero means "in the cone";
# genuinely redundant rays land at the solver's noise floor, ~1e-13 relative
# to a unit-normalized ray. Anything meaningfully above that but still under
# tol is a ray the tolerance, not the geometry, is about to discard, and
# the failure is silent and one-directional: the dropped ray is extremal and
# the result no longer generates the cone.
#
# Values seen in the band: 7.07e-10 for the third generator of the simplicial
# cone [[1e9,0,0],[0,1e9,0],[1e9,1e9,1]] (dropped, and the old audit blessed
# the loss), 3.75e-08 for a 12-ray cone spanning 1e6. Both are >= 1e-12 while
# true members stay near 1e-13, so this floor separates them by orders of
# magnitude rather than by a hair.
#
# Verdicts inside the band are re-decided in exact rational arithmetic when
# the rays are integral, and always reported
_AMBIGUOUS_BAND = 1e-12

# ray shooting takes a candidate submatrix instead of a full matvec once the
# unresolved set is below n / this. See _shoot
_SUBMATRIX_SHARE = 10


# -----------------------------------------------------------------------------
# preprocessing
# -----------------------------------------------------------------------------

def _as_integer(R: np.ndarray) -> "np.ndarray | None":
    """
    Return an integer copy of ``R``, or None if ``R`` is genuinely
    non-integral.

    The test is RELATIVE (rtol, atol=0), so an entry only counts as
    integral when it is close to its own rounded value in proportion to
    that value. An absolute tolerance here silently reinterprets small
    float data as integers: with atol=1e-9, [[1, 1e-10], [1, -1e-10]]
    snapped to two copies of [1, 0] (distinct rays collapsed into one)
    and [[1e-10, 2e-10]] snapped to the zero ray (annihilating a cone),
    so a legitimate float cone scaled by 1e-10 raised "no nonzero ray"
    while the same cone at scale 1 worked. With atol=0 a nonzero entry
    can never round to zero, and genuinely near-integral floats (the
    reason to snap at all, e.g. 3 arriving as 2.9999999999999996 from a
    matrix product) still qualify.
    """
    if np.issubdtype(R.dtype, np.integer):
        return R.astype(np.int64)
    Rr = np.round(R)
    if (np.allclose(R, Rr, rtol=1e-12, atol=0.0)
            and np.abs(Rr).max(initial=0.0) < 2.0 ** 53):
        return Rr.astype(np.int64)
    return None


def _row(P, i: int) -> np.ndarray:
    """Row ``i`` of a dense or CSR matrix as a dense 1d array."""
    if sparse.issparse(P):
        lo, hi = P.indptr[i], P.indptr[i + 1]
        out = np.zeros(P.shape[1], dtype=P.data.dtype)
        out[P.indices[lo:hi]] = P.data[lo:hi]
        return out
    return P[i]


def _rows(P, idx) -> np.ndarray:
    """Rows ``idx`` of a dense or CSR matrix as a dense 2d array."""
    if sparse.issparse(P):
        return P[idx].toarray()
    return P[idx]


def _padded_key(U: "sparse.csr_matrix") -> np.ndarray:
    """
    Fixed-width row keys for a CSR matrix, packed into few uint64 columns.

    Each row becomes its column indices (padded with ``dim``, packed
    several per word) followed by one word per value, the float's bits
    mapped so that unsigned order equals numeric order. Distinct rows get
    distinct keys, and lexicographic key order is lexicographic
    (support, values) row order.
    """
    n, d = U.shape
    nnz = np.diff(U.indptr)
    width = int(nnz.max()) if n else 0
    idx = np.full((n, width), d, dtype=np.uint64)
    val = np.zeros((n, width), dtype=np.float64)
    rows = np.repeat(np.arange(n), nnz)
    cols = np.arange(U.nnz) - np.repeat(U.indptr[:-1], nnz)
    idx[rows, cols] = U.indices
    val[rows, cols] = U.data

    bits = max(1, int(d).bit_length())
    per = max(1, 63 // bits)
    packed = []
    for start in range(0, width, per):
        key = np.zeros(n, dtype=np.uint64)
        for j in range(start, min(start + per, width)):
            key = (key << np.uint64(bits)) | idx[:, j]
        packed.append(key)

    b = val.view(np.int64)
    monotone = np.where(b < 0, ~b, b ^ np.int64(-2**63)).view(np.uint64)
    return np.column_stack(packed + [monotone[:, j] for j in range(width)])


def _first_unique(keys: np.ndarray) -> np.ndarray:
    """
    Row indices of the first occurrence of each distinct key row, ascending.

    One scalar argsort on the leading key finds candidate duplicate runs;
    only rows inside a run get the exact multi-key comparison. Duplicates
    keep the copy with the smallest row index.
    """
    n = len(keys)
    order = np.argsort(keys[:, 0], kind="stable")
    k0 = keys[order, 0]
    eq_prev = np.zeros(n, dtype=bool)
    eq_prev[1:] = (k0[1:] == k0[:-1])
    in_run = eq_prev.copy()
    in_run[:-1] |= eq_prev[1:]
    rr = order[in_run]

    mask = np.ones(n, dtype=bool)
    if len(rr):
        sub = keys[rr]
        o2 = np.lexsort((rr,) + tuple(sub.T[::-1]))
        subs = sub[o2]
        dup = (subs[1:] == subs[:-1]).all(axis=1)
        mask[rr[o2[1:][dup]]] = False
    return np.flatnonzero(mask)


def _unique_primitive_sparse(
    R: "sparse.spmatrix",
) -> "tuple[sparse.csr_matrix, np.ndarray]":
    """
    Sparse counterpart of ``_unique_primitive``: unique primitive (integer)
    or unit-norm (float) representatives, zero rays dropped, as float CSR.
    """
    R = sparse.csr_matrix(R, copy=True)
    R.sum_duplicates()
    R.eliminate_zeros()
    R.sort_indices()
    nnz = np.diff(R.indptr)
    nonzero = nnz > 0
    orig = np.flatnonzero(nonzero)
    starts = R.indptr[:-1][nonzero]

    data_int = _as_integer(R.data)
    if data_int is not None:
        g = np.gcd.reduceat(np.abs(data_int), starts)
        data = (data_int // np.repeat(g, nnz[nonzero])).astype(float)
    else:
        norms = np.sqrt(np.add.reduceat(R.data.astype(float) ** 2, starts))
        data = R.data / np.repeat(norms, nnz[nonzero])

    indptr = np.zeros(len(orig) + 1, dtype=np.int64)
    np.cumsum(nnz[nonzero], out=indptr[1:])
    U = sparse.csr_matrix((data, R.indices, indptr),
                          shape=(len(orig), R.shape[1]))

    first = _first_unique(_padded_key(U))
    return U[first], orig[first]


def _unique_primitive(R: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """
    Reduce rays to unique primitive representatives.

    For integer input, each ray is divided by the GCD of its entries (exact);
    float rays are normalized to unit length instead. Zero rays are dropped.

    Returns
    -------
    U : ndarray of shape (n_unique, dim)
        The unique reduced rays, as floats.
    rep : ndarray of shape (n_unique,)
        rep[k] is the index in the original array of the first ray with
        direction U[k].
    """
    Ri = _as_integer(R)
    if Ri is not None:
        g = np.gcd.reduce(np.abs(Ri), axis=1)
        nonzero = g > 0
        prim = Ri[nonzero] // g[nonzero, None]
        orig = np.flatnonzero(nonzero)
    else:
        norms = np.linalg.norm(R, axis=1)
        # only an exactly zero row is not a ray: a direction is scale-free,
        # so an absolute floor here would delete legitimate rays of a
        # uniformly small cone (and raise "no nonzero ray" for R * 1e-20)
        nonzero = norms > 0
        prim = R[nonzero] / norms[nonzero, None]
        orig = np.flatnonzero(nonzero)

    # np.unique reports the first occurrence of each distinct row; sorting
    # those positions keeps input order (rep ascending)
    _, keep = np.unique(prim, axis=0, return_index=True)
    keep = np.sort(keep)
    return prim[keep].astype(float), orig[keep].astype(int)


def _reduce(
    R: ArrayLike,
) -> "tuple[np.ndarray | sparse.csr_matrix, np.ndarray]":
    """
    Validate ``R`` and reduce it to unique primitive representatives.

    Dispatches on sparse vs dense input; the result is float, zero rays are
    dropped, and duplicated directions collapse to their first occurrence.
    Returns ``(U, rep)`` as in ``_unique_primitive``.
    """
    if sparse.issparse(R):
        return _unique_primitive_sparse(R)
    R = np.asarray(R)
    if R.ndim != 2 or R.shape[1] == 0:
        raise ValueError("R must be a 2d array of rays")
    return _unique_primitive(R)


def _slice(U, w: np.ndarray):
    """Scale each ray onto the affine slice w . x = 1 (sparse-aware)."""
    if sparse.issparse(U):
        P = sparse.diags(1.0 / (U @ w)) @ U
        P.sort_indices()
        return P
    return U / (U @ w)[:, None]


def positive_functional(R: ArrayLike) -> np.ndarray:
    """
    Find w with R @ w >= 1 for every row of R.

    Such a w exists iff cone(R) is pointed (equivalently, contained in an
    open halfspace). The returned w defines the affine slice w . x = 1 on
    which rays are normalized to points, turning conic redundancy into
    point-hull redundancy.

    Parameters
    ----------
    R : array-like of shape (n, dim)
        Matrix whose rows are the rays.

    Returns
    -------
    w : ndarray of shape (dim,)
        A functional strictly positive on every ray.

    Raises
    ------
    ValueError
        If the cone is not pointed (the LP is infeasible).

    Notes
    -----
    Zero rows are ignored (they do not change the cone but would make
    R @ w >= 1 unsatisfiable).
    """
    if sparse.issparse(R):
        R = sparse.csr_matrix(R)
        nonzero = np.diff(R.indptr) > 0
    else:
        R = np.asarray(R, dtype=float)
        nonzero = np.any(R != 0, axis=1)
    if not nonzero.all():
        R = R[nonzero]
    n, d = R.shape
    if n == 0:
        raise ValueError("R must contain at least one nonzero ray")
    if n > _CG_THRESHOLD:
        # constraint generation: HiGHS fails outright on very tall direct
        # LPs, but a subset LP plus one full matvec of verification per
        # round stays exact and fast (an infeasible subsystem certifies
        # non-pointedness of the whole)
        rng = np.random.default_rng(0)
        # the subset and the growth step must both stay inside n: with a
        # very wide cone (d > n / _CG_BATCH) an unclamped size raises from
        # rng.choice and np.argpartition instead of just sampling everything
        batch = min(_CG_BATCH * d, n)
        S = np.sort(rng.choice(n, batch, replace=False))
        for _ in range(_CG_ROUNDS):
            w = _pf_lp(R[S])
            m = R @ w
            if m.min() > 0:
                return w / m.min()  # rescale: R @ w >= 1 everywhere
            worst = np.argpartition(m, min(batch, n - 1))[:batch]
            S = np.union1d(S, worst[m[worst] < 1])
        raise RuntimeError(
            f"constraint generation did not converge in {_CG_ROUNDS} "
            "rounds; if a positive functional is known, pass it via w="
        )
    w = _pf_lp(R)
    return w / (R @ w).min()  # rescale: R @ w >= 1 on the raw rows


def _pf_lp(R) -> np.ndarray:
    """
    The direct LP behind positive_functional: min sum(R w), R w >= 1, solved
    on unit-normalized rows (positivity is row-scale-free, and unit rows
    keep HiGHS conditioned when coefficients span many orders of magnitude).
    The returned w is strictly positive on every row of R, with slack >= 1
    on the normalized rows; callers rescale as needed.
    """
    n, d = R.shape
    if sparse.issparse(R):
        norms = np.sqrt(np.asarray(R.multiply(R).sum(axis=1)).ravel())
        Rn = sparse.diags(1.0 / norms) @ R
    else:
        Rn = R / np.linalg.norm(R, axis=1)[:, None]
    # minimizing sum(R @ w) keeps w tame; any feasible w would do
    res = linprog(
        c=np.asarray(Rn.sum(axis=0), dtype=float).ravel(),
        A_ub=-Rn.astype(float),
        b_ub=-np.ones(n),
        bounds=[(None, None)] * d,
        method="highs",
    )
    if res.status == 2:
        raise ValueError(
            "Cone is not pointed (no functional is strictly positive on all "
            "rays). Decompose into lineality space + pointed quotient first."
        )
    if not res.success:  # solver failure is not an infeasibility verdict
        raise RuntimeError(
            f"pointedness LP failed (status {res.status}: {res.message}); "
            "if a positive functional is known, pass it via w="
        )
    slack = Rn @ res.x
    if slack.min() <= 0.5:  # solver junk: never trust a bad certificate
        raise RuntimeError(
            "positive-functional certificate failed "
            f"(min slack {slack.min():.3e})"
        )
    return res.x


def _unit(v: np.ndarray) -> np.ndarray:
    """v / |v|_2 (v must be nonzero)."""
    return v / np.linalg.norm(v)


# -----------------------------------------------------------------------------
# separation oracle: one persistent HiGHS model, warm-started across calls
# -----------------------------------------------------------------------------

class _SeparationOracle:
    """
    Persistent LP  max c . p  s.t.  c . e <= 0 (e in E),  -1 <= c <= 1.

    Rows are added as E grows; only the objective changes between candidate
    tests, so HiGHS warm-starts from the previous basis. Rows can be relaxed
    (bounds widened to free) and restored, which is how the cleanup pass
    tests a confirmed ray against the others without rebuilding the model.

    The membership question is conic, hence homogeneous in every row and
    in p, so all vectors are unit-normalized on entry. This makes the
    separation value scale-free (a fixed tolerance then means the same
    thing for every candidate) and keeps the LP conditioned. Slice
    coordinates (whose magnitudes can vary by orders of magnitude across
    rays under an anisotropic w) would otherwise squash the separation
    value of small points below the solver's noise floor: on the h11=491
    Mori cone, an extremal ray with |p| ~ 1e-4 scored 8e-8 unnormalized
    (below tol) versus 7e-4 normalized, while every redundant ray scores
    below 1e-11 normalized.

    Rare HiGHS solve errors are healed by rebuilding a fresh model and
    retrying once, losing only that solve's warm start.
    """

    def __init__(self, d: int):
        self.d = d
        self._col_idx = np.arange(d, dtype=np.int32)
        self.row_of = {}
        self._rows = []
        self._free = set()
        self._fresh()

    def _fresh(self) -> None:
        self.h = highspy.Highs()
        self.h.silent()
        for name, value in _HIGHS_OPTIONS.items():
            self.h.setOptionValue(name, value)
        self.h.addVars(self.d, np.full(self.d, -1.0), np.full(self.d, 1.0))
        for nz, vals in self._rows:
            self.h.addRow(-_INF, 0.0, len(nz), nz, vals)
        for key in self._free:
            self.h.changeRowBounds(self.row_of[key], -_INF, _INF)

    def add_row(self, e: np.ndarray, key: int) -> None:
        e = _unit(np.asarray(e, dtype=float))
        nz = np.flatnonzero(e).astype(np.int32)
        vals = e[nz]
        self._rows.append((nz, vals))
        self.h.addRow(-_INF, 0.0, len(nz), nz, vals)
        self.row_of[key] = len(self.row_of)

    def relax(self, key: int) -> None:
        self._free.add(key)
        self.h.changeRowBounds(self.row_of[key], -_INF, _INF)

    def restore(self, key: int) -> None:
        self._free.discard(key)
        self.h.changeRowBounds(self.row_of[key], -_INF, 0.0)

    def separate(self, p: np.ndarray) -> "tuple[float, np.ndarray]":
        """
        Return ``(val, c)`` with val = max c . p/|p|. By Farkas' lemma,
        val ~ 0 iff p is in cone(E); val > 0 gives a separating functional
        c. The value is scale-free in p, so it can be compared against a
        fixed tolerance.

        Raises on any repeated non-optimal solver status: a solver failure
        must never be silently interpreted as a verdict.

        c = 0 is always feasible, so the true optimum is >= 0, but a
        warm-started solve can return a small NEGATIVE value: on the
        simplicial cone [[1e9,0,0],[0,1e9,0],[1e9,1e9,1]] it returns
        -7.07e-10 where a cold solve gives +7.07e-10. The sign is an
        artifact; the MAGNITUDE is the signal, and callers test |val|
        against _AMBIGUOUS_BAND for exactly that reason. Re-solving cold
        to clean up the sign was measured 3x slower overall (it discards
        the warm start on every noisy solve) and buys nothing.
        """
        p = _unit(np.asarray(p, dtype=float))
        for _ in range(2):
            self.h.changeColsCost(self.d, self._col_idx, -p)
            self.h.run()
            if self.h.getModelStatus() == highspy.HighsModelStatus.kOptimal:
                val = -self.h.getInfo().objective_function_value
                c = np.array(self.h.getSolution().col_value, dtype=float)
                return val, c
            self._fresh()
        raise RuntimeError("separation LP not optimal after model rebuild")


class _MembershipOracle:
    """
    Persistent LP  E^T lam = p,  lam >= 0: is p a non-negative combination
    of the rows of E?

    Columns are the coefficients lam (zero cost; this is pure
    feasibility); one equality row per coordinate carries E^T lam = p, and
    only the row bounds change between queries. Changing the right-hand
    side preserves DUAL feasibility of the incumbent basis, so HiGHS's
    default dual simplex resumes from it, the mirror image of
    _SeparationOracle, where only the objective moves and primal simplex
    is the right choice.

    The alternative, one fresh scipy.optimize.linprog per candidate,
    re-converts the whole coefficient matrix every call: measured 1.6x
    slower (24.8 vs 15.7 ms per LP over 400 candidates against 884 rows).

    The basis is deliberately DISCARDED between queries. Keeping it is
    another 7x faster (2.1 ms per LP), but it makes the verdict depend on
    query history: on a cone where a ray sits ~1e-15 outside cone(E), a
    cold solve proves infeasibility while a warm one returns a
    1.7e-15-residual certificate, so the same question answered yes or no
    depending on what was asked before it. Every caller here either
    removes a ray or issues an audit verdict, and neither may depend on
    iteration order. Tightening the feasibility tolerances does not fix
    it (1e-10 agrees, 1e-12 disagrees again; coincidence, not a rule).

    Verdicts are returned as reconstruction residuals, never as solver
    status codes: a caller decides membership by checking E^T lam against
    p itself, and an unsolved LP yields inf (no certificate) rather than a
    claim in either direction.
    """

    def __init__(self, E: np.ndarray, time_limit: "float | None" = None):
        self.E = np.ascontiguousarray(E, dtype=float)
        self.m, self.d = self.E.shape
        self.time_limit = time_limit
        self._row_idx = np.arange(self.d, dtype=np.int32)
        self._fresh()
        # the batch row-bounds call only appears in recent highspy (absent
        # in 1.11 and earlier); fall back to the per-row call, which costs
        # d cheap C calls per query, a few percent of one solve
        self._set_rhs = (self._set_rhs_batch
                         if hasattr(self.h, "changeRowsBounds")
                         else self._set_rhs_loop)

    def _fresh(self) -> None:
        self.h = highspy.Highs()
        self.h.silent()
        if self.time_limit is not None:
            self.h.setOptionValue("time_limit", float(self.time_limit))
        self.h.addVars(self.m, np.zeros(self.m), np.full(self.m, _INF))
        for i in range(self.d):
            col = self.E[:, i]
            nz = np.flatnonzero(col).astype(np.int32)
            self.h.addRow(0.0, 0.0, len(nz), nz, col[nz])

    def _set_rhs_batch(self, p: np.ndarray) -> None:
        self.h.changeRowsBounds(self.d, self._row_idx, p, p)

    def _set_rhs_loop(self, p: np.ndarray) -> None:
        for i in range(self.d):
            self.h.changeRowBounds(i, p[i], p[i])

    def residual(self, p: np.ndarray) -> "tuple[float, np.ndarray | None]":
        """
        Return ``(resid, lam)``: the worst-coordinate reconstruction error
        of E^T lam against p, and the coefficients. A proven-infeasible or
        unsolved LP gives ``(inf, None)``: no certificate, which is the
        sound direction for an audit.

        The question is asked about p/|p| (membership in a cone is
        scale-free), so the residual is comparable across candidates
        whatever their magnitude and a fixed tolerance means the same
        thing for each, as in _SeparationOracle. Slice coordinates span
        orders of magnitude under an anisotropic w, where an absolute
        threshold is simultaneously too strict for large p and too
        permissive for small p. lam is returned for the normalized
        question; it is used only through its support, which scaling does
        not change.
        """
        p = _unit(np.ascontiguousarray(p, dtype=float))
        for _ in range(2):
            self.h.clearSolver()  # history-independent verdicts; see above
            self._set_rhs(p)
            self.h.run()
            status = self.h.getModelStatus()
            if status == highspy.HighsModelStatus.kOptimal:
                lam = np.array(self.h.getSolution().col_value, dtype=float)
                return float(np.abs(self.E.T @ lam - p).max()), lam
            if status == highspy.HighsModelStatus.kInfeasible:
                return np.inf, None  # a verdict: p is outside cone(E)
            self._fresh()  # solver trouble: rebuild once, then give up
        return np.inf, None


# -----------------------------------------------------------------------------
# ray shooting
# -----------------------------------------------------------------------------

def _shoot(P,
           c: np.ndarray,
           cand: np.ndarray,
           rel_tol: float = 1e-9,
           all_vals: "np.ndarray | None" = None) -> "tuple[int, bool]":
    """
    Lexicographically tie-broken maximizer of P[cand] @ c.

    Returns ``(index from cand, tied)``. When ``tied`` is False the maximizer
    was unique within tolerance: since the float error of these dot products
    is orders of magnitude below the tie tolerance, the point is then the
    true unique maximizer of a linear functional and hence provably a vertex
    of conv(P), so no cleanup retest is needed. Only tie-broken results
    (``tied=True``) can be corrupted by floating point and require the
    cleanup pass.

    ``all_vals`` supplies P @ c when the caller already has it (seeding
    computes every shot in one matmul), skipping the matvec entirely.
    """
    if all_vals is not None:
        vals = all_vals[cand]
    elif len(cand) * _SUBMATRIX_SHARE < P.shape[0]:
        # few candidates left: copying their rows beats a full matvec.
        # Measured crossover at ~10% of n for 3509x491 (below that the copy
        # is up to 20x cheaper; above it the full matvec wins by ~2x)
        vals = _rows(P, cand) @ c
    else:
        vals = (P @ c)[cand]  # full matvec: no candidate-submatrix copy
    atol = rel_tol * max(1.0, float(np.abs(vals).max()))
    T = cand[vals >= vals.max() - atol]
    tied = len(T) > 1
    if tied:
        PT = _rows(P, T)
        k = 0
        d = P.shape[1]
        while len(T) > 1 and k < d:
            col = PT[:, k]
            atol_k = rel_tol * max(1.0, float(np.abs(col).max()))
            keep = col >= col.max() - atol_k
            T, PT = T[keep], PT[keep]
            k += 1
    return int(T[0]), tied


# -----------------------------------------------------------------------------
# exact rational fallback (integer rays only)
# -----------------------------------------------------------------------------

def _exact_membership(r: np.ndarray,
                      A_rows: np.ndarray,
                      lam_float: np.ndarray,
                      support_tol: float = 1e-9) -> bool:
    """
    One-sided exact certifier: try to confirm r = sum lam_i a_i with
    lam >= 0, over the rationals, using the support of a float LP solution.
    Returns True only on a rigorous success; False means inconclusive. When
    the restricted system is underdetermined, the particular solution with
    free variables at zero is tested (a valid certificate if non-negative).

    Uses python-flint when available (exact solve in C); otherwise
    fraction-free Bareiss elimination over Python ints. Per-entry Fraction
    arithmetic is avoided in both paths: it measured pathologically slow
    (minutes per candidate near dimension 500).
    """
    supp = np.flatnonzero(lam_float > support_tol)
    if len(supp) == 0:
        return bool(np.all(r == 0))
    d = len(r)
    ncol = len(supp)
    A = [[int(A_rows[j][k]) for j in supp] for k in range(d)]
    b = [int(r[k]) for k in range(d)]

    try:
        import flint
        M = flint.fmpq_mat([row + [rhs] for row, rhs in zip(A, b)])
        M = M.rref()[0]
        x = [Fraction(0)] * ncol
        # in a reduced row echelon form the pivot columns strictly increase
        # down the rows, and every row after the last pivot is zero. Scanning
        # each row from column 0 costs O(d * ncol) flint element reads;
        # resuming from the previous pivot makes the whole sweep O(ncol + d)
        start = 0
        for q in range(d):
            lead = next((c for c in range(start, ncol + 1) if M[q, c] != 0),
                        None)
            if lead is None:
                break  # a zero row: everything below it is zero too
            if lead == ncol:  # 0 = nonzero: inconsistent
                return False
            # particular solution with free variables at zero (both paths)
            v = M[q, ncol] / M[q, lead]
            x[lead] = Fraction(int(v.p), int(v.q))
            start = lead + 1
        return all(v >= 0 for v in x)
    except ImportError:
        pass

    # Bareiss fraction-free forward elimination on [A | b]
    M = [row + [rhs] for row, rhs in zip(A, b)]
    piv_cols, ri, prev = [], 0, 1
    for cj in range(ncol):
        pr = next((q for q in range(ri, d) if M[q][cj] != 0), None)
        if pr is None:
            continue
        M[ri], M[pr] = M[pr], M[ri]
        pv = M[ri][cj]
        for q in range(ri + 1, d):
            f = M[q][cj]
            M[q] = [(pv * a - f * b_) // prev
                    for a, b_ in zip(M[q], M[ri])]
        prev = pv
        piv_cols.append(cj)
        ri += 1
        if ri == d:
            break
    for q in range(ri, d):
        if M[q][ncol] != 0:
            return False
    x = [Fraction(0)] * ncol
    for k in range(len(piv_cols) - 1, -1, -1):
        cj = piv_cols[k]
        acc = Fraction(M[k][ncol])
        for c in range(cj + 1, ncol):
            acc -= M[k][c] * x[c]
        x[cj] = acc / M[k][cj]
    return all(v >= 0 for v in x)


# -----------------------------------------------------------------------------
# checkpointing and parallel sweeps
# -----------------------------------------------------------------------------

def _fingerprint(U) -> str:
    """
    Cheap input fingerprint guarding checkpoint resumes against a changed
    ray matrix (shape, edge blocks, and total mass; not cryptographic).
    """
    if sparse.issparse(U):
        blocks = [U.data[:64], U.data[-64:], U.indices[:64]]
        head = f"{U.shape}{U.nnz}"
        mass = np.abs(U.data).sum()
    else:
        b = np.ascontiguousarray(U)
        blocks = [b[:64], b[-64:]]
        head = f"{b.shape}"
        mass = np.abs(b).sum()
    h = hashlib.sha256(head.encode())
    for blk in blocks:
        h.update(np.ascontiguousarray(blk).tobytes())
    h.update(str(float(mass)).encode())
    return h.hexdigest()


def _ckpt_save(path: str, **arrays) -> None:
    """Atomic save (tmp + fsync + replace), previous kept as .bak."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    os.replace(tmp, path)


def _ckpt_load(path: str, fingerprint: str) -> "dict | None":
    """Load a checkpoint matching the fingerprint, trying the .bak rotation."""
    for cand in (path, path + ".bak"):
        try:
            with np.load(cand, allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
            if str(d.get("fingerprint")) == fingerprint:
                return d
        except Exception:
            continue
    return None


def _is_worker_process() -> bool:
    """
    True inside a spawned/forked child of this package's pool.

    Used to refuse to nest pools: spawn re-imports the caller's module in
    every child, so a caller without an ``if __name__ == "__main__":``
    guard would otherwise re-enter exhaustive() and spawn again.
    """
    import multiprocessing
    parent = getattr(multiprocessing, "parent_process", None)
    return parent is not None and parent() is not None


_POOL = {}  # per-worker-process state


def _shm_export(P) -> "tuple[list, dict]":
    """
    Copy P (dense or CSR) into shared-memory blocks. Returns the blocks
    (kept alive and unlinked by the caller) and the descriptor `_pool_init`
    needs to reconstruct P in a worker.
    """
    from multiprocessing import shared_memory
    arrays = (
        {"P": P} if not sparse.issparse(P)
        else {"data": P.data, "indices": P.indices, "indptr": P.indptr}
    )
    blocks, desc = [], {"shape": P.shape, "parts": {}}
    for name, a in arrays.items():
        shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
        np.ndarray(a.shape, a.dtype, buffer=shm.buf)[:] = a
        blocks.append(shm)
        desc["parts"][name] = (shm.name, a.shape, str(a.dtype))
    return blocks, desc


def _pool_init(desc):
    from multiprocessing import shared_memory
    parts = {}
    for name, (shm_name, shape, dtype) in desc["parts"].items():
        shm = shared_memory.SharedMemory(name=shm_name)
        _POOL[f"shm_{name}"] = shm
        parts[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    if "P" in parts:
        _POOL["P"] = parts["P"]
    else:
        _POOL["P"] = sparse.csr_matrix(
            (parts["data"], parts["indices"], parts["indptr"]),
            shape=desc["shape"])


def _pool_sweep(args):
    """
    Test one candidate chunk against a frozen E snapshot. Returns
    (redundant, failed): membership verdicts are final (E only grows), and
    failed candidates are re-resolved serially against the live E.
    """
    chunk, E_idx, tol = args
    P = _POOL["P"]
    # Keep the oracle between chunks and extend it, rather than rebuilding
    # it every time. E only grows and its rows are appended in order, so a
    # snapshot is always a prefix of a later one; a worker that already has
    # MORE rows than this snapshot may keep them, because membership in
    # cone(E) is monotone in E: a "redundant" verdict against a superset
    # of the live E is still a redundant verdict.
    #
    # Rebuilding cost the large jobs dearly: at 10M candidates with
    # |E| ~ 1218 and 2000-row chunks that is ~6 million wasted addRow calls,
    # which is most of why 8 workers returned only ~1.9x there
    oracle = _POOL.get("oracle")
    if oracle is None:
        oracle = _POOL["oracle"] = _SeparationOracle(P.shape[1])
        # One solver thread per worker. HiGHS defaults to threads=0 (auto)
        # with simplex_max_concurrency=8, so N worker processes can ask for
        # 8N threads, 64 on an 8-worker sweep, against 10 cores here. The
        # processes already supply the parallelism; letting each solver
        # oversubscribe on top of that is how a sweep ends up spending its
        # time in the scheduler instead of the simplex
        oracle.h.setOptionValue("threads", 1)
        _POOL["n_rows"] = 0
    for e in E_idx[_POOL["n_rows"]:]:
        oracle.add_row(_row(P, int(e)), int(e))
    _POOL["n_rows"] = max(_POOL["n_rows"], len(E_idx))
    redundant, failed = [], []
    for k in chunk:
        p = _row(P, int(k))
        val, _ = oracle.separate(p)
        if abs(val) <= _AMBIGUOUS_BAND:
            redundant.append(int(k))
        else:
            # includes near-tol members: resolve() re-decides those exactly
            failed.append(int(k))
    return redundant, failed


# -----------------------------------------------------------------------------
# main algorithm
# -----------------------------------------------------------------------------

def exhaustive(R: ArrayLike,
               tol: float = 1e-7,
               seed_shots: "int | str" = "auto",
               cleanup: bool = True,
               verbosity: int = 0,
               rng_seed: int = 0,
               n_workers: int = 0,
               checkpoint: "str | None" = None,
               sort_candidates: bool = False,
               known: "ArrayLike | None" = None,
               w: "ArrayLike | None" = None) -> np.ndarray:
    """
    Indices of a minimal generating subset of the rays R of a pointed cone.

    Implements Clarkson's output-sensitive algorithm: each candidate is
    tested only against the set E of confirmed extremal rays via a small
    separation LP (always feasible and bounded; extremality is never
    established through an infeasibility proof). A positive separation value
    yields a functional whose tie-broken maximizer over the remaining
    candidates is provably extremal and joins E; the candidate is then
    retested. Total LP count is at most n + |E|, each LP having at most |E|
    rows.

    Parameters
    ----------
    R : array-like or scipy.sparse matrix of shape (n, dim)
        Matrix whose rows generate the cone. Integer input enables exact
        primitive-vector deduplication and the exact rational fallback in
        the cleanup pass. Sparse input is kept sparse throughout (rows are
        densified one at a time at the LP boundary), so cones far too
        large to materialize densely are fine.
    tol : float, optional
        Threshold on the separation LP value (computed for unit-normalized
        vectors, so it is scale-free) below which a candidate counts as a
        member of cone(E). Redundant rays score at the solver's noise
        floor (~1e-12); the closest calls made are reported in
        ``core.LAST_PROFILE["closest_member"]`` and ``["closest_separated"]``.
        Defaults to 1e-7.
    seed_shots : int or "auto", optional
        Number of random functionals shot before the main loop to
        pre-populate E cheaply (each shot is a matvec argmax, no LP).
        "auto" (default) picks min(2 dim, n); 0 disables seeding.
    cleanup : bool, optional
        Retest tie-admitted rays against the other confirmed rays.
        Floating-point tie-breaking in ray shooting can rarely admit a
        redundant ray into E; cleanup restores minimality and should stay
        True unless a slightly non-minimal generating set is acceptable.
        Rays admitted as the unique maximizer of some functional are
        provably extremal and skip the retest. Defaults to True.
    verbosity : int, optional
        The verbosity level. Defaults to 0.
    rng_seed : int, optional
        Seed for the seeding functionals (results are deterministic for a
        fixed value). Defaults to 0.
    n_workers : int, optional
        Worker processes sweeping candidate chunks against a frozen snapshot
        of E ("redundant" verdicts stay valid because E only grows); the
        rare separation failures are re-resolved serially against the live
        E. 0 runs fully serially. Defaults to 0.

        Workers are spawned (not forked), so a script that passes
        ``n_workers > 0`` MUST guard its entry point with
        ``if __name__ == "__main__":``. Without it each worker re-imports
        and re-executes the caller module, producing recursive runs and
        leaked shared-memory segments. The rays are exported to shared
        memory for the duration of the call and unlinked before it returns.

        Parallel sweeping trades CPU for wall time and only pays off on
        long jobs: at the ~13 s benchmark scale it is slightly slower in
        wall time AND costs roughly 2x the CPU-seconds, so on a shared
        machine prefer serial there. See the README's Limitations.
    checkpoint : str | None, optional
        Path for periodic atomic state saves. A rerun with the same rays
        and path resumes from the last checkpoint (at most ~60s of work is
        lost on a crash). Defaults to None.
    sort_candidates : bool, optional
        Process candidates in lexicographic ray order rather than input
        order. Candidate order sets the oracle's warm-start quality.
        Locality-ordered input (e.g. generated group-by-group, the
        common case) beats the lexsort by ~18% on the h11=491 benchmark
        cone, 13.8 s against 16.3 s. Shuffling that input costs
        ~1.5-1.8x (21-25 s), which the lexsort recovers (13.7 s). Pass
        True unless the input order is known to be structured. The
        returned indices are unaffected. Defaults to False.
    known : array-like of int | None, optional
        Indices into R of rays already certified extremal, preloaded into
        the confirmed set. Almost always None: certified rays are rarely
        in hand at onset, and seeding is not a speed play (one preloaded
        ray spares about one LP, capping the saving near n_extremal/n).
        Wrong entries corrupt the result. Ignored on checkpoint resume.
        Defaults to None.
    w : array-like | None, optional
        A functional with w . r > 0 for every ray, when one is known
        structurally (e.g. from a compact description of the dual cone).
        Verified exactly with one matvec and rescaled, skipping the
        pointedness LP, which at 10M rays dwarfs it. An invalid w
        raises. Defaults to None, which solves the LP.

    Returns
    -------
    idx : ndarray of shape (n_extremal,)
        Sorted indices into R (first occurrence for duplicated directions).
        The corresponding rows are the extremal rays.

    Raises
    ------
    ValueError
        If the cone is not pointed.

    Notes
    -----
    A wall-time breakdown of the call is stored in ``core.LAST_PROFILE``.
    """
    prof = {k: 0.0 for k in (
        "preprocess", "positive_functional", "seeding",
        "main_separation_lp", "main_ray_shoot", "main_other",
        "cleanup_separation_lp", "cleanup_membership_lp", "total",
    )}
    prof["n_lp_main"] = prof["n_lp_cleanup"] = prof["n_shoot"] = 0
    prof["n_suspects"] = 0
    prof["n_exact_checks"] = prof["n_near_tol_rescued"] = 0
    prof["n_ambiguous"] = 0
    # largest value ruled a member / smallest value ruled separated: the
    # two should be far apart; a small gap means tol is doing real work
    closest = {"member": 0.0, "separated": np.inf}
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    is_sparse = sparse.issparse(R)
    U, rep = _reduce(R)
    # integer copy of the input (None if non-integral): the exact cleanup
    # fallback certifies over the original integer rays, not the float slice
    if is_sparse:
        R_in = sparse.csr_matrix(R)
        R_in.sum_duplicates()
        data_int = _as_integer(R_in.data)
        R_int = None
        if data_int is not None:
            R_int = sparse.csr_matrix(
                (data_int, R_in.indices, R_in.indptr), shape=R_in.shape)
    else:
        R_int = _as_integer(np.asarray(R))
    n, d = U.shape
    if n == 0:
        raise ValueError("R must contain at least one nonzero ray")
    if n == 1:
        return rep.copy()

    if sort_candidates:
        # similar rays adjacent => warm-started LPs; the minimal set is
        # order-independent, so only speed changes
        if is_sparse:
            order = np.lexsort(_padded_key(U).T[::-1])
        else:
            order = np.lexsort(U.T[::-1])
        U, rep = U[order], rep[order]
    prof["preprocess"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if w is not None:
        w = np.asarray(w, dtype=float)
        margins = U @ w
        if margins.min() <= 0:
            raise ValueError("supplied w is not positive on every ray "
                             f"(min margin {margins.min():.3e})")
        w = w / margins.min()  # rescale so U @ w >= 1, matching the LP
    else:
        w = positive_functional(U)
    P = _slice(U, w)
    prof["positive_functional"] = time.perf_counter() - t0

    # status: 0 unknown, 1 confirmed extremal, -1 confirmed redundant
    status = np.zeros(n, dtype=np.int8)
    E = []
    suspect = {}  # E member -> admitted only via tie-broken shots
    oracle = _SeparationOracle(d)

    def confirm(j, tied):
        status[j] = 1
        E.append(j)
        suspect[j] = tied
        oracle.add_row(_row(P, j), j)

    # --- checkpoint resume
    fp = _fingerprint(U) if checkpoint else None
    resumed = False
    if checkpoint:
        ck = _ckpt_load(checkpoint, fp)
        if ck is not None:
            status = ck["status"].astype(np.int8)
            for j, s in zip(ck["E"], ck["suspects"]):
                E.append(int(j))
                suspect[int(j)] = bool(s)
                oracle.add_row(_row(P, int(j)), int(j))
            resumed = True
            if verbosity >= 1:
                print(f"resumed: |E| = {len(E)}, "
                      f"{int((status != 0).sum())}/{n} resolved")

    def save_ckpt():
        if checkpoint:
            _ckpt_save(checkpoint, status=status,
                       E=np.array(E, dtype=np.int64),
                       suspects=np.array([suspect[j] for j in E], dtype=bool),
                       fingerprint=np.array(fp))

    # preload known rays; their certificates match the untied-shot standard
    if known is not None and not resumed:
        inv = {int(r): j for j, r in enumerate(rep)}
        for k in np.asarray(known, dtype=int):
            j = inv.get(int(k))
            if j is None:
                raise ValueError(f"known index {int(k)} is not a "
                                 "representative row (duplicate or zero?)")
            if status[j] == 0:
                confirm(j, False)
        if verbosity >= 1:
            print(f"known: {len(E)} extremal rays preloaded")

    # --- seeding: argmaxes of random functionals are vertices; no LPs needed
    if seed_shots == "auto":
        seed_shots = min(2 * d, n)
    t0 = time.perf_counter()
    if seed_shots and not resumed:
        rng = np.random.default_rng(rng_seed)
        C = rng.standard_normal((d, seed_shots))
        # every shot in one matmul: 982 separate matvecs measured 437 ms on
        # the h11=491 cone against 10 ms for the single GEMM (45x), and the
        # argmaxes are identical
        V = np.asarray(P @ C)
        all_idx = np.arange(n)
        hits = {}  # index -> tied on every shot that produced it
        for k in range(seed_shots):
            j, tied = _shoot(P, C[:, k], all_idx, all_vals=V[:, k])
            hits[j] = hits.get(j, True) and tied
        for j in sorted(hits):
            if status[j] == 0:  # may already be in E via `known`
                confirm(j, hits[j])
        if verbosity >= 1:
            print(f"seeding: {len(E)} extremal rays from {seed_shots} shots")
    prof["seeding"] = time.perf_counter() - t0

    # --- main loop
    n_lp = 0
    last_ckpt = time.time()
    t_main = time.perf_counter()
    hinted = checkpoint is not None

    def hint():
        # one-shot: suggest checkpointing while restarting is still cheap
        nonlocal hinted
        elapsed = time.perf_counter() - t_main
        if hinted or elapsed < _CKPT_HINT_DELAY:
            return
        hinted = True
        done = int((status != 0).sum())
        remaining = elapsed * (n - done) / max(1, done)
        if remaining > _CKPT_HINT_REMAINING:
            warnings.warn(
                f"roughly {remaining / 60:.0f} min of sweeping remain and "
                "no checkpoint path is set; pass checkpoint= to make this "
                "run resumable")

    def exact_member(i):
        """
        Re-decide a near-tol "redundant" verdict in exact arithmetic.

        Returns True only if r_i is provably a non-negative combination of
        the confirmed rays over the rationals. Integer input only; the
        float LP supplies the support, whose positive row scalings are the
        same in slice and original coordinates.
        """
        lam = _MembershipOracle(_rows(P, E),
                                time_limit=_CLEANUP_LP_TIME_LIMIT).residual(
                                    _row(P, i))[1]
        if lam is None:
            return False
        prof["n_exact_checks"] += 1
        return _exact_membership(_row(R_int, rep[i]),
                                 _rows(R_int, rep[E]), lam)

    def resolve(i):
        """One Clarkson resolution: redundant, or E grows until i is shot."""
        nonlocal n_lp
        p = _row(P, i)
        for _ in range(n + 1):  # safety bound; each pass grows E or resolves
            t0 = time.perf_counter()
            val, c = oracle.separate(p)
            prof["main_separation_lp"] += time.perf_counter() - t0
            n_lp += 1
            prof["n_lp_main"] += 1
            if val <= tol:
                # a verdict this close to the threshold is not safe to take
                # on faith; for integer input, settle it exactly instead
                if abs(val) > _AMBIGUOUS_BAND:
                    prof["n_ambiguous"] += 1
                if (abs(val) > _AMBIGUOUS_BAND and R_int is not None
                        and E and not exact_member(i)):
                    prof["n_near_tol_rescued"] += 1
                else:
                    status[i] = -1
                    closest["member"] = max(closest["member"], abs(val))
                    return
            else:
                closest["separated"] = min(closest["separated"], val)
            t0 = time.perf_counter()
            j, tied = _shoot(P, c, np.flatnonzero(status == 0))
            prof["main_ray_shoot"] += time.perf_counter() - t0
            prof["n_shoot"] += 1
            confirm(j, tied)
            if j == i:
                return
        raise RuntimeError(f"failed to resolve candidate {i}")

    if n_workers and _is_worker_process():
        # A caller that passes n_workers>0 without an
        # `if __name__ == "__main__":` guard has every spawned child
        # re-import its module and call this again; without this check that
        # recurses until the process table gives out. Degrade instead
        warnings.warn(
            "n_workers > 0 was requested inside a worker process, which "
            "means the calling module is missing an "
            '`if __name__ == "__main__":` guard. Sweeping serially in this '
            "process instead. Add the guard to enable parallel sweeps."
        )
        n_workers = 0

    if n_workers:
        # workers pull chunks against frozen E snapshots (valid: E only
        # grows); the rare separation failures are re-resolved serially
        from multiprocessing import get_context
        blocks, desc = _shm_export(P)
        pool = get_context("spawn").Pool(
            n_workers, initializer=_pool_init, initargs=(desc,))
        try:
            chunk = max(64, min(2000, n // (8 * n_workers) or 64))

            def jobs():
                # runs in the pool's feeder thread; stale reads of status/E
                # only cause harmless duplicate work
                k = 0
                while k < n:
                    todo = []
                    while k < n and len(todo) < chunk:
                        if status[k] == 0:
                            todo.append(k)
                        k += 1
                    if todo:
                        yield (todo, np.array(E, dtype=np.int64), tol)

            for red, failed in pool.imap_unordered(_pool_sweep, jobs()):
                for k in red:
                    if status[k] == 0:
                        status[k] = -1
                for k in failed:
                    if status[k] == 0:
                        resolve(k)
                hint()
                if verbosity >= 1:
                    done = int((status != 0).sum())
                    print(f"  {done}/{n} candidates, |E| = {len(E)}")
                if checkpoint and time.time() - last_ckpt > 60:
                    save_ckpt()
                    last_ckpt = time.time()
        finally:
            pool.close()
            pool.join()
            for shm in blocks:
                shm.close()
                shm.unlink()

    for i in range(n):
        if status[i] != 0:
            continue
        resolve(i)
        hint()
        if verbosity >= 1 and (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n} candidates, |E| = {len(E)}, LPs = {n_lp}")
        if checkpoint and time.time() - last_ckpt > 60:
            save_ckpt()
            last_ckpt = time.time()
    prof["main_other"] = (
        time.perf_counter() - t_main
        - prof["main_separation_lp"] - prof["main_ray_shoot"]
    )

    # --- cleanup: restore minimality lost to floating-point tie-breaking.
    # Only rays admitted through tie-broken shots can be impostors; rays that
    # were the unique maximizer of some functional are provably vertices
    if cleanup:
        suspects = [e for e in sorted(E) if suspect.get(e, True)]
        prof["n_suspects"] = len(suspects)
        if verbosity >= 1:
            print(f"cleanup: {len(suspects)}/{len(E)} rays were tie-admitted")
        for k_sus, e in enumerate(suspects):
            if verbosity >= 1 and k_sus % 50 == 0:
                print(f"  cleanup: {k_sus}/{len(suspects)} suspects")
            others = [x for x in E if x != e]
            if not others:
                continue
            pe = _row(P, e)
            oracle.relax(e)
            t0 = time.perf_counter()
            val, _c = oracle.separate(pe)
            prof["cleanup_separation_lp"] += time.perf_counter() - t0
            n_lp += 1
            prof["n_lp_cleanup"] += 1
            if val > tol:
                oracle.restore(e)
                continue
            # not separable: demand a positive certificate of redundancy.
            # time-limited: single HiGHS solves have measured hour-scale
            # pathologies on degenerate geometries, and an uncertified
            # suspect is kept (sound), so a bounded attempt suffices
            t0 = time.perf_counter()
            oracle_m = _MembershipOracle(
                _rows(P, others), time_limit=_CLEANUP_LP_TIME_LIMIT)
            resid, lam = oracle_m.residual(pe)
            prof["cleanup_membership_lp"] += time.perf_counter() - t0
            certified = resid < _MEMBERSHIP_RESID_TOL
            if not certified and R_int is not None and lam is not None:
                certified = _exact_membership(
                    _row(R_int, rep[e]), _rows(R_int, rep[others]), lam
                )
            if certified:
                E.remove(e)
                status[e] = -1
                if verbosity >= 1:
                    print(f"  cleanup: removed redundant ray {e}")
            else:
                oracle.restore(e)
                warnings.warn(
                    f"ray {rep[e]} is numerically borderline (separation "
                    f"margin {val:.2e}, no membership certificate); keeping "
                    "it; the result generates the cone but may not be "
                    "minimal. Consider verify()."
                )

    # An ambiguous verdict means the tolerance, not the geometry, made the
    # call. For integer rays those were already settled in exact arithmetic
    # above, so there is nothing to warn about; for float rays no exact path
    # exists and this warning is the only signal the caller gets
    if closest["member"] > _AMBIGUOUS_BAND and R_int is None:
        warnings.warn(
            f"{prof['n_ambiguous']} redundancy verdict(s) landed between the "
            f"solver noise floor and tol (worst {closest['member']:.2e} vs "
            f"tol {tol:.1e}), so the tolerance decided them rather than the "
            "geometry, and extremal rays may be missing. Float rays cannot "
            "be re-decided exactly: pass integer rays to enable that, or "
            "re-run with a smaller tol, and audit with verify()."
        )

    save_ckpt()
    prof["closest_member"] = closest["member"]
    prof["closest_separated"] = closest["separated"]
    prof["total"] = time.perf_counter() - t_total
    LAST_PROFILE.clear()
    LAST_PROFILE.update(prof)
    if verbosity >= 1:
        print(f"done: {len(E)} extremal rays, {n_lp} LPs")
    return np.sort(rep[sorted(E)])
