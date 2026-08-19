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

# wall-time breakdown of the most recent extremal_rays() call, in seconds
LAST_PROFILE: dict = {}


# -----------------------------------------------------------------------------
# preprocessing
# -----------------------------------------------------------------------------

def _as_integer(R: np.ndarray) -> "np.ndarray | None":
    """
    Return an integer copy of ``R``, or None if ``R`` is genuinely
    non-integral.
    """
    if np.issubdtype(R.dtype, np.integer):
        return R.astype(np.int64)
    Rr = np.round(R)
    if np.allclose(R, Rr, rtol=0, atol=1e-9):
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

    Each row becomes its column indices (padded with ``dim``, then packed
    several per word) followed by one word per value slot, holding the
    float's bit pattern with the sign bit flipped (and the rest complemented
    for negatives), so unsigned bitwise order equals numeric order. Distinct
    rows get distinct keys (indices are sorted, the value map is injective),
    so the keys support exact deduplication; lexicographic order on them is
    lexicographic (support, values) order for ``sort_candidates``. Few
    int64-sized columns keep the sorts on numpy's fast scalar paths instead
    of np.unique(axis=0)'s per-row byte comparisons.
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
    """
    if not sparse.issparse(R):
        R = np.asarray(R)
    n, d = R.shape
    # minimizing sum(R @ w) keeps w tame; any feasible w would do
    res = linprog(
        c=np.asarray(R.sum(axis=0), dtype=float).ravel(),
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
            "positive-functional certificate failed "
            f"(min slack {slack.min():.3e})"
        )
    return w


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
        self.h.addVars(self.d, np.full(self.d, -1.0), np.full(self.d, 1.0))
        for nz, vals in self._rows:
            self.h.addRow(-_INF, 0.0, len(nz), nz, vals)
        for key in self._free:
            self.h.changeRowBounds(self.row_of[key], -_INF, _INF)

    def add_row(self, e: np.ndarray, key: int) -> None:
        nz = np.flatnonzero(e).astype(np.int32)
        vals = e[nz].astype(float)
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
        Return ``(val, c)`` with val = max c . p. By Farkas' lemma, val ~ 0
        iff p is in cone(E); val > 0 gives a separating functional c.

        Raises on any repeated non-optimal solver status: a solver failure
        must never be silently interpreted as a verdict.
        """
        for _ in range(2):
            self.h.changeColsCost(self.d, self._col_idx, (-p).astype(float))
            self.h.run()
            if self.h.getModelStatus() == highspy.HighsModelStatus.kOptimal:
                val = -self.h.getInfo().objective_function_value
                c = np.array(self.h.getSolution().col_value, dtype=float)
                return val, c
            self._fresh()
        raise RuntimeError("separation LP not optimal after model rebuild")


# -----------------------------------------------------------------------------
# ray shooting
# -----------------------------------------------------------------------------

def _shoot(P,
           c: np.ndarray,
           cand: np.ndarray,
           rel_tol: float = 1e-9) -> "tuple[int, bool]":
    """
    Lexicographically tie-broken maximizer of P[cand] @ c.

    Returns ``(index from cand, tied)``. When ``tied`` is False the maximizer
    was unique within tolerance: since the float error of these dot products
    is orders of magnitude below the tie tolerance, the point is then the
    true unique maximizer of a linear functional and hence provably a vertex
    of conv(P) -- no cleanup retest needed. Only tie-broken results
    (``tied=True``) can be corrupted by floating point and require the
    cleanup pass.
    """
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
    Returns True only on a rigorous success; False means inconclusive.
    """
    supp = np.flatnonzero(lam_float > support_tol)
    if len(supp) == 0:
        return bool(np.all(r == 0))
    d = len(r)
    ncol = len(supp)
    # Gaussian elimination over Q on [A_supp | r]
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


# -----------------------------------------------------------------------------
# checkpointing and parallel sweeps
# -----------------------------------------------------------------------------

def _fingerprint(U) -> str:
    """
    Cheap input fingerprint guarding checkpoint resumes against a changed
    ray matrix (shape, edge blocks, and total mass -- not cryptographic).
    """
    h = hashlib.sha256()
    if sparse.issparse(U):
        h.update(str(U.shape).encode())
        h.update(str(U.nnz).encode())
        h.update(U.data[:64].tobytes())
        h.update(U.data[-64:].tobytes())
        h.update(U.indices[:64].tobytes())
        h.update(str(float(np.abs(U.data).sum())).encode())
        return h.hexdigest()
    b = np.ascontiguousarray(U)
    h.update(str(b.shape).encode())
    h.update(b[:64].tobytes())
    h.update(b[-64:].tobytes())
    h.update(str(float(np.abs(b).sum())).encode())
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
    oracle = _SeparationOracle(P.shape[1])
    for e in E_idx:
        oracle.add_row(_row(P, int(e)), int(e))
    redundant, failed = [], []
    for k in chunk:
        p = _row(P, int(k))
        val, _ = oracle.separate(p)
        if val <= tol * max(1.0, float(np.abs(p).max())):
            redundant.append(int(k))
        else:
            failed.append(int(k))
    return redundant, failed


# -----------------------------------------------------------------------------
# main algorithm
# -----------------------------------------------------------------------------

def extremal_rays(R: ArrayLike,
                  tol: float = 1e-7,
                  seed_shots: "int | str" = "auto",
                  cleanup: bool = True,
                  verbose: bool = False,
                  rng_seed: int = 0,
                  n_workers: int = 0,
                  checkpoint: "str | None" = None,
                  sort_candidates: bool = True,
                  known: "ArrayLike | None" = None) -> np.ndarray:
    """
    Indices of a minimal generating subset of the rays R of a pointed cone.

    Implements Clarkson's output-sensitive algorithm: each candidate is
    tested only against the set E of confirmed extremal rays via a small
    separation LP (always feasible and bounded -- extremality is never
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
        Relative threshold on the separation LP value for deciding
        membership vs. separation. Defaults to 1e-7.
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
    verbose : bool, optional
        Whether to print progress. Defaults to False.
    rng_seed : int, optional
        Seed for the seeding functionals (results are deterministic for a
        fixed value). Defaults to 0.
    n_workers : int, optional
        Worker processes sweeping candidate chunks against a frozen snapshot
        of E ("redundant" verdicts stay valid because E only grows); the
        rare separation failures are re-resolved serially against the live
        E. 0 runs fully serially. Defaults to 0.
    checkpoint : str | None, optional
        Path for periodic atomic state saves. A rerun with the same rays
        and path resumes from the last checkpoint (at most ~60s of work is
        lost on a crash). Defaults to None.
    sort_candidates : bool, optional
        Process candidates in lexicographic ray order rather than input
        order, keeping similar rays adjacent so the oracle's warm starts
        pay off (measured up to ~10x on shuffled input). The returned
        indices are unaffected. Defaults to True.
    known : array-like of int | None, optional
        Indices into R of rays already certified extremal, preloaded into
        the confirmed set. Almost always None: certified rays are rarely
        in hand at onset (see the note above). Wrong entries corrupt the
        result -- pass only certificate-backed indices. Ignored on
        checkpoint resume. Defaults to None.

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

    Certified extremal rays are almost never in hand at the onset of a
    computation, so `known` should almost always be left None. It exists
    for the rare resume-like cases where they exist anyway (a prior run on
    the same cone, an interrupted job). It is NOT a speed play: one
    preloaded ray spares the sweep about one discovery (~1 extra
    separation LP + ray shoot), capping the saving near n_extremal/n of
    the total, so sampling rays first just to seed costs more than it
    saves.

    Candidate ORDER matters for speed: the oracle warm-starts between
    consecutive LPs, so orderings that keep similar rays adjacent (e.g.
    sorted rows) run up to ~10x faster than shuffled input.
    """
    prof = {k: 0.0 for k in (
        "preprocess", "positive_functional", "seeding",
        "main_separation_lp", "main_ray_shoot", "main_other",
        "cleanup_separation_lp", "cleanup_membership_lp", "total",
    )}
    prof["n_lp_main"] = prof["n_lp_cleanup"] = prof["n_shoot"] = 0
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    is_sparse = sparse.issparse(R)
    if is_sparse:
        R_in = sparse.csr_matrix(R, copy=True)
        R_in.sum_duplicates()
        R_in.sort_indices()
        U, rep = _unique_primitive_sparse(R_in)
        data_int = _as_integer(R_in.data)
        R_int = None
        if data_int is not None:
            R_int = sparse.csr_matrix(
                (data_int, R_in.indices, R_in.indptr), shape=R_in.shape)
    else:
        R_in = np.asarray(R)
        if R_in.ndim != 2:
            raise ValueError("R must be a non-empty 2d array of rays")
        U, rep = _unique_primitive(R_in)
        R_int = _as_integer(R_in)
    if R_in.shape[0] == 0:
        raise ValueError("R must be a non-empty 2d array of rays")
    n, d = U.shape
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
    w = positive_functional(U)
    if is_sparse:
        P = sparse.diags(1.0 / (U @ w)) @ U
        P.sort_indices()
    else:
        P = U / (U @ w)[:, None]
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
            if verbose:
                print(f"resumed: |E| = {len(E)}, "
                      f"{int((status != 0).sum())}/{n} resolved")

    def save_ckpt():
        if checkpoint:
            _ckpt_save(checkpoint, status=status,
                       E=np.array(E, dtype=np.int64),
                       suspects=np.array([suspect[j] for j in E], dtype=bool),
                       fingerprint=np.array(fp))

    # --- known extremal rays (e.g. from sample_extremal_rays) join E
    # directly; their unique-tight certificates meet the same standard that
    # exempts untied shots from cleanup, so they are not marked suspect
    if known is not None and not resumed:
        inv = {int(r): j for j, r in enumerate(rep)}
        for k in np.asarray(known, dtype=int):
            j = inv.get(int(k))
            if j is None:
                raise ValueError(f"known index {int(k)} is not a "
                                 "representative row (duplicate or zero?)")
            if status[j] == 0:
                confirm(j, False)
        if verbose:
            print(f"known: {len(E)} extremal rays preloaded")

    # --- seeding: argmaxes of random functionals are vertices; no LPs needed
    if seed_shots == "auto":
        seed_shots = min(2 * d, n)
    t0 = time.perf_counter()
    if seed_shots and not resumed:
        rng = np.random.default_rng(rng_seed)
        C = rng.standard_normal((d, seed_shots))
        all_idx = np.arange(n)
        hits = {}  # index -> tied on every shot that produced it
        for k in range(seed_shots):
            j, tied = _shoot(P, C[:, k], all_idx)
            hits[j] = hits.get(j, True) and tied
        for j in sorted(hits):
            if status[j] == 0:  # may already be in E via `known`
                confirm(j, hits[j])
        if verbose:
            print(f"seeding: {len(E)} extremal rays from {seed_shots} shots")
    prof["seeding"] = time.perf_counter() - t0

    # --- main loop
    n_lp = 0
    last_ckpt = time.time()
    t_main = time.perf_counter()

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
            if val <= tol * max(1.0, float(np.abs(p).max())):
                status[i] = -1
                return
            t0 = time.perf_counter()
            j, tied = _shoot(P, c, np.flatnonzero(status == 0))
            prof["main_ray_shoot"] += time.perf_counter() - t0
            prof["n_shoot"] += 1
            confirm(j, tied)
            if j == i:
                return
        raise RuntimeError(f"failed to resolve candidate {i}")

    if n_workers:
        # streaming parallel sweeps: workers pull candidate chunks as they
        # finish (no round barrier), each against the E snapshot current at
        # dispatch; the rare separation failures are resolved serially here
        # while the workers keep going. Verdicts against a stale snapshot
        # stay valid because E only grows.
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
                if verbose:
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
        if verbose and (i + 1) % 500 == 0:
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
    # were the unique maximizer of some functional are provably vertices.
    if cleanup:
        suspects = [e for e in sorted(E) if suspect.get(e, True)]
        prof["n_suspects"] = len(suspects)
        if verbose:
            print(f"cleanup: {len(suspects)}/{len(E)} rays were tie-admitted")
        for e in suspects:
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
            scale = max(1.0, float(np.abs(pe).max()))
            if val > tol * scale:
                oracle.restore(e)
                continue
            # not separable: demand a positive certificate of redundancy
            t0 = time.perf_counter()
            Po = _rows(P, others)
            res = linprog(
                c=np.zeros(len(others)),
                A_eq=Po.T,
                b_eq=pe,
                bounds=[(0, None)],
                method="highs",
            )
            resid = (
                float(np.abs(Po.T @ res.x - pe).max())
                if res.success else np.inf
            )
            prof["cleanup_membership_lp"] += time.perf_counter() - t0
            certified = res.success and resid < 1e-6
            if not certified and R_int is not None and res.success:
                certified = _exact_membership(
                    _row(R_int, rep[e]), _rows(R_int, rep[others]), res.x
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

    save_ckpt()
    prof["total"] = time.perf_counter() - t_total
    LAST_PROFILE.clear()
    LAST_PROFILE.update(prof)
    if verbose:
        print(f"done: {len(E)} extremal rays, {n_lp} LPs")
    return np.sort(rep[sorted(E)])
