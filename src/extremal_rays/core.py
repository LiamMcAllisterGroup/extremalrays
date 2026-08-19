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
    R = np.asarray(R)
    n, d = R.shape
    # minimizing sum(R @ w) keeps w tame; any feasible w would do
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

def _shoot(P: np.ndarray,
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
    vals = P[cand] @ c
    atol = rel_tol * max(1.0, float(np.abs(vals).max()))
    T = cand[vals >= vals.max() - atol]
    tied = len(T) > 1
    k = 0
    d = P.shape[1]
    while len(T) > 1 and k < d:
        col = P[T, k]
        atol_k = rel_tol * max(1.0, float(np.abs(col).max()))
        T = T[col >= col.max() - atol_k]
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

def _fingerprint(U: np.ndarray) -> str:
    """
    Cheap input fingerprint guarding checkpoint resumes against a changed
    ray matrix (shape, edge blocks, and total mass -- not cryptographic).
    """
    h = hashlib.sha256()
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


def _pool_init(shm_name, shape, dtype):
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=shm_name)
    _POOL["shm"] = shm
    _POOL["P"] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)


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
        oracle.add_row(P[int(e)], int(e))
    redundant, failed = [], []
    for k in chunk:
        p = P[int(k)]
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
                  sort_candidates: bool = True) -> np.ndarray:
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
    R : array-like of shape (n, dim)
        Matrix whose rows generate the cone. Integer input enables exact
        primitive-vector deduplication and the exact rational fallback in
        the cleanup pass.
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
    R_in = np.asarray(R)
    if R_in.ndim != 2 or R_in.shape[0] == 0:
        raise ValueError("R must be a non-empty 2d array of rays")
    U, rep = _unique_primitive(R_in)
    n, d = U.shape
    if n == 1:
        return rep.copy()

    R_int = _as_integer(R_in)

    if sort_candidates:
        # similar rays adjacent => warm-started LPs; the minimal set is
        # order-independent, so only speed changes
        order = np.lexsort(U.T[::-1])
        U, rep = U[order], rep[order]
    prof["preprocess"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    w = positive_functional(U)
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
        oracle.add_row(P[j], j)

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
                oracle.add_row(P[int(j)], int(j))
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
        for _ in range(n + 1):  # safety bound; each pass grows E or resolves
            t0 = time.perf_counter()
            val, c = oracle.separate(P[i])
            prof["main_separation_lp"] += time.perf_counter() - t0
            n_lp += 1
            prof["n_lp_main"] += 1
            if val <= tol * max(1.0, float(np.abs(P[i]).max())):
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
        from multiprocessing import get_context, shared_memory
        shm = shared_memory.SharedMemory(create=True, size=P.nbytes)
        np.ndarray(P.shape, P.dtype, buffer=shm.buf)[:] = P
        pool = get_context("spawn").Pool(
            n_workers, initializer=_pool_init,
            initargs=(shm.name, P.shape, str(P.dtype)))
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
            oracle.relax(e)
            t0 = time.perf_counter()
            val, _c = oracle.separate(P[e])
            prof["cleanup_separation_lp"] += time.perf_counter() - t0
            n_lp += 1
            prof["n_lp_cleanup"] += 1
            scale = max(1.0, float(np.abs(P[e]).max()))
            if val > tol * scale:
                oracle.restore(e)
                continue
            # not separable: demand a positive certificate of redundancy
            t0 = time.perf_counter()
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
            prof["cleanup_membership_lp"] += time.perf_counter() - t0
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

    save_ckpt()
    prof["total"] = time.perf_counter() - t_total
    LAST_PROFILE.clear()
    LAST_PROFILE.update(prof)
    if verbose:
        print(f"done: {len(E)} extremal rays, {n_lp} LPs")
    return np.sort(rep[sorted(E)])
