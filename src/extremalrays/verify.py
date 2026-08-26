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
# Description:  This module contains an independent certificate-based audit of
#               an extremal-ray computation. Every claim is re-derived from
#               explicit, inspectable witnesses rather than solver status
#               codes: a non-negative combination for every discarded ray, a
#               separating functional for every kept ray.
# -----------------------------------------------------------------------------
"""
An independent certificate-based audit of an extremal-ray computation.

:func:`verify` re-derives every claim from explicit, inspectable witnesses
rather than solver status codes: a non-negative combination for every
discarded ray, and a separating functional for every kept one. Certificates
are checked, not trusted: reconstruction residuals are recomputed, and for
integer rays a borderline float certificate is settled in exact rational
arithmetic.

The independence is real but bounded, and the docstring of :func:`verify`
says where it stops: the certificates are built independently of
:func:`extremalrays.exhaustive`, but both share this package's
preprocessing, so an error inside that shared step is invisible here.
"""
from __future__ import annotations

# external imports
import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse

# local imports
from .core import (_as_integer, _exact_membership, _MembershipOracle,
                   _reduce, _row, _rows, _shm_export, _slice,
                   _SeparationOracle, positive_functional)


_POOL = {}  # per-worker state


def _audit_init(desc, E, time_limit):
    """Rebuild the shared slice matrix and both oracles inside a worker."""
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=desc["name"])
    _POOL["shm"] = shm
    _POOL["P"] = np.ndarray(desc["shape"], dtype=desc["dtype"], buffer=shm.buf)
    _POOL["E"] = E
    _POOL["membership"] = _MembershipOracle(E, time_limit=time_limit)
    oracle = _SeparationOracle(E.shape[1])
    for k, row in enumerate(E):
        oracle.add_row(row, k)
    _POOL["separation"] = oracle


def _audit_chunk(chunk):
    """
    Membership residual and separation margin for a block of candidates.

    Every candidate is an independent question once the result set is
    fixed, so this parallelises cleanly. Only three floats come back per
    candidate: the coefficients that an exact re-check would need are far
    larger than the answer, and are recomputed in the parent for the few
    rays that turn out to need them.
    """
    P, membership, separation = (_POOL["P"], _POOL["membership"],
                                 _POOL["separation"])
    out = []
    for k in chunk:
        resid, _lam = membership.residual(P[k])
        margin = abs(separation.separate(P[k])[0])
        out.append((int(k), float(resid), float(margin)))
    return out


def _audit_parallel(P, E, candidates, n_workers, time_limit):
    """Run the per-candidate audit across worker processes."""
    from multiprocessing import get_context
    blocks, desc = _shm_export(P)
    desc = {"name": desc["parts"]["P"][0], "shape": P.shape,
            "dtype": str(P.dtype)}
    def split(items):
        n = max(16, len(items) // (4 * n_workers) or 16)
        return [items[i:i + n] for i in range(0, len(items), n)]

    pool = get_context("spawn").Pool(
        n_workers, initializer=_audit_init, initargs=(desc, E, time_limit))
    try:
        membership = []
        for part in pool.imap_unordered(_audit_chunk, split(candidates)):
            membership.extend(part)
        separation = []
        for part in pool.imap_unordered(_audit_sep_chunk,
                                        split(list(range(len(E))))):
            separation.extend(part)
    finally:
        pool.close()
        pool.join()
        for shm in blocks:
            shm.close()
            shm.unlink()
    return sorted(membership), dict(separation)


def _audit_sep_chunk(chunk):
    """
    Separation margin for a block of RESULT rays, each against the others.

    Independent per ray once the result is fixed: relaxing row i and
    restoring it leaves the model as it was, so a worker can walk its own
    block in any order. This pass is 42% of the audit's work, so leaving it
    serial capped the whole thing at ~2.4x however many workers were used.
    """
    E, oracle = _POOL["E"], _POOL["separation"]
    out = []
    for i in chunk:
        oracle.relax(i)
        val = oracle.separate(E[i])[0]
        oracle.restore(i)
        out.append((int(i), float(val)))
    return out


def _margin(oracle: "_SeparationOracle", p: np.ndarray,
            band: float = 1e-7) -> bool:
    """
    True when p sits close enough to the boundary of cone(E) that a float
    membership certificate cannot be trusted on its own. Genuinely interior
    rays separate at the solver's noise floor (~1e-13); a ray a hair
    outside scores orders of magnitude above it but still under any usable
    tolerance, which is exactly the case worth spending exact arithmetic on.
    """
    val = abs(oracle.separate(p)[0])  # the sign is a warm-start artifact
    return 1e-13 < val < band


def verify(R: ArrayLike,
           ext_indices: ArrayLike,
           tol: float = 1e-6,
           sep_tol: float = 1e-7,
           time_limit: "float | None" = None,
           w: "ArrayLike | None" = None,
           n_workers: int = 0,
           verbosity: int = 0) -> "tuple[bool, dict]":
    """
    Check that R[ext_indices] is a minimal generating set for cone(R).

    Two families of certificates are produced:

    - each ray NOT in ``ext_indices`` gets a non-negative combination of the
      result rays reproducing it (membership certificate; the reconstruction
      residual is checked, not trusted),
    - each ray IN ``ext_indices`` gets a functional that is non-positive on
      all other result rays and strictly positive on it (separation
      certificate, proving both extremality within the cone and minimality
      of the result).

    This roughly doubles the cost of the original computation; it is
    intended as an audit, not a routine step.

    Independence has a limit worth stating plainly: the certificates are
    built independently of ``exhaustive`` (different formulation, opposite
    LP direction), but both share this package's preprocessing --
    ``_reduce``'s primitive deduplication and the slice scaling. An error
    inside that shared step would be invisible here. What this audits is
    the extremal-ray decision, not the deduplication.

    Parameters
    ----------
    R : array-like or scipy.sparse matrix of shape (n, dim)
        Matrix whose rows generate the cone.
    ext_indices : array-like of int
        Indices into R of the claimed extremal rays (first occurrences, as
        exhaustive() returns them).
    tol : float, optional
        Maximum allowed reconstruction residual for membership
        certificates, measured on the unit-normalized question so that it
        means the same thing for every candidate. Defaults to 1e-6.
    sep_tol : float, optional
        Separation margin below which a claimed extremal ray is not
        accepted on its margin alone. Such a ray is re-decided by the
        opposite question, is it in the cone of the other result rays,
        because a nearly-redundant ray is still a ray: an infeasible
        membership LP proves extremality however narrow the margin.
        Defaults to 1e-7.
    w : array-like | None, optional
        A functional positive on every ray, used for the slice exactly as
        in ``exhaustive``. Supply the same ``w`` when re-solving the
        pointedness LP is impractical; on the 10M-ray Mori-cone cap that
        LP is what makes the audit hard to start at all. Defaults to None,
        which solves the LP.
    n_workers : int, optional
        Worker processes for the per-candidate pass. Once the claimed
        result is fixed, every candidate is an independent question, so
        this parallelises cleanly, unlike ``exhaustive``, where the
        confirmed set grows as the sweep runs. Requires dense input and,
        like ``exhaustive``, an ``if __name__ == "__main__":`` guard in the
        caller. 0 runs serially. Defaults to 0.
    time_limit : float | None, optional
        Seconds allowed per membership LP. Single HiGHS solves have
        measured hour-scale pathologies on degenerate geometries; a ray
        whose LP times out is reported as a failure (no certificate), so
        a bound keeps the audit finite without making it unsound.
        Defaults to None (no limit).
    verbosity : int, optional
        The verbosity level; >= 1 prints the worst certificate margins and
        any failures. Defaults to 0.

    Returns
    -------
    ok : bool
        True if every certificate checks out.
    report : dict
        The worst membership residual, the worst separation margin, and a
        list of failure descriptions (empty when ok).

    Raises
    ------
    ValueError
        If an index in ``ext_indices`` is not a representative row (a zero
        ray, or a later duplicate of an earlier direction).
    """
    U, rep = _reduce(R)
    # integer rays let a borderline float certificate be settled exactly,
    # which is the only way to separate "in the cone" from "1e-10 outside"
    R_int = None if sparse.issparse(R) else _as_integer(np.asarray(R))
    rep_pos = {int(orig): k for k, orig in enumerate(rep)}
    ext = set()
    for i in np.asarray(ext_indices, dtype=int).ravel():
        k = rep_pos.get(int(i))
        if k is None:
            raise ValueError(f"index {int(i)} is not a representative row "
                             "(duplicate or zero ray?)")
        ext.add(k)
    ext = sorted(ext)

    if w is not None:
        w = np.asarray(w, dtype=float)
        margins = U @ w
        if margins.min() <= 0:
            raise ValueError("supplied w is not positive on every ray "
                             f"(min margin {margins.min():.3e})")
        w = w / margins.min()
    else:
        w = positive_functional(U)
    P = _slice(U, w)
    E = _rows(P, ext)
    ext_set = set(ext)

    failures = []
    worst_resid = 0.0
    if not ext:  # a nonzero cone has at least one extremal ray
        failures.append("no rays claimed: cone(R) is nonzero")
        worst_resid = np.inf
    # one persistent model for every membership question: only the
    # right-hand side changes per candidate, so it warm-starts
    membership = _MembershipOracle(E, time_limit=time_limit) if ext else None
    # a separation oracle over the whole result, used to spot rays sitting
    # on the boundary where a float certificate is not decisive
    oracle_s = _SeparationOracle(U.shape[1])
    for k_e, e in enumerate(ext):
        oracle_s.add_row(E[k_e], e)
    candidates = [k for k in range(U.shape[0]) if k not in ext_set] if ext \
        else []
    want_margin = R_int is not None       # only integer rays can escalate
    margins = None
    if n_workers and candidates and not sparse.issparse(P):
        measured, margins = _audit_parallel(P, E, candidates, n_workers,
                                            time_limit)
    else:
        measured = []
        for k in candidates:
            pk = _row(P, k)
            resid, _lam = membership.residual(pk)
            margin = abs(oracle_s.separate(pk)[0]) if want_margin else 0.0
            measured.append((k, resid, margin))

    for k, resid, margin in measured:
        worst_resid = max(worst_resid, resid)
        if not resid < tol:
            failures.append(f"ray {rep[k]}: no membership certificate "
                            f"(residual {resid:.2e})")
            continue
        # A float certificate cannot tell "inside the cone" from "just
        # outside it"; when the ray is only just inside, settle it exactly.
        # The coefficients are recomputed here rather than shipped back
        # from a worker: they are large and only these few rays need them
        if want_margin and 1e-13 < margin < 1e-7:
            lam = membership.residual(_row(P, k))[1]
            if lam is not None and not _exact_membership(
                    _row(R_int, rep[k]), _rows(R_int, rep[ext]), lam):
                failures.append(
                    f"ray {rep[k]}: float certificate (residual "
                    f"{resid:.2e}) is not confirmed in exact arithmetic "
                    "-- the ray is outside the cone and is missing from "
                    "the result")

    oracle = oracle_s
    worst_margin = np.inf
    for k, e in enumerate(ext):
        if margins is not None:
            val = margins[k]          # computed by the workers above
        else:
            oracle.relax(e)
            val, _c = oracle.separate(E[k])
            oracle.restore(e)
        worst_margin = min(worst_margin, val)
        if val > sep_tol:  # scale-free: the oracle normalizes its vectors
            continue
        # Too narrow to call on the margin alone, so ask the opposite
        # question. If the ray is not in the cone of the others it is
        # extremal however small its margin; rejecting it on the margin
        # was itself a bug, which failed correct answers as "redundant"
        others = np.delete(E, k, axis=0)
        resid, lam = _MembershipOracle(others,
                                       time_limit=time_limit).residual(E[k])
        if resid < tol and R_int is not None and lam is not None:
            # exact arithmetic decides it: a ray that is merely CLOSE to
            # the cone of the others is still extremal
            other_rows = [rep[x] for j, x in enumerate(ext) if j != k]
            if not _exact_membership(_row(R_int, rep[e]),
                                     _rows(R_int, other_rows), lam):
                if verbosity >= 1:
                    print(f"ray {rep[e]}: margin {val:.2e} below sep_tol, "
                          "but exactly outside the cone: extremal")
                continue
        if resid < tol:
            failures.append(
                f"ray {rep[e]}: lies in the cone of the other result rays "
                f"(residual {resid:.2e}): redundant"
            )
        elif verbosity >= 1:
            print(f"ray {rep[e]}: margin {val:.2e} below sep_tol but "
                  "provably outside the cone of the others: extremal")

    report = {
        "worst_membership_residual": worst_resid,
        "worst_separation_margin": worst_margin,
        "failures": failures,
    }
    if verbosity >= 1:
        print(f"worst membership residual: {worst_resid:.2e}")
        print(f"worst separation margin:   {worst_margin:.2e}")
        for f in failures:
            print("FAIL:", f)
    return len(failures) == 0, report
