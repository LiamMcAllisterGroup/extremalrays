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
from __future__ import annotations

# external imports
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog

# local imports
from .core import (_reduce, _row, _rows, _slice, _SeparationOracle,
                   positive_functional)


def verify(R: ArrayLike,
           ext_indices: ArrayLike,
           tol: float = 1e-6,
           time_limit: "float | None" = None,
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

    Parameters
    ----------
    R : array-like or scipy.sparse matrix of shape (n, dim)
        Matrix whose rows generate the cone.
    ext_indices : array-like of int
        Indices into R of the claimed extremal rays (first occurrences, as
        exhaustive() returns them).
    tol : float, optional
        Maximum allowed reconstruction residual for membership certificates.
        Defaults to 1e-6.
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
    rep_pos = {int(orig): k for k, orig in enumerate(rep)}
    ext = set()
    for i in np.asarray(ext_indices, dtype=int).ravel():
        k = rep_pos.get(int(i))
        if k is None:
            raise ValueError(f"index {int(i)} is not a representative row "
                             "(duplicate or zero ray?)")
        ext.add(k)
    ext = sorted(ext)

    w = positive_functional(U)
    P = _slice(U, w)
    E = _rows(P, ext)
    ext_set = set(ext)

    failures = []
    worst_resid = 0.0
    if not ext:  # a nonzero cone has at least one extremal ray
        failures.append("no rays claimed: cone(R) is nonzero")
        worst_resid = np.inf
    for k in range(U.shape[0]):
        if k in ext_set or not ext:
            continue
        pk = _row(P, k)
        res = linprog(
            c=np.zeros(len(E)),
            A_eq=E.T,
            b_eq=pk,
            bounds=[(0, None)],
            method="highs",
            options=None if time_limit is None else {"time_limit": time_limit},
        )
        resid = (float(np.abs(E.T @ res.x - pk).max())
                 if res.success else np.inf)
        worst_resid = max(worst_resid, resid)
        if not (res.success and resid < tol):
            failures.append(f"ray {rep[k]}: no membership certificate "
                            f"(residual {resid:.2e})")

    oracle = _SeparationOracle(U.shape[1])
    for k, e in enumerate(ext):
        oracle.add_row(E[k], e)
    worst_margin = np.inf
    for k, e in enumerate(ext):
        oracle.relax(e)
        val, _c = oracle.separate(E[k])
        oracle.restore(e)
        worst_margin = min(worst_margin, val)
        if val <= 1e-7:  # scale-free: the oracle normalizes its vectors
            failures.append(
                f"ray {rep[e]}: no separation certificate (margin {val:.2e}) "
                "-- possibly redundant"
            )

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
