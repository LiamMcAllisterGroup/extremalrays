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
from .core import positive_functional, _unique_primitive, _SeparationOracle


def verify(R: ArrayLike,
           ext_indices: ArrayLike,
           tol: float = 1e-6,
           verbose: bool = False) -> "tuple[bool, dict]":
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
    R : array-like of shape (n, dim)
        Matrix whose rows generate the cone.
    ext_indices : array-like of int
        Indices into R of the claimed extremal rays.
    tol : float, optional
        Maximum allowed reconstruction residual for membership certificates.
        Defaults to 1e-6.
    verbose : bool, optional
        Whether to print the worst certificate margins and any failures.
        Defaults to False.

    Returns
    -------
    ok : bool
        True if every certificate checks out.
    report : dict
        The worst membership residual, the worst separation margin, and a
        list of failure descriptions (empty when ok).
    """
    R_in = np.asarray(R)
    U, rep = _unique_primitive(R_in)
    rep_pos = {orig: k for k, orig in enumerate(rep)}
    ext = sorted({rep_pos[i] for i in np.asarray(ext_indices)})

    w = positive_functional(U)
    P = U / (U @ w)[:, None]
    E = P[ext]
    ext_set = set(ext)

    failures = []
    worst_resid = 0.0
    for k in range(len(U)):
        if k in ext_set:
            continue
        res = linprog(
            c=np.zeros(len(E)),
            A_eq=E.T,
            b_eq=P[k],
            bounds=[(0, None)],
            method="highs",
        )
        resid = (float(np.abs(E.T @ res.x - P[k]).max())
                 if res.success else np.inf)
        worst_resid = max(worst_resid, resid)
        if not (res.success and resid < tol):
            failures.append(f"ray {rep[k]}: no membership certificate "
                            f"(residual {resid:.2e})")

    oracle = _SeparationOracle(U.shape[1])
    for e in ext:
        oracle.add_row(P[e], e)
    worst_margin = np.inf
    for e in ext:
        oracle.relax(e)
        val, _c = oracle.separate(P[e])
        oracle.restore(e)
        worst_margin = min(worst_margin, val)
        scale = max(1.0, float(np.abs(P[e]).max()))
        if val <= 1e-7 * scale:
            failures.append(
                f"ray {rep[e]}: no separation certificate (margin {val:.2e}) "
                "-- possibly redundant"
            )

    report = {
        "worst_membership_residual": worst_resid,
        "worst_separation_margin": worst_margin,
        "failures": failures,
    }
    if verbose:
        print(f"worst membership residual: {worst_resid:.2e}")
        print(f"worst separation margin:   {worst_margin:.2e}")
        for f in failures:
            print("FAIL:", f)
    return len(failures) == 0, report
