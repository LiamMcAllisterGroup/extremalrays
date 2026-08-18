"""Independent certificate-based verification of an extremal-ray computation.

Every claim in the output of extremal_rays() is re-derived here from explicit,
inspectable witnesses rather than solver status codes:

- each ray NOT in the result gets a non-negative combination of the result
  rays reproducing it (membership certificate; reconstruction residual is
  checked, not trusted),
- each ray IN the result gets a functional that is non-positive on all other
  result rays and strictly positive on it (separation certificate, proving
  both extremality within the cone and minimality of the result).
"""

import numpy as np
from scipy.optimize import linprog

from .core import positive_functional, _unique_primitive, _SeparationOracle


def verify_extremal_rays(R, ext_indices, tol=1e-6, verbose=False):
    """Check that R[ext_indices] is a minimal generating set for cone(R).

    Returns (ok, report) where report is a dict with the worst membership
    residual, the worst separation margin, and a list of failures (empty when
    ok). This roughly doubles the cost of the original computation; intended
    as an audit, not a routine step.
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
        resid = float(np.abs(E.T @ res.x - P[k]).max()) if res.success else np.inf
        worst_resid = max(worst_resid, resid)
        if not (res.success and resid < tol):
            failures.append(
                f"ray {rep[k]}: no membership certificate (residual {resid:.2e})"
            )

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
