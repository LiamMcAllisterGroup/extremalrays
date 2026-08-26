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
# Description:  Inner bound on the extremal-ray set: a cheap sampler that
#               certifies rays extremal (never complete) by finding heights
#               in the dual cone {h : R h >= 0} at which exactly one row is
#               tight. Walkers land on a facet by ray shooting, then pursue
#               uncertified rows facet-to-facet across ridges. Work is
#               counted in R-matvecs.
# -----------------------------------------------------------------------------
"""
A cheap inner bound on the extremal-ray set.

:func:`sample` certifies rays extremal without ever claiming completeness,
by working in the dual cone {h : R h >= 0} and finding heights at which
exactly one row is tight, a supporting hyperplane touching a single ray.
Walkers land on a facet by ray shooting from an interior point, then pursue
uncertified rows facet-to-facet across ridges, with work counted in
R-matvecs.

Use it when any witness will do and cheapness matters. When the deliverable
is THE extremal rays, use :func:`extremalrays.exhaustive` instead: this
saturates below the full set and has no reliable coverage estimate.
"""
from __future__ import annotations

# external imports
import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.optimize import linprog

# local imports
from .core import (_CG_BATCH, _CG_ROUNDS, _CG_THRESHOLD, _reduce, _rows,
                   positive_functional)

_RECESSIVE = np.inf  # exit time of a direction that never leaves the cone


def _margin_center(U) -> np.ndarray:
    """
    An interior point of {h : U h >= 0} maximizing the minimum
    row-norm-relative slack over the box |h| <= 1 (fairer angular sampling
    than positive_functional's min-sum point). Tall inputs go through
    constraint generation: a subset LP per round, verified against all
    rows by one matvec (direct HiGHS fails on very tall LPs).
    """
    n, d = U.shape
    if sparse.issparse(U):
        norms = np.sqrt(np.asarray(U.multiply(U).sum(axis=1)).ravel())
    else:
        norms = np.linalg.norm(U, axis=1)

    def solve(idx):
        Ui, ni = U[idx], norms[idx]
        if sparse.issparse(Ui):
            A = sparse.hstack([-Ui, sparse.csr_matrix(ni[:, None])])
        else:
            A = np.column_stack([-Ui, ni])
        c = np.zeros(d + 1)
        c[-1] = -1.0
        res = linprog(c=c, A_ub=A, b_ub=np.zeros(len(idx)),
                      bounds=[(-1, 1)] * d + [(0, None)], method="highs")
        if not res.success or res.x[-1] <= 0:
            raise ValueError("no interior point: cone(R) is not "
                             "full-dimensional or not pointed")
        return res.x[:-1], res.x[-1]

    if n <= _CG_THRESHOLD:
        return solve(np.arange(n))[0]
    rng = np.random.default_rng(0)
    batch = min(_CG_BATCH * d, n)  # a wide cone must not oversample n
    S = np.sort(rng.choice(n, batch, replace=False))
    for _ in range(_CG_ROUNDS):
        h, t = solve(S)
        m = (U @ h) / norms
        if m.min() > 0:
            return h
        worst = np.argpartition(m, min(batch, n - 1))[:batch]
        S = np.union1d(S, worst[m[worst] < t])
    raise RuntimeError(f"margin-center constraint generation did not "
                       f"converge in {_CG_ROUNDS} rounds")


def _exit_times(u: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Per-row exit times along directions, from slacks u > 0 and rates W.

    Row slacks along direction k evolve as u + t*W[:, k]; the entry (i, k)
    is the time row i goes tight (inf if it never does).

    Parameters
    ----------
    u : ndarray of shape (n, m)
        Current positive slacks R @ h, per walker.
    W : ndarray of shape (n, m)
        Slack rates R @ G for the direction matrix G.

    Returns
    -------
    T : ndarray of shape (n, m)
        Exit times, positive or inf.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(W < 0, u / -W, _RECESSIVE)
    return T


def _first_exits(T: np.ndarray,
                 tie_tol: float) -> "tuple[np.ndarray, np.ndarray]":
    """
    First-tight row per column, or -1 for recessive/tied columns.

    Parameters
    ----------
    T : ndarray of shape (n, m)
        Exit times (see _exit_times).
    tie_tol : float
        Relative gap under which the two smallest exit times count as a tie
        (the exit certifies nothing: two rows go tight together).

    Returns
    -------
    rows : ndarray of shape (m,)
        argmin row per column; -1 where recessive or tied.
    times : ndarray of shape (m,)
        The winning exit times (inf where recessive).
    """
    rows = np.argmin(T, axis=0)
    two = np.partition(T, 1, axis=0)[:2]
    times = two[0]
    with np.errstate(invalid="ignore"):  # inf - inf on recessive columns
        gap = two[1] - two[0]
        bad = ~np.isfinite(times) | (gap <= tie_tol * np.maximum(times, 1.0))
    rows = np.where(bad, -1, rows)
    return rows, times


def sample(R: ArrayLike,
           work: int = 1000,
           n_walkers: int = 64,
           stall: int = 20,
           center: str = "margin",
           targeted: bool = True,
           jitter: float = 0.1,
           tie_tol: float = 1e-7,
           rng_seed: int = 0,
           verbosity: int = 0) -> "tuple[np.ndarray, np.ndarray]":
    """
    Certified-extremal rays of cone(R), sampled cheaply (an inner bound).

    Works in the dual cone H = {h : R h >= 0} and certifies a ray by
    exhibiting a height at which exactly its row is tight (a supporting
    hyperplane touching only that ray). Cannot prove completeness, has
    no reliable coverage estimate, and saturates below the full set. Use
    exhaustive() whenever the deliverable is THE extremal rays; this is
    for cheaply picking off certified rays when any witness will do.

    Walkers land on a facet by ray shooting from an interior point, then
    hop facet-to-facet across ridges (2 matvecs per hop, at best one new
    certificate each). With targeted=True each walker pursues one
    uncertified row across hops; fruitless pursuits teleport to a fresh
    shot.

    Parameters
    ----------
    R : array-like or scipy.sparse matrix of shape (n, dim)
        Matrix whose rows generate the cone. Requires rank(R) = dim (H
        pointed), as exhaustive() effectively does.
    work : int, optional
        Budget in R-matvec equivalents (the dominant cost; one shot = 1,
        one hop = 2). Defaults to 1000.
    n_walkers : int, optional
        Concurrent walkers, hopping in lockstep so each round is 2 matmuls
        of n_walkers columns. Defaults to 64.
    stall : int, optional
        Legs before a fruitless pursuit is abandoned (the walker teleports
        to a fresh shot and picks a new target); certifying anything new
        resets the clock. Defaults to 20.
    center : str, optional
        Interior point to shoot from: "margin" (default) maximizes the
        minimum row-norm-relative slack (one LP; fair angular sampling), or
        "functional" reuses positive_functional's min-sum point (cheaper,
        but it sits close to many facets and skews the sampling).
    targeted : bool, optional
        Pursue not-yet-certified rows (direction -r_i plus jitter, never
        recessive since row i's slack strictly decreases) instead of
        hopping at random. Defaults to True.
    jitter : float, optional
        Relative isotropic noise on targeted directions; 0 makes each
        target deterministic. Defaults to 0.1.
    tie_tol : float, optional
        Relative tolerance on exit-time gaps; tied exits certify nothing.
        Defaults to 1e-7.
    rng_seed : int, optional
        Seed; results are deterministic for a fixed value. Defaults to 0.
    verbosity : int, optional
        The verbosity level. Defaults to 0.

    Returns
    -------
    idx : ndarray of shape (n_certified,)
        Sorted indices into R of certified-extremal rays (first occurrence
        for duplicated directions).
    curve : ndarray of shape (n_records, 2)
        Discovery curve: rows (matvecs_spent, n_certified so far).
    """
    U, rep = _reduce(R)
    n, d = U.shape

    if center == "margin":
        h0 = _margin_center(U)
    elif center == "functional":
        h0 = positive_functional(U)
    else:
        raise ValueError(f"unknown center {center!r}")
    u0 = U @ h0
    rng = np.random.default_rng(rng_seed)

    found = set()
    curve = [(0, 0)]
    spent = 0

    def record():
        curve.append((spent, len(found)))
        if verbosity >= 1:
            print(f"  {spent} matvecs: {len(found)} certified")

    # per-walker state; tight < 0 marks a walker shooting from h0
    m = n_walkers
    H = np.tile(h0[:, None], (1, m))
    S = np.tile(u0[:, None], (1, m))
    tight = np.full(m, -1, dtype=int)
    target = np.full(m, -1, dtype=int)
    stalled = np.zeros(m, dtype=int)
    rounds = 0

    def retarget(mask):
        """Assign fresh uncertified targets to the masked walkers."""
        if not targeted or not mask.any():
            return
        cand = np.setdiff1d(np.arange(n), np.fromiter(found, dtype=int),
                            assume_unique=True)
        if len(cand):
            target[mask] = rng.choice(cand, int(mask.sum()))
            stalled[mask] = 0

    retarget(np.ones(m, dtype=bool))

    while spent < work:
        progressed = np.zeros(m, dtype=bool)
        # leg 1: step tangent to the facet; fresh walkers shoot from h0
        G = rng.standard_normal((d, m))
        has_t = target >= 0
        if has_t.any():
            aim = _rows(U, target[has_t])
            aim = aim / np.linalg.norm(aim, axis=1)[:, None]
            G[:, has_t] = -aim.T + jitter * G[:, has_t] / np.sqrt(d)
        on_facet = tight >= 0
        if on_facet.any():
            Ri = _rows(U, tight[on_facet])
            dots = np.einsum("ij,ji->i", Ri, G[:, on_facet])
            G[:, on_facet] -= Ri.T * (dots / (Ri * Ri).sum(axis=1))
        W = U @ G
        spent += m
        T = _exit_times(S, W)
        T[tight[on_facet], np.flatnonzero(on_facet)] = _RECESSIVE
        rows, times = _first_exits(T, tie_tol)

        ok = rows >= 0
        t = np.where(ok, times, 0.0)
        H += G * t
        S += W * t
        S[rows[ok], np.flatnonzero(ok)] = 0.0

        # walkers that shot from h0 land on a facet and certify it now
        landed = ok & ~on_facet
        for k in np.flatnonzero(landed):
            found.add(int(rows[k]))
        tight = np.where(landed, rows, tight)

        # leg 2: at a ridge (i, j), step into facet j's interior
        ridge = ok & on_facet
        if ridge.any():
            kk = np.flatnonzero(ridge)
            i_rows, j_rows = tight[kk], rows[kk]
            Ri, Rj = _rows(U, i_rows), _rows(U, j_rows)
            coef = (Ri * Rj).sum(axis=1) / (Rj * Rj).sum(axis=1)
            E = (Ri - Rj * coef[:, None]).T
            V = U @ E
            spent += len(kk)
            T = _exit_times(np.where(S[:, kk] > 0, S[:, kk], _RECESSIVE), V)
            t_max = T.min(axis=0)
            t2 = np.where(np.isfinite(t_max), t_max / 2, 1.0)
            H[:, kk] += E * t2
            S[:, kk] += V * t2
            S[j_rows, kk] = 0.0
            new = np.array([int(j) not in found for j in j_rows])
            for j in j_rows:
                found.add(int(j))
            tight[kk] = j_rows
            progressed[kk[new]] = True

        # retarget caught pursuits; abandon hopeless ones
        stalled += 1
        stalled[progressed] = 0
        caught = has_t & np.isin(target, np.fromiter(found, dtype=int))
        retarget(caught)
        give_up = ~ok | (stalled > stall)
        if give_up.any():
            H[:, give_up] = h0[:, None]
            S[:, give_up] = u0[:, None]
            tight[give_up] = -1
            retarget(give_up)
            stalled[give_up] = 0

        # rescale (heights are projective); refresh slacks against drift
        scale = np.abs(H).max(axis=0)
        H /= scale
        S /= scale
        rounds += 1
        if rounds % 50 == 0:
            S = U @ H
            spent += m
            S[tight[tight >= 0], np.flatnonzero(tight >= 0)] = 0.0
        record()

    idx = np.array(sorted(found), dtype=int)
    return np.sort(rep[idx]), np.array(curve)
