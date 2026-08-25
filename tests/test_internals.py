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
# Description:  Unit tests of internal building blocks: integer detection,
#               primitive deduplication (dense and sparse key packing),
#               positive_functional, the separation oracle, tie-broken ray
#               shooting, the exact rational membership certifier (both the
#               python-flint and the pure-Python Bareiss paths), and the
#               checkpoint fingerprint.
# -----------------------------------------------------------------------------

# stdlib imports
import sys

# external imports
import numpy as np
import pytest
from scipy.sparse import csr_matrix

# local imports
from extremal_rays import core
from extremal_rays.core import (_as_integer, _exact_membership, _fingerprint,
                                _first_unique, _padded_key, _reduce, _shoot,
                                _SeparationOracle, _unique_primitive,
                                _unique_primitive_sparse, positive_functional)
from extremal_rays.inner import _exit_times, _first_exits
from conftest import random_pointed_rays


# --- integer detection -------------------------------------------------------

def test_as_integer():
    assert _as_integer(np.array([[1, 2]])).dtype == np.int64
    assert _as_integer(np.array([[1.0, 2.0]])).tolist() == [[1, 2]]
    assert _as_integer(np.array([[1.0, 2.0 + 1e-12]])).tolist() == [[1, 2]]
    assert _as_integer(np.array([[1.5, 2.0]])) is None
    # beyond the exactly representable float integers
    assert _as_integer(np.array([[1e17, 2.0]])) is None


# --- deduplication -----------------------------------------------------------

def test_unique_primitive_dense():
    R = np.array([[2, 4], [1, 2], [0, 0], [3, -3], [-1, 1], [1, -1]])
    U, rep = _unique_primitive(R)
    assert U.tolist() == [[1, 2], [1, -1], [-1, 1]]
    assert rep.tolist() == [0, 3, 4]
    assert U.dtype == float


def test_unique_primitive_floats_unit_norm():
    R = np.array([[3.0, 4.0], [0.3, 0.4], [1.0, 0.0], [0.0, 0.0]])
    U, rep = _unique_primitive(R)
    assert rep.tolist() == [0, 2]
    assert np.allclose(np.linalg.norm(U, axis=1), 1)


def test_unique_primitive_sparse_matches_dense():
    R = np.array([[2, 0, 4], [1, 0, 2], [0, 0, 0], [0, 3, -3], [0, -1, 1],
                  [5, 0, 0]])
    Ud, repd = _unique_primitive(R)
    Us, reps = _unique_primitive_sparse(csr_matrix(R))
    assert repd.tolist() == reps.tolist()
    assert np.array_equal(Ud, Us.toarray())


def test_unique_primitive_sparse_sign_and_support():
    # rows with the same support but different signs / values are distinct;
    # rows with different supports are distinct
    R = np.array([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 1, 0], [-1, -1, 0]])
    _, rep = _unique_primitive_sparse(csr_matrix(R))
    assert rep.tolist() == [0, 1, 2, 4]


def test_padded_key_distinct_and_ordered():
    R = np.array([[1, 0, 2], [1, 0, -2], [0, 1, 2], [1, 0, 2], [3, 3, 3]])
    U = csr_matrix(R.astype(float))
    K = _padded_key(U)
    assert K.dtype == np.uint64
    rows = [tuple(k) for k in K]
    assert rows[0] == rows[3] and len(set(rows)) == 4
    assert _first_unique(K).tolist() == [0, 1, 2, 4]


def test_first_unique_collision_runs():
    keys = np.array([[1, 5], [1, 6], [1, 5], [2, 0], [1, 6]], dtype=np.uint64)
    assert _first_unique(keys).tolist() == [0, 1, 3]


def test_reduce_dispatch():
    R = random_pointed_rays(1, n=20)
    Ud, repd = _reduce(R)
    Us, reps = _reduce(csr_matrix(R))
    assert repd.tolist() == reps.tolist() and np.array_equal(Ud, Us.toarray())
    with pytest.raises(ValueError, match="2d"):
        _reduce([1, 2, 3])


# --- positive functional -----------------------------------------------------

@pytest.mark.parametrize("trial", range(3))
def test_positive_functional_certificate(trial):
    R = random_pointed_rays(trial, n=60, d=5)
    w = positive_functional(R)
    assert (R @ w).min() >= 1 - 1e-9
    ws = positive_functional(csr_matrix(R))
    assert (R @ ws).min() >= 1 - 1e-9


def test_positive_functional_ignores_zero_rows():
    w = positive_functional(np.array([[1, 0], [0, 0], [0, 1]]))
    assert w.min() >= 1 - 1e-9
    with pytest.raises(ValueError, match="nonzero"):
        positive_functional(np.zeros((2, 2)))


def test_positive_functional_not_pointed():
    with pytest.raises(ValueError, match="not pointed"):
        positive_functional([[1, 1], [-1, -1]])
    with pytest.raises(ValueError, match="not pointed"):
        positive_functional([[1, 0], [-1, 1], [0, -1]])  # 0 interior to conv


def test_positive_functional_badly_scaled_rows():
    R = np.array([[1e-6, 0.0], [0.0, 1e6], [1e3, 1e-3]])
    w = positive_functional(R)
    assert (R @ w).min() >= 1 - 1e-9


# --- separation oracle -------------------------------------------------------

def test_separation_oracle_farkas():
    o = _SeparationOracle(2)
    o.add_row(np.array([1.0, 0.0]), 0)
    o.add_row(np.array([0.0, 1.0]), 1)
    val, _ = o.separate(np.array([0.5, 0.5]))  # inside cone(E)
    assert abs(val) < 1e-9
    q = np.array([-1.0, 0.5])  # outside
    val, c = o.separate(q)
    assert val > 0.5 and c @ q / np.linalg.norm(q) == pytest.approx(val)
    assert (c[0] <= 1e-9) and (c[1] <= 1e-9)  # c . e <= 0 on E
    # the value is scale-free in p and in the rows of E
    assert o.separate(1e-8 * q)[0] == pytest.approx(val)
    o2 = _SeparationOracle(2)
    o2.add_row(np.array([1e6, 0.0]), 0)
    o2.add_row(np.array([0.0, 1e-6]), 1)
    assert o2.separate(1e3 * q)[0] == pytest.approx(val)
    o.relax(0)  # row 0 freed: e_0 is now separable from {e_1}
    val, _ = o.separate(np.array([1.0, 0.0]))
    assert val > 0.5
    o.restore(0)
    val, _ = o.separate(np.array([1.0, 0.0]))
    assert abs(val) < 1e-9


def test_separation_oracle_rebuild_preserves_state():
    o = _SeparationOracle(2)
    o.add_row(np.array([1.0, 0.0]), 0)
    o.add_row(np.array([0.0, 1.0]), 1)
    o.relax(1)
    o._fresh()  # simulate the solver-error recovery path
    val, _ = o.separate(np.array([0.0, 1.0]))
    assert val > 0.5  # relaxation survived the rebuild
    val, _ = o.separate(np.array([1.0, 0.0]))
    assert abs(val) < 1e-9


# --- ray shooting ------------------------------------------------------------

def test_shoot_unique_and_tied():
    P = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=float)
    j, tied = _shoot(P, np.array([0.0, 0.0, 1.0]), np.arange(4))
    assert tied and j == 3  # rows 1 and 3 tie; lexicographic max wins
    j, tied = _shoot(P, np.array([0.0, 1.0, 2.0]), np.arange(4))
    assert not tied and j == 3
    j, tied = _shoot(P, np.array([0.0, -1.0, -1.0]), np.array([1, 2, 3]))
    assert tied and j == 2  # within cand; (1,1,0) > (1,0,1) lexicographically


def test_shoot_sparse_matches_dense():
    P = random_pointed_rays(5, n=30, d=4).astype(float)
    c = np.random.default_rng(0).standard_normal(4)
    cand = np.arange(0, 30, 2)
    assert _shoot(P, c, cand) == _shoot(csr_matrix(P), c, cand)


# --- exact membership --------------------------------------------------------

EXACT_CASES = [
    # (r, rows, lam_float, expected)
    ([2, 3], [[1, 0], [0, 1]], [2.0, 3.0], True),
    ([2, -3], [[1, 0], [0, 1]], [2.0, 3.0], False),   # negative coefficient
    ([1, 1, 1], [[1, 0, 0], [0, 1, 0]], [1.0, 1.0], False),  # inconsistent
    ([2, 2], [[1, 1], [1, 1]], [1.0, 1.0], True),     # free variable -> zero
    ([2, 2], [[1, 1], [-1, -1]], [1.0, 1.0], True),   # x = (2, 0) works
    ([0, 0], [[1, 1], [-1, -1]], [1.0, 1.0], True),   # x = (0, 0)
    ([0, 0], [[1, 0], [0, 1]], [0.0, 0.0], True),     # zero r, empty support
    ([1, 0], [[1, 0], [0, 1]], [0.0, 0.0], False),    # nonzero r, empty support
    ([6, 4], [[3, 2], [1, 1]], [2.0, 0.0], True),     # support excludes a row
    ([10**12, 10**12 + 1], [[1, 0], [0, 1]], [1e12, 1e12], True),  # big ints
    ([1, 1], [[3, 0], [0, 2]], [0.3, 0.5], True),     # rational coefficients
]


def _run_exact(case):
    r, rows, lam, expected = case
    got = _exact_membership(np.array(r), np.array(rows), np.array(lam))
    assert got is expected


@pytest.mark.parametrize("case", EXACT_CASES)
def test_exact_membership(case):
    _run_exact(case)


@pytest.mark.parametrize("case", EXACT_CASES)
def test_exact_membership_bareiss_path(case, monkeypatch):
    # force the pure-Python fallback even when python-flint is installed
    monkeypatch.setitem(sys.modules, "flint", None)
    _run_exact(case)


def test_exact_membership_paths_agree_random():
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.integers(-3, 4, (5, 4))  # 5 generators in dimension 4
        lam = rng.integers(0, 3, 5)
        r = lam @ A
        args = (r, A, lam.astype(float) + 1e-3)
        with_flint = _exact_membership(*args)
        try:
            sys.modules["flint"] = None
            bareiss = _exact_membership(*args)
        finally:
            del sys.modules["flint"]
        assert with_flint == bareiss


# --- fingerprint -------------------------------------------------------------

def test_fingerprint_sensitivity():
    U = random_pointed_rays(1, n=200, d=5).astype(float)
    fp = _fingerprint(U)
    assert fp == _fingerprint(U.copy())
    V = U.copy()
    V[100, 2] += 1  # interior change: outside the edge blocks, caught by mass
    assert _fingerprint(V) != fp
    assert _fingerprint(U[:-1]) != fp
    assert _fingerprint(csr_matrix(U)) == _fingerprint(csr_matrix(U))


# --- inner.py primitives -----------------------------------------------------

def test_exit_times_and_first_exits():
    u = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 1.0]])
    W = np.array([[-1.0, 0.0], [-1.0, -1.0], [-3.0, -1.0]])
    T = _exit_times(u, W)
    assert T[0, 1] == np.inf and T[0, 0] == 1.0 and T[2, 0] == 1.0
    rows, times = _first_exits(T, tie_tol=1e-7)
    assert rows[0] == -1  # rows 0 and 2 tie at t = 1 in column 0
    assert rows[1] == 2 and times[1] == 1.0
    rows, _ = _first_exits(np.full((3, 1), np.inf), tie_tol=1e-7)
    assert rows[0] == -1  # recessive


def test_pool_sweep_verdicts():
    # worker function against a frozen E: redundant rays are certified,
    # separated ones reported as failures
    P = np.array([[1, 0], [0, 1], [0.5, 0.5], [-1, 0.5]], dtype=float)
    core._POOL["P"] = P
    try:
        red, failed = core._pool_sweep(([2, 3], np.array([0, 1]), 1e-7))
    finally:
        core._POOL.clear()
    assert red == [2] and failed == [3]
