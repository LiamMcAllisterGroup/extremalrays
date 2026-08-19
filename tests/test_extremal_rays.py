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
# Description:  Tests for extremal_rays: known cones, degenerate inputs, tie
#               stress, and randomized cross-checks against a brute-force
#               reference implementation.
# -----------------------------------------------------------------------------

# external imports
import numpy as np
import pytest
from scipy.optimize import linprog

# local imports
from extremal_rays import extremal_rays, verify_extremal_rays


def brute_force(R):
    """Reference implementation: test each ray against all others by LP.
    Only viable for small inputs. Requires unique primitive integer rays."""
    R = np.asarray(R, dtype=float)
    ext = []
    for i in range(len(R)):
        others = np.delete(R, i, axis=0)
        res = linprog(
            c=np.zeros(len(others)),
            A_eq=others.T,
            b_eq=R[i],
            bounds=[(0, None)],
            method="highs",
        )
        assert res.status in (0, 2), f"reference LP failed: {res.message}"
        if res.status == 2:  # infeasible: not a combination of the others
            ext.append(i)
    return sorted(ext)


def test_quadrant_2d():
    R = [[0, 1], [1, 3], [1, 1], [3, 1], [1, 0]]
    idx = extremal_rays(R)
    assert sorted(idx) == [0, 4]


def test_duplicates_and_scalings():
    # duplicated directions (exact copies and positive multiples) collapse
    # to the first occurrence
    R = [[1, 0], [2, 0], [1, 1], [0, 3], [0, 1], [1, 0]]
    idx = extremal_rays(R)
    assert sorted(idx) == [0, 3]


def test_zero_ray_dropped():
    R = [[0, 0], [1, 0], [0, 1], [1, 1]]
    idx = extremal_rays(R)
    assert sorted(idx) == [1, 2]


def test_single_ray():
    assert extremal_rays([[2, 3, 5]]).tolist() == [0]
    assert extremal_rays([[2, 3, 5], [4, 6, 10]]).tolist() == [0]


def test_simplicial_3d():
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [2, 1, 3]]
    idx = extremal_rays(R)
    assert sorted(idx) == [0, 1, 2]


def test_all_extremal():
    # cube vertices lifted to the slice x0 = 1: all 8 are extremal
    cube = [(1, a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    idx = extremal_rays(cube)
    assert sorted(idx) == list(range(8))


def test_tie_stress_grid_face():
    # a 5x5 integer grid on a square face: heavy ties under ray shooting;
    # only the 4 corners are extremal
    R = [(1, i, j) for i in range(5) for j in range(5)]
    idx = extremal_rays(R)
    corners = sorted(
        k for k, (_, i, j) in enumerate(R) if i in (0, 4) and j in (0, 4)
    )
    assert sorted(idx) == corners


def test_non_pointed_raises():
    R = [[1, 0], [-1, 0], [0, 1]]
    with pytest.raises(ValueError, match="not pointed"):
        extremal_rays(R)


@pytest.mark.parametrize("trial", range(10))
def test_random_vs_brute_force(trial):
    rng = np.random.default_rng(trial)
    d = rng.integers(3, 6)
    n = rng.integers(10, 50)
    # first coordinate >= 1 guarantees pointedness (w = e_0 works)
    R = np.column_stack(
        [rng.integers(1, 6, n), rng.integers(-5, 6, (n, d - 1))]
    )
    # brute force assumes unique primitive rays
    g = np.gcd.reduce(np.abs(R), axis=1)
    R = np.unique(R // g[:, None], axis=0)

    idx = extremal_rays(R)
    assert sorted(idx.tolist()) == brute_force(R)

    ok, report = verify_extremal_rays(R, idx)
    assert ok, report["failures"]


def test_seeding_disabled_matches():
    rng = np.random.default_rng(7)
    R = np.column_stack(
        [rng.integers(1, 4, 30), rng.integers(-4, 5, (30, 3))]
    )
    g = np.gcd.reduce(np.abs(R), axis=1)
    R = np.unique(R // g[:, None], axis=0)
    a = extremal_rays(R, seed_shots=0)
    b = extremal_rays(R, seed_shots="auto")
    assert a.tolist() == b.tolist()


def test_float_input_integral_values():
    R = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert sorted(extremal_rays(R)) == [0, 2]


def test_verify_catches_wrong_answers():
    R = np.array([[1, 0], [1, 1], [0, 1], [2, 1]])
    # missing an extremal ray
    ok, _ = verify_extremal_rays(R, [0])
    assert not ok
    # including a redundant ray
    ok, _ = verify_extremal_rays(R, [0, 1, 2])
    assert not ok
    # correct answer
    ok, _ = verify_extremal_rays(R, [0, 2])
    assert ok


def _random_pointed_rays(seed, n=40, d=4):
    rng = np.random.default_rng(seed)
    R = np.column_stack(
        [rng.integers(1, 6, n), rng.integers(-5, 6, (n, d - 1))]
    )
    g = np.gcd.reduce(np.abs(R), axis=1)
    return np.unique(R // g[:, None], axis=0)


@pytest.mark.parametrize("trial", range(3))
def test_parallel_matches_serial(trial):
    R = _random_pointed_rays(trial)
    a = extremal_rays(R)
    b = extremal_rays(R, n_workers=2)
    assert a.tolist() == b.tolist()


def test_parallel_tie_stress():
    R = [(1, i, j) for i in range(5) for j in range(5)]
    idx = extremal_rays(R, n_workers=2)
    corners = sorted(
        k for k, (_, i, j) in enumerate(R) if i in (0, 4) and j in (0, 4)
    )
    assert sorted(idx) == corners


def test_checkpoint_resume(tmp_path):
    R = _random_pointed_rays(7)
    ck = str(tmp_path / "state.npz")
    a = extremal_rays(R, checkpoint=ck)
    assert (tmp_path / "state.npz").exists()
    # rerun resumes the completed state and reproduces the result
    b = extremal_rays(R, checkpoint=ck)
    assert a.tolist() == b.tolist()


def test_checkpoint_fingerprint_guard(tmp_path):
    ck = str(tmp_path / "state.npz")
    R1 = _random_pointed_rays(1)
    extremal_rays(R1, checkpoint=ck)
    # different rays with the same path must not resume the stale state
    R2 = _random_pointed_rays(2)
    idx = extremal_rays(R2, checkpoint=ck)
    assert idx.tolist() == extremal_rays(R2).tolist()
